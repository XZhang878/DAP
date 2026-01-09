import numpy as np
import torch
import torch.nn as nn
import copy
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList
import cv2

def data2labels(data):
    labels = []
    for i in range(len(data)):
        labels_i = data[i]['instances'].gt_classes
        # labels_i = data[i].gt_classes
        if labels_i.shape[0]:
            labels.append(labels_i)
    labels = torch.cat(labels, dim=0)
    return labels
#######################
###############  Prototype Network  ##################
class Prtotype_Net(nn.Module):
    def __init__(self, output_shape = 128, ndf1=512):
        super(Prtotype_Net, self).__init__()

        self.linear1 = nn.Linear(1024, ndf1)
        self.linear2 = nn.Linear(ndf1, output_shape)
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        #print("Protoproto")
        return x
#################################
@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            prototype_layer: int,
            contra: int,
            num_classes: int
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.num_classes = num_classes
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.contra = contra
        self.prototype_layer = prototype_layer
        if self.prototype_layer:
            self.proto = Prtotype_Net()

        self.criterion = torch.nn.MSELoss()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "prototype_layer": cfg.SEMISUPNET.PROTOTYPE_LAYER,
            "contra": cfg.SEMISUPNET.USE_CONTRA,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,

        }

    def build_prototype(self):
        self.prototype_s = torch.zeros((self.num_classes, 128), device=torch.device('cuda'))
        self.prototype_s = self.prototype_s
        self.number_of_occurance_s = [0] * self.num_classes

        self.prototype_t = torch.zeros((self.num_classes, 128), device=torch.device('cuda'))
        self.number_of_occurance_t = [0] * self.num_classes
        self.prototype_t = self.prototype_t
        self.prototype_t_c = torch.zeros((self.num_classes, 128), device=torch.device('cuda'))
        self.number_of_occurance_c = [0] * self.num_classes
        self.prototype_t_c = self.prototype_t_c
        self.prototype_s_c = torch.zeros((self.num_classes, 128), device=torch.device('cuda'))
        self.prototype_s_c = self.prototype_s_c

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def convert_gt_to_rcn(self, gt):
        temp = copy.deepcopy(gt)
        for item in temp:
            item.set('objectness_logits', torch.ones(len(item)).to(self.device))
            item.set('proposal_boxes', item.get('gt_boxes'))
            item.remove('gt_classes')
            item.remove('gt_boxes')
        return temp

    def forward(
            self, batched_inputs, branch="supervised_stu", given_proposals=None, val_mode=False, proposal_index=None
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)


        if (branch == "prototype_layer"):

            self.build_prototype()

            source, target = batched_inputs
            # print(target)

            images_s = self.preprocess_image(source)
            images_t = self.preprocess_image(target)

            gt_instances_s = [x["instances"].to(self.device) for x in source]
            gt_instances_t = [x["instances"].to(self.device) for x in target]
            gt_labels_s = data2labels(source)
            gt_labels_t = data2labels(target)

            features_s = self.backbone(images_s.tensor)
            features_t = self.backbone(images_t.tensor)

            # Change GT to RPN type, expected by detectron2
            proposals_rpn_s = self.convert_gt_to_rcn(gt_instances_s)
            proposals_rpn_t = self.convert_gt_to_rcn(gt_instances_t)


            # Output is box feature only, due to argument branch.. Check roi_heads code...
            box_features_s, text_features_s = self.roi_heads(
                images_s,
                features_s,
                proposals_rpn_s,
                compute_loss=True,
                targets=gt_instances_s,
                branch=branch,
            )

            box_features_t, text_features_t = self.roi_heads(
                images_t,
                features_t,
                proposals_rpn_t,
                compute_loss=True,
                targets=gt_instances_t,
                branch=branch,
            )

            if (self.prototype_layer):
                box_features_s = self.proto(box_features_s)
                box_features_t = self.proto(box_features_t)
            for lab, pro in zip(gt_labels_t, box_features_t):
                self.prototype_t[lab] = (self.prototype_t[lab] * self.number_of_occurance_t[lab] + pro) / (
                        self.number_of_occurance_t[lab] + 1)
                self.number_of_occurance_t[lab] += 1
                # print(gt_labels_t)
                v_t = self.number_of_occurance_t[lab] / len(gt_labels_t)
                # print(v_t)
                # v_t = self.number_of_occurance_t[lab] / sum(self.number_of_occurance_t)
                if (self.contra):
                    self.prototype_t_c[lab] = v_t * self.prototype_t_c[lab] + (1 - v_t) * self.prototype_t[lab]


            for lab, pro in zip(gt_labels_s, box_features_s):
                self.prototype_s[lab] = ((self.prototype_s[lab] * self.number_of_occurance_s[lab]) + pro) / (
                        self.number_of_occurance_s[lab] + 1)
                self.number_of_occurance_s[lab] += 1
                # print(self.number_of_occurance_s)
                # v_s = self.number_of_occurance_s[lab] / sum(self.number_of_occurance_s)
                # v_s = F.softmax(self.number_of_occurance_s)
                v_s = self.number_of_occurance_s[lab] / len(gt_labels_s)

                # print(v_s)
                if (self.contra):
                    # self.prototype_c[lab] = ((self.prototype_c[lab] * self.number_of_occurance_c[lab]) + pro) / (
                    #             self.number_of_occurance_c[lab] + 1)
                    # self.number_of_occurance_c[lab] += 1
                    self.prototype_s_c[lab] = v_s * self.prototype_s_c[lab] + (1 - v_s) * self.prototype_s[lab]
                    # self.prototype_s_c[lab] = (self.prototype_s[lab] * self.number_of_occurance_s[lab])/(sum(self.number_of_occurance_s))

            loss_pro = F.cosine_similarity(self.prototype_s, self.prototype_t).abs().mean().to(self.device)
            loss = -loss_pro
            # print(loss)
            self.prototype_t.detach_()
            self.prototype_s.detach_()

            if self.contra:
                loss_c = F.cosine_similarity(self.prototype_s_c, self.prototype_t_c).abs().mean().to(self.device)
                # print(loss_c)
                self.prototype_s_c.detach_()
                self.prototype_t_c.detach_()

            text_features_s = text_features_s.mean(axis=0, keepdim=True).detach()
            text_features_t = text_features_t.mean(axis=0, keepdim=True).detach()

            text_embedding_s = F.normalize(text_features_s, p=2.0, dim=1)
            text_embedding_t = F.normalize(text_features_t, p=2.0, dim=1)
            text_loss = self.criterion(text_embedding_s, text_embedding_t)


            return loss, loss_c, text_loss



        source_label = 0
        target_label = 1

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        if "file_name" in batched_inputs[0]:
            file_names = [i['file_name'] for i in batched_inputs]
        else:
            file_names = None
        features = self.backbone(images.tensor)

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised_stu":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None


        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            # print(proposals_rpn)
            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch.split("_")[0] == "unsupdata":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            proposals_roih, proposals_into_roih, proposal_index = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=gt_instances,
                compute_loss=False,
                branch=branch,
                file_names=file_names
            )
            return proposal_losses, proposals_into_roih, proposals_rpn, proposals_roih, proposal_index

        elif branch == "val_loss":
            raise NotImplementedError()

    def convert_image_to_rgb(image, input_format):
        if input_format == 'BGR':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif input_format == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif input_format == 'RGB':
            return image
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = self.convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                    "Left: GT bounding boxes "
                    + branch
                    + ";  Right: Predicted proposals "
                    + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch