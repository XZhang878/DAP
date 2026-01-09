import torch
import cv2

from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, PolygonMasks
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from coStudents.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from coStudents.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler
from torch import nn


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        proxy_head = None if cfg.MODEL.STUDENT_DUAL_DA is False else build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )

        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
            if proxy_head is not None:
                proxy_predictor = FastRCNNOutputLayers(cfg, proxy_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
            if proxy_head is not None:
                proxy_predictor = FastRCNNFocaltLossOutputLayers(cfg, proxy_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")
        box_heads = nn.ModuleList([box_head, proxy_head]) if proxy_head is not None else box_head
        box_predictors = nn.ModuleList([box_predictor, proxy_predictor]) if proxy_head is not None else box_predictor


        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_heads,
            "box_predictor": box_predictors,
        }

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            compute_loss=True,
            branch="",
            compute_val_loss=False,
            proposal_index=None,
            file_names=None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        # print(self.gt_classes_img_int)

        self.images = images
        del images
        if (self.training and compute_loss) or branch.split('_')[0] == 'unsupdata':  # apply if training loss
            assert targets
            # 1000 --> 512

            # sup target goes here
            # a = proposals[0].gt_classes
            # c = targets[0].gt_classes
            # print(a.size())
            # print(c.size())
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (branch.startswith("prototype_layer")):
            vis_features, text_features = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return vis_features, text_features

        if (self.training and compute_loss) or compute_val_loss:

            losses, _ = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return proposals, losses
        else:
            if branch.split('_')[0] == 'unsupdata':
                pred_instances, pred_index = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss, branch, file_names=file_names,
                )
                return pred_instances, proposals, pred_index

            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return pred_instances, predictions

    def _forward_box(
            self,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            compute_loss: bool = True,
            compute_val_loss: bool = False,
            branch: str = "",
            proposal_index: List[torch.Tensor] = None,
            file_names=None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        if isinstance(self.box_head, nn.ModuleList):
            box_features, box_features_local = self.box_head[0](box_features)
            # box_features, box_features_local = self.attnpool(box_features)
            predictions = self.box_predictor[0](box_features, use_clip_cls_emb=True, visual_embeddings=box_features_local)
        else:
            box_features, box_features_local = self.box_head(box_features)
            # print(box_features.size())
            # box_features, box_features_local = self.attnpool(box_features)
            predictions = self.box_predictor(box_features, use_clip_cls_emb=True, visual_embeddings=box_features_local)

        if (branch.startswith("prototype_layer")):
            return predictions[2], predictions[3]
        del box_features

        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss
            if isinstance(self.box_head, nn.ModuleList):
                losses = self.box_predictor[0].losses(predictions, proposals)
                for p in self.parameters():
                    losses['loss_cls'] += 0.0 * p.sum()
            else:
                losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            if branch.split('_')[0] == 'unsupdata':
                if branch.split('_')[1] == 'stu':
                    # pred_instances, pred_index = self.box_predictor[0].inference(predictions, proposals)
                    # prev_pred_scores = pred_instances.scores
                    prev_pred_scores = self.box_predictor[0].predict_probs(predictions, proposals)
                    prev_pred_scores = [prev_pred_score.detach() for prev_pred_score in prev_pred_scores]
                    prev_pred_boxes = self.box_predictor[0].predict_boxes(predictions, proposals)
                    # prev_pred_boxes = pred_instances.pred_boxes

                    self.pred_class_img_logits = (
                        self.box_predictor[0].predict_probs_img(predictions, proposals).clone().detach()
                    )
                    if self.sam:
                        self.sam.reset_buffer()
                    # if self.refine_mist:
                    if self.refine_mist:

                        targets = self.get_pgt_mist(
                            prev_pred_boxes,
                            prev_pred_scores,
                            proposals,
                            sam=self.sam,
                            file_names=file_names
                        )
                    else:
                        targets = self.get_pgt_top_k(
                            prev_pred_boxes,
                            prev_pred_scores,
                            proposals,
                            sam=self.sam,
                            file_names=file_names
                        )

                    proposals_k = self.label_and_sample_proposals(proposals, targets, branch=branch)
                    pred_instances, pred_index = self.box_predictor[0].inference(predictions, proposals_k)


                else:
                    # pred_instances, pred_index = self.box_predictor[1].inference(predictions, proposals)
                    prev_pred_scores = self.box_predictor[0].predict_probs(predictions, proposals)
                    prev_pred_scores = [prev_pred_score.detach() for prev_pred_score in prev_pred_scores]
                    prev_pred_boxes = self.box_predictor[0].predict_boxes(predictions, proposals)
                    self.pred_class_img_logits = (
                        self.box_predictor[0].predict_probs_img(predictions, proposals).clone().detach()
                    )
                    if self.sam:
                        self.sam.reset_buffer()
                    if self.refine_mist:
                        targets_1 = self.get_pgt_mist(
                            prev_pred_boxes,
                            prev_pred_scores,
                            proposals,
                            sam=self.sam,
                            file_names=file_names
                        )
                    else:
                        targets_1 = self.get_pgt_top_k(
                            prev_pred_boxes,
                            prev_pred_scores,
                            proposals,
                            sam=self.sam,
                            file_names=file_names
                        )

                    proposals_k = self.label_and_sample_proposals(proposals, targets_1, branch=branch)

                    pred_instances, pred_index = self.box_predictor[0].inference(predictions, proposals_k)


                #return proposals_k
                return pred_instances, pred_index

            else:
                if isinstance(self.box_head, nn.ModuleList):
                    pred_instances, pred_index = self.box_predictor[0].inference(predictions, proposals)
                else:
                    pred_instances, pred_index = self.box_predictor.inference(predictions, proposals)
                # print(proposals)
                return pred_instances, predictions

    @torch.no_grad()
    def get_pgt_mist(
            self,
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_pro=0.15,
            sam=None,
            file_names=None
    ):
        pgt_scores, pgt_boxes, pgt_classes, pgt_weights = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_pro,
            thres=0.05,
            # thres=0.0,
            need_instance=False,
            need_weight=True,
        )

        # NMS
        pgt_idxs = [torch.zeros_like(pgt_class) for pgt_class in pgt_classes]
        keeps = [
            batched_nms(pgt_box, pgt_score, pgt_class, 0.2)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_idxs)
        ]
        pgt_scores = [pgt_score[keep] for pgt_score, keep in zip(pgt_scores, keeps)]
        pgt_boxes = [pgt_box[keep] for pgt_box, keep in zip(pgt_boxes, keeps)]
        pgt_classes = [pgt_class[keep] for pgt_class, keep in zip(pgt_classes, keeps)]
        pgt_weights = [pgt_weight[keep] for pgt_weight, keep in zip(pgt_weights, keeps)]

        # sam refine
        pgt_boxes_old = [None for _ in range(len(self.images))]
        polygons_masks_per_image = [None for _ in range(len(self.images))]
        if sam:
            pgt_boxes_old = [Boxes(pgt_box.clone()) for pgt_box in pgt_boxes]
            bitmasks_per_image = []
            for i, pgt_box in enumerate(pgt_boxes):
                center_x = (pgt_box[:, 0] + pgt_box[:, 2]) / 2
                center_y = (pgt_box[:, 1] + pgt_box[:, 3]) / 2
                width = pgt_box[:, 2] - pgt_box[:, 0]
                height = pgt_box[:, 3] - pgt_box[:, 1]
                width *= 1.1
                height *= 1.1
                # width *= 1.0
                # height *= 1.0
                new_x1 = center_x - width / 2
                new_y1 = center_y - height / 2
                new_x2 = center_x + width / 2
                new_y2 = center_y + height / 2
                x1 = new_x1.clamp(min=0, max=self.images[i].shape[-1])
                y1 = new_y1.clamp(min=0, max=self.images[i].shape[-2])
                x2 = new_x2.clamp(min=0, max=self.images[i].shape[-1])
                y2 = new_y2.clamp(min=0, max=self.images[i].shape[-2])
                input_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
                transformed_boxes = sam.transform.apply_boxes_torch(input_boxes, self.images[i].shape[-2:])
                sam.set_image(
                    self.images[i].cpu().clone().numpy().astype(np.uint8).transpose(1, 2, 0),
                    image_format="BGR",
                    file_name=file_names[i],
                )
                bitmasks, mask_scores, _ = sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    file_name=file_names[i],
                )
                bitmasks = bitmasks.squeeze(1)
                mask_scores = mask_scores.squeeze(1)
                bitmasks_per_image.append(bitmasks)

            def mask_to_polygons(mask):
                # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
                # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
                # Internal contours (holes) are placed in hierarchy-2.
                # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
                mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
                res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                hierarchy = res[-1]
                if hierarchy is None:  # empty mask
                    return [], False
                has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
                res = res[-2]
                res = [x.flatten() for x in res]
                # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
                # We add 0.5 to turn them into real-value coordinate space. A better solution
                # would be to first +0.5 and then dilate the returned polygon by 0.5.
                res = [x + 0.5 for x in res if len(x) >= 6]
                return res, has_holes

            polygons_masks_per_image = [
                [mask_to_polygons(bitmask)[0] for bitmask in bitmasks.cpu().numpy()]
                for bitmasks in bitmasks_per_image
            ]
            polygons_masks_per_image = [PolygonMasks(polygons_masks) for polygons_masks in polygons_masks_per_image]
            pgt_boxes = [polygons_masks.get_bounding_boxes().tensor.to(sam.device) for polygons_masks in
                         polygons_masks_per_image]

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        if sam:
            targets = [
                Instances(
                    proposals[i].image_size,
                    gt_boxes=pgt_box,
                    ori_pgt_boxes=ori_pgt_box,
                    gt_masks=pgt_masks,
                    gt_classes=pgt_class,
                    gt_scores=pgt_score,
                    gt_weights=pgt_weight,
                )
                for i, (pgt_box, ori_pgt_box, pgt_masks, pgt_class, pgt_score, pgt_weight) in enumerate(
                    zip(pgt_boxes, pgt_boxes_old, polygons_masks_per_image, pgt_classes, pgt_scores, pgt_weights)
                )
            ]
        else:
            targets = [
                Instances(
                    proposals[i].image_size,
                    gt_boxes=pgt_box,
                    gt_classes=pgt_class,
                    gt_scores=pgt_score,
                    gt_weights=pgt_weight,
                )
                for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                    zip(pgt_boxes, pgt_classes, pgt_scores, pgt_scores)
                )
            ]

        return targets

    @torch.no_grad()
    def get_pgt_top_k(
            self,
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            # top_k=0.01,
            top_k=1,
            thres=0.05,
            need_instance=True,
            need_weight=True,
            sam=None,
            file_names=None,
    ):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)

        assert isinstance(prev_pred_boxes[0], torch.Tensor)
        num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
        if prev_pred_boxes[0].size(1) == 4:
            prev_pred_boxes = [
                prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
            ]
        else:
            assert (prev_pred_boxes[0].size(1) == self.num_classes * 4) or (
                    prev_pred_boxes[0].size(1) == self.num_classes and prev_pred_boxes[0].size(2) == 4)
            prev_pred_boxes = [
                prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
            ]

        if isinstance(prev_pred_scores, torch.Tensor):
            num_preds_per_image = [len(p) for p in proposals]
            prev_pred_scores = prev_pred_scores.split(num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, tuple) or isinstance(
                prev_pred_scores, list
            ), prev_pred_scores
            assert isinstance(prev_pred_scores[0], torch.Tensor), prev_pred_scores[0]

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        prev_pred_boxes = [
            torch.index_select(prev_pred_box, 1, gt_int)
            for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
        ]

        # print(prev_pred_boxes)
        # filter small pgt
        def get_area(box):

            return (box[:, :, 2] - box[:, :, 0]) * (box[:, :, 3] - box[:, :, 1])


        prev_pred_boxes_keep = [
            get_area(box) > 0
            for box in prev_pred_boxes
        ]

        # a = [torch.unsqueeze(mask, 2).expand(-1, gt_int.numel(), 4)
        # for boxes, mask, gt_int in zip(
        #     prev_pred_boxes, prev_pred_boxes_keep, self.gt_classes_img_int)]
        # print(a[0].size())
        # print(prev_pred_boxes[0].size())
        # print(self.gt_classes_img_int[0].numel())
        # print(gt_int.numel())
        # print(prev_pred_boxes_keep)
        prev_pred_boxes = [
            boxes.masked_select(
                torch.unsqueeze(mask, 2).expand(-1, gt_int.numel(), 4)
            ).view(-1, gt_int.numel(), 4)
            for boxes, mask, gt_int in zip(
                prev_pred_boxes, prev_pred_boxes_keep, self.gt_classes_img_int
            )
        ]
        prev_pred_scores = [
            scores.masked_select(mask).view(-1, gt_int.numel())
            for scores, mask, gt_int in zip(
                prev_pred_scores, prev_pred_boxes_keep, self.gt_classes_img_int
            )
        ]

        # get top k
        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        if top_k >= 1:
            # print(num_preds)
            top_ks = [min(num_pred, int(top_k)) for num_pred in num_preds]
        elif top_k < 1 and top_k > 0:
            # print(num_preds)
            top_ks = [max(int(num_pred * top_k), 1) for num_pred in num_preds]
        else:
            top_ks = [min(num_pred, 1) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, gt_int.numel(), 4)
            for pgt_idx, top_k, gt_int in zip(pgt_idxs, top_ks, self.gt_classes_img_int)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(gt_int, 0).expand(top_k, gt_int.numel())
            for gt_int, top_k in zip(self.gt_classes_img_int, top_ks)
        ]
        if need_weight:
            pgt_weights = [
                torch.index_select(pred_logits, 1, gt_int).expand(top_k, gt_int.numel())
                for pred_logits, gt_int, top_k in zip(
                    self.pred_class_img_logits.split(1, dim=0), self.gt_classes_img_int, top_ks
                )
            ]

        if thres > 0:
            # get large scores
            masks = [pgt_score.ge(thres) for pgt_score in pgt_scores]
            masks = [
                torch.cat([torch.full_like(mask[0:1, :], True), mask[1:, :]], dim=0)
                for mask in masks
            ]
            pgt_scores = [
                torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
            ]
            pgt_boxes = [
                torch.masked_select(
                    pgt_box, torch.unsqueeze(mask, 2).expand(top_k, gt_int.numel(), 4)
                )
                for pgt_box, mask, top_k, gt_int in zip(
                    pgt_boxes, masks, top_ks, self.gt_classes_img_int
                )
            ]
            pgt_classes = [
                torch.masked_select(pgt_class, mask) for pgt_class, mask in zip(pgt_classes, masks)
            ]
            if need_weight:
                pgt_weights = [
                    torch.masked_select(pgt_weight, mask)
                    for pgt_weight, mask in zip(pgt_weights, masks)
                ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        if need_weight:
            pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

            pgt_weights = [
                pgt_weight
                if pgt_weight.numel() > 0
                else torch.tensor([1], dtype=pgt_weight.dtype, device=pgt_weight.device)
                for pgt_weight in pgt_weights
            ]

        pgt_scores = [
            pgt_score
            if pgt_score.numel() > 0
            else torch.tensor([1], dtype=pgt_score.dtype, device=pgt_score.device)
            for pgt_score in pgt_scores
        ]
        pgt_boxes = [
            pgt_box
            if pgt_box.numel() > 0
            else torch.tensor(
                [[-10000, -10000, 10000, 10000]], dtype=pgt_box.dtype, device=pgt_box.device
            )
            for pgt_box in pgt_boxes
        ]
        pgt_classes = [
            pgt_class
            if pgt_class.numel() > 0
            else torch.tensor([0], dtype=pgt_class.dtype, device=pgt_class.device)
            for pgt_class in pgt_classes
        ]

        if not need_instance:
            if need_weight:
                return pgt_scores, pgt_boxes, pgt_classes, pgt_weights
            else:
                return pgt_scores, pgt_boxes, pgt_classes

        # sam refine
        pgt_boxes_old = [None for _ in range(len(self.images))]
        polygons_masks_per_image = [None for _ in range(len(self.images))]
        if sam:
            pgt_boxes_old = [Boxes(pgt_box.clone()) for pgt_box in pgt_boxes]
            bitmasks_per_image = []
            for i, pgt_box in enumerate(pgt_boxes):
                center_x = (pgt_box[:, 0] + pgt_box[:, 2]) / 2
                center_y = (pgt_box[:, 1] + pgt_box[:, 3]) / 2
                width = pgt_box[:, 2] - pgt_box[:, 0]
                height = pgt_box[:, 3] - pgt_box[:, 1]
                width *= 1.1
                height *= 1.1
                # width *= 1.0
                # height *= 1.0
                new_x1 = center_x - width / 2
                new_y1 = center_y - height / 2
                new_x2 = center_x + width / 2
                new_y2 = center_y + height / 2
                x1 = new_x1.clamp(min=0, max=self.images[i].shape[-1])
                y1 = new_y1.clamp(min=0, max=self.images[i].shape[-2])
                x2 = new_x2.clamp(min=0, max=self.images[i].shape[-1])
                y2 = new_y2.clamp(min=0, max=self.images[i].shape[-2])
                input_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
                transformed_boxes = sam.transform.apply_boxes_torch(input_boxes, self.images[i].shape[-2:])
                sam.set_image(
                    self.images[i].cpu().clone().numpy().astype(np.uint8).transpose(1, 2, 0),
                    image_format="BGR",
                    file_name=file_names[i],
                )

                bitmasks, mask_scores, _ = sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    file_name=file_names[i],
                )
                bitmasks = bitmasks.squeeze(1)
                mask_scores = mask_scores.squeeze(1)
                bitmasks_per_image.append(bitmasks)

            def mask_to_polygons(mask):
                # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
                # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
                # Internal contours (holes) are placed in hierarchy-2.
                # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
                mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
                res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                hierarchy = res[-1]
                if hierarchy is None:  # empty mask
                    return [], False
                has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
                res = res[-2]
                res = [x.flatten() for x in res]
                # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
                # We add 0.5 to turn them into real-value coordinate space. A better solution
                # would be to first +0.5 and then dilate the returned polygon by 0.5.
                res = [x + 0.5 for x in res if len(x) >= 6]
                return res, has_holes

            polygons_masks_per_image = [
                [mask_to_polygons(bitmask)[0] for bitmask in bitmasks.cpu().numpy()]
                for bitmasks in bitmasks_per_image
            ]
            polygons_masks_per_image = [PolygonMasks(polygons_masks) for polygons_masks in polygons_masks_per_image]
            for i, polygons_masks in enumerate(polygons_masks_per_image):
                pgt_box = polygons_masks.get_bounding_boxes().tensor.to(sam.device)
                inf_indices = torch.any(pgt_box == float('inf'), dim=1).nonzero(as_tuple=True)[0]
                pgt_box[inf_indices] = pgt_boxes[i][inf_indices]
                pgt_boxes[i] = pgt_box

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]
        if sam:
            if need_weight:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        ori_pgt_boxes=ori_pgt_box,
                        gt_masks=pgt_masks,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                        gt_weights=pgt_weight,
                    )
                    for i, (pgt_box, ori_pgt_box, pgt_masks, pgt_class, pgt_score, pgt_weight) in enumerate(
                        zip(pgt_boxes, pgt_boxes_old, polygons_masks_per_image, pgt_classes, pgt_scores, pgt_weights)
                    )
                ]
            else:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        pgt_boxes=ori_pgt_box,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                    )
                    for i, (pgt_box, ori_pgt_box, pgt_class, pgt_score) in enumerate(
                        zip(pgt_boxes, pgt_boxes_old, pgt_classes, pgt_scores)
                    )
                ]
        else:
            if need_weight:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                        gt_weights=pgt_weight,
                    )
                    for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                        zip(pgt_boxes, pgt_classes, pgt_scores, pgt_weights)
                    )
                ]
            else:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                    )
                    for i, (pgt_box, pgt_class, pgt_score) in enumerate(
                        zip(pgt_boxes, pgt_classes, pgt_scores)
                    )
                ]

        return targets

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            # print(matched_idxs.size())
            # print(matched_labels.size())
            # print(targets_per_image.gt_classes.size())
            # print(gt_classes.size())

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                            trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        # a = proposals_with_gt[0].gt_classes
        # print(a.size())
        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt

    @torch.no_grad()
    def label_and_sample_proposals_1(
            self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # if has_gt and not self.cls_agnostic_bbox_known:
            if has_gt:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_scores"):
                proposals_per_image.gt_scores = targets_per_image.gt_scores[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_weights"):
                proposals_per_image.gt_weights = targets_per_image.gt_weights[
                    matched_idxs[sampled_idxs]
                ]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + branch, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + branch, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + branch, np.mean(num_ig_samples))

        return proposals_with_gt

@torch.no_grad()
def get_image_level_gt(targets, num_classes):
    if targets is None:
        return None, None, None
    gt_classes_img = [torch.unique(t.gt_classes, sorted=True) for t in targets]
    gt_classes_img_int = [gt.to(torch.int64) for gt in gt_classes_img]
    # print(targets)
    gt_classes_img_oh = torch.cat(
        [
            torch.zeros(
                (1, num_classes), dtype=torch.float, device=gt_classes_img[0].device
            ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
            for gt in gt_classes_img_int
        ],
        dim=0,
    )

    return gt_classes_img, gt_classes_img_int, gt_classes_img_oh
