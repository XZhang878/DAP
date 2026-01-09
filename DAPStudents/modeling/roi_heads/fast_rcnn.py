import logging
from typing import Dict, List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import (
    ShapeSpec,
    batched_nms,
    cat,
    ciou_loss,
    cross_entropy,
    diou_loss,
    nonzero_tuple,
)
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from .domain_aware_prompt import DAPromptHead
from timm.models.layers import trunc_normal_
import clip

__all__ = ["fast_rcnn_inference", "FastRCNNOutputLayers"]


def fast_rcnn_inference(
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        scores_flag: bool,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    # print(result_per_image)

    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


def fast_rcnn_inference_single_image(
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    # save the scores for later
    scores_all = torch.clone(scores)

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    if scores.numel() == 0:
        avg_score = torch.tensor(0.0)
        max_score = torch.tensor(0.0)  # 你需要的默认值
        thres_score = avg_score + max_score
    else:
        avg_score = scores.mean()
        max_score = scores.max()
        thres_score = (avg_score + max_score) / 2.0
    # max_score = scores.max()


    # print(avg_score)
    # score_thresh = min(score_thresh, avg_score)
    score_thresh = min(score_thresh, thres_score)

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    # print(filter_mask)

    filter_inds = filter_mask.nonzero()

    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # filter scores_all
    scores_all = scores_all[filter_inds[:, 0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    # index scores_all
    scores_all = scores_all[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.full_scores = scores_all
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

class DeformableAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., offset_dim=32, num_offsets=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Projection to get offset in (B, N, num_offsets * 2)
        self.offset_proj = nn.Linear(dim, num_offsets * 2)
        self.num_offsets = num_offsets

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape

        H = self.num_heads
        D = C // H

        # Project q, k, v
        q_proj = self.q_proj(q).view(B, N, H, D).transpose(1, 2)
        k_proj = self.k_proj(k).view(B, M, H, D).transpose(1, 2)
        v_proj = self.v_proj(v).view(B, M, H, D).transpose(1, 2)

        # Calculate offsets
        offsets = self.offset_proj(q)  # (B, N, num_offsets * 2)
        D = offsets.size(1)
        # Reshape to (B, 16, num_offsets, 2) to match the offsets shape
        offsets = offsets.view(B, D, self.num_offsets, 2)  # Reshape to (B, 16, num_offsets, 2)

        # Ensure the total size of offsets matches
        assert offsets.numel() == B * D * self.num_offsets * 2, f"Offsets shape mismatch: {offsets.numel()} != {B * 16 * self.num_offsets * 2}"

        # Apply offsets to k_proj
        offset_k_projs = []
        for i in range(self.num_offsets):
            offset = offsets[:, :, i, :]  # (B, 16, 2)
            offset_k_proj = self.apply_offset(k_proj, offset, M)
            offset_k_projs.append(offset_k_proj)

        # Average over offsets
        offset_k_projs = torch.stack(offset_k_projs, dim=3).mean(dim=3)

        # Attention computation
        attn = torch.einsum('bhnd,bhnd->bhn', q_proj, offset_k_projs) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of v_proj
        x = torch.einsum('bhn,bhmd->bhnd', attn, v_proj).transpose(1, 2).reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def apply_offset(self, k_proj, offset, M):
        """Applies offsets to the key projections using grid_sample."""
        B, H, N, D = k_proj.shape

        # Reshape k_proj for grid_sample: (B * H, D, 1, M)
        k_proj = k_proj.reshape(B * H, D, 1, M)
        E = offset.size(1)

        # Normalize the grid coordinates to [-1, 1] range
        grid = offset.view(B, E, 1, 2)  # (B, 16, 1, 2)

        # Normalize grid to [-1, 1] range
        #grid = grid / (M - 1) * 2 - 1  # Scale to [-1, 1]
        # Normalize grid to [-1, 1] range
        if M > 1:
            grid = grid / (M - 1) * 2 - 1  # Scale to [-1, 1]
        else:
            # Handle M = 1 case: skip normalization or set a default value
            # For example, skip normalization:
            pass
        # Expand grid for all heads
        grid = grid.repeat(1, 1, H, 1).view(B * H, E, 1, 2)

        # Apply grid_sample
        offset_k_proj = F.grid_sample(k_proj, grid, align_corners=True)

        # Restore shape and transpose to (B, H, D, N)
        offset_k_proj = offset_k_proj.view(B, H, D, E).transpose(2, 3)

        return offset_k_proj



class GatedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = DeformableAttention(dim, num_heads)
        self.gate = nn.Parameter(torch.ones(1))  # 可学习门控参数

    def forward(self, q, k, v):
        attn_out = self.attn(q, k, v)
        return self.gate * attn_out  # 使用 gate 控制特征融合
class SwiGLU(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        self.fc1 = nn.Linear(d_model * expansion_factor, d_model)
        self.fc2 = nn.Linear(d_model, d_model * expansion_factor)
        self.gate = nn.Linear(d_model * expansion_factor, d_model)

    def forward(self, x):
        #a = self.gate(x)
        #b = F.silu(self.fc1(x))
        return self.fc2(F.silu(self.fc1(x)) * self.gate(x))  # SwiGLU 激活函数
class MultiScaleProjection(nn.Module):
    def __init__(self, visual_dim, transformer_width):
        super().__init__()
        self.proj1 = nn.Linear(visual_dim, transformer_width)
        self.proj2 = nn.Linear(visual_dim // 2, transformer_width)
        self.proj3 = nn.Linear(visual_dim // 4, transformer_width)

    def forward(self, visual):
        v1 = self.proj1(visual)
        v2 = self.proj2(F.adaptive_avg_pool2d(visual, (visual.shape[1] // 2, visual.shape[2] // 2)))
        v3 = self.proj3(F.adaptive_avg_pool2d(visual, (visual.shape[1] // 4, visual.shape[2] // 4)))
        return v1 + v2 + v3  # 多尺度视觉特征融合


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = GatedCrossAttention(d_model, nhead)
        self.cross_attn = GatedCrossAttention(d_model, nhead)  # 使用 GatedCrossAttention

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            SwiGLU(d_model),  # 使用 SwiGLU 激活函数
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x
class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=512,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        visual_dim = 1024
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        #self.multi_scale_proj = MultiScaleProjection(visual_dim, transformer_width)  # 使用多尺度特征融合

        self.decoder = nn.ModuleList([TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)  # B, N, C
        # visual = self.multi_scale_proj(visual)  # 多尺度特征融合
        x = self.text_proj(text)  # 2K, 77, C

        for layer in self.decoder:
            x = layer(x, visual)

        return self.out_proj(x)

class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            clip_cls_emb: bool = False,
            ctx_size: int = 8,
            prompt_class: tuple = (None),
            context_feature='attention',
            use_visual_prompt_generator: bool = True,
            use_text_prompt_generator: bool = True,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.context_feature = context_feature
        self.use_visual_prompt_generator = use_visual_prompt_generator
        self.use_text_prompt_generator = use_text_prompt_generator
        self.context_decoder = ContextDecoder()  # PromptGeneratorWithDecoder(cfg) #ContextDecoder(cfg)
        text_emb_require_grad = False
        self.use_bias = False

        # background embedding
        self.cls_bg_score = nn.Linear(input_size, 1, bias=self.use_bias)
        with torch.no_grad():
            nn.init.constant_(self.cls_bg_score.weight, 0)  # zero embeddings
            self.cls_bg_score.weight.requires_grad = text_emb_require_grad
            if self.use_bias:
                nn.init.constant_(self.cls_bg_score.bias, 0)

        #clip
        self.use_clip_cls_emb = clip_cls_emb
        if self.use_clip_cls_emb:  # use CLIP text embeddings as classifier's weights

            # self.temperature = 0.01  # 0.01 is default for CLIP
            self.temperature = 0.008  # 0.01 is default for CLIP

            ######################################
            # learnable prompt embeddings
            self.clip_model, self.preprocess = clip.load('RN50', 'cuda', jit=False)
            self.clip_model.eval()
            self.ctx_size = ctx_size

            for params in self.clip_model.parameters():
                params.requires_grad_(False)
            self.DAHead = DAPromptHead(prompt_class, self.clip_model, self.ctx_size)
            #######################################

        else:  # regular classification layer
            self.cls_score = nn.Linear(input_size, num_classes + 1)  # one background class (hence + 1)
            nn.init.normal_(self.cls_score.weight, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # RegionCLIP
            "clip_cls_emb": cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
            "ctx_size": cfg.LEARNABLE_PROMPT.CTX_SIZE,
            "prompt_class": cfg.LEARNABLE_PROMPT.CLASS,
            # fmt: on
        }

    def forward(self, x, use_clip_cls_emb, visual_embeddings):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
            # use clip text embeddings as classifier's weights
        if use_clip_cls_emb:
            B, C, H, W = visual_embeddings.shape
            if self.context_feature == 'attention':
                # (B, C, 1+H*W)
                visual_contexts = torch.cat([x.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                            dim=2).permute(0, 2, 1)  # B, (1+H*W), C


            #normalized_x = F.normalize(x, p=2.0, dim=1)
            text_embedding, text_contexts = self.DAHead.get_embedding()  # [domains * (cls), 1024]
            # text_embedding = F.normalize(text_embedding, p=2.0, dim=1)
            text_embedding = text_embedding.expand(B, -1, -1)  # B, K, C
            text_contexts = text_contexts.expand(B, -1, -1, -1)[:, 0, :self.DAHead.prompt_learner.n_ctx, :]  # B, L, C
            if self.use_visual_prompt_generator:
                # update visual_embeddings by text_context, post-model prompting refines the visual_embeddings
                # visual_embeddings: # (B, 1, C) text_contexts: B, (L-1), C
                vis_prompt_diff = self.context_decoder(x.reshape(B, C, 1).permute(0, 2, 1), text_contexts)
                vis_prompt_diff = vis_prompt_diff.permute(0, 2, 1).reshape(B, C)
                updated_vision_embedding = x + self.DAHead.prompt_learner.gamma_v * vis_prompt_diff
                # update text prompting
            if self.use_text_prompt_generator:
                # update text_embeddings by visual_context, post-model prompting refines the text_embeddings
                # text_embeddings: # (B, K, C) visual_contexts: B, (1+H*W), C
                text_diff = self.context_decoder(text_embedding, visual_contexts)
                # (B, K, C)
                updated_text_embeddings = text_embedding + self.DAHead.prompt_learner.gamma_t * text_diff

            normalized_x = F.normalize(updated_vision_embedding, p=2.0, dim=1)
            text_embedding = F.normalize(updated_text_embeddings, p=2.0, dim=2)
            cls_scores_total = torch.einsum('bc,bkc->bk', normalized_x, text_embedding)


            #cls_scores_total = normalized_x @ text_embedding.t()
            cls_scores = cls_scores_total[:, :self.num_classes]
            # EMA embeddings
            # text_embedding_ema, _ = self.DAHead.get_embedding_ema()  # [domains * (cls), 1024]
            # text_embedding_ema = text_embedding_ema.detach()
            # text_embedding_ema = F.normalize(text_embedding_ema, p=2.0, dim=1)
            # ema_cls_scores_total = normalized_x @ text_embedding_ema.t()
            # ema_cls_scores = ema_cls_scores_total[:, :self.num_classes]
            # background class (zero embeddings)

            bg_score = self.cls_bg_score(normalized_x)
            if self.use_bias:
                bg_score += self.cls_bg_score.bias

            # scores = torch.cat((cls_scores, bg_score), dim=1)
            scores = torch.cat((cls_scores, bg_score), dim=1)
            scores = scores / self.temperature

            # EMA scores
            # ema_scores = torch.cat((ema_cls_scores, bg_score), dim=1)
            # ema_scores = ema_scores / self.temperature

            proposal_deltas = self.bbox_pred(x)
            return scores, proposal_deltas, updated_vision_embedding, updated_text_embeddings

                # regular classifier
        else:
            scores = self.cls_score(x)
            proposal_deltas = self.bbox_pred(x)
            return scores, proposal_deltas, x, x
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, _, _ = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)
        # print(len(proposals))
        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            # if temperature:
            # loss_box_reg = smooth_l1_loss(fg_pred_deltas, gt_pred_deltas, self.smooth_l1_bet, reduction="sum")
            # loss_box_reg=torch.sum(loss_box_reg,dim=1)
            # loss_box_reg = loss_box_reg*temperature
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        elif self.box_reg_loss_type == "diou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = diou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        elif self.box_reg_loss_type == "ciou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = ciou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances], scores_flag=False,
                  proposal_index=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        if proposal_index:
            predictions = []
            for i in range(len(proposals)):
                inst = Instances(image_shapes[0])
                inst.full_scores = scores[i][proposal_index[i], :]
                inst.pred_classes = torch.max(scores[i][proposal_index[i], :-1], axis=1).indices

                num_bbox_reg_classes = boxes[i].shape[1] // 4
                new_boxes = Boxes(boxes[i].reshape(-1, 4))
                new_boxes.clip(image_shapes[i])
                new_boxes = new_boxes.tensor.view(-1, num_bbox_reg_classes, 4)
                inst.pred_boxes = new_boxes[proposal_index[i], inst.pred_classes]

                predictions.append(inst)

            return predictions, []
        # ADD filtering for teacher here if needed

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            scores_flag
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas, _, _ = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas, _, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _, _, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        # probs = scores
        # print(probs)

        return probs.split(num_inst_per_image, dim=0)

    def predict_probs_img(self, predictions, proposals):
        scores, _, _, _ = predictions
        if len(proposals) == 1:
            pred_class_img_logits = torch.sum(scores, dim=0, keepdim=True)
        else:
            num_inst_per_image = [len(p) for p in proposals]
            pred_class_img_logits = cat(
                [
                    torch.sum(score, dim=0, keepdim=True)
                    for score in scores.split(num_inst_per_image, dim=0)
                ],
                dim=0,
            )
        pred_class_img_logits = torch.clamp(pred_class_img_logits, min=1e-6, max=1.0 - 1e-6)
        return pred_class_img_logits


class FastRCNNOutputs:
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self,
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # "gt_classes" exists if and only if training. But other gt fields may
            # not necessarily exist in training for images that have no groundtruth.
            if proposals[0].has("gt_classes"):
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = [
                    p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes for p in proposals
                ]
                self.gt_boxes = box_type.cat(gt_boxes)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(self.proposals) == 0  # no instances found

    def softmax_cross_entropy_loss(self):
        """
        Deprecated
        """
        _log_classification_stats(self.pred_class_logits, self.gt_classes)
        return cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def box_reg_loss(self):
        """
        Deprecated
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.proposals.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds should produce a valid loss of zero because reduction=sum.
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * self.gt_classes[fg_inds, None] + torch.arange(
                box_dim, device=device
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                self.proposals.tensor[fg_inds],
            )
            loss_box_reg = giou_loss(
                fg_pred_boxes,
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Deprecated
        """
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}

    def predict_boxes(self):
        """
        Deprecated
        """
        pred = self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
        return pred.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)


# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()

        return losses


class FastRCNNFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self,
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
            num_classes=80,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes

    def losses(self):
        return {
            "loss_cls": self.comput_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
            total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            gamma=1.0,
            num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()
