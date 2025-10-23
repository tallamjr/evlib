"""YOLOX detection head implementation.

This module implements the YOLOX detection head which performs:
1. Classification prediction (object classes)
2. Regression prediction (bounding box coordinates)
3. Objectness prediction (confidence scores)
4. Loss computation during training
5. Post-processing with NMS during inference

The head uses a decoupled design with separate branches for classification
and regression, following the anchor-free detection paradigm.

Based on the YOLOX paper: "YOLOX: Exceeding YOLO Series in 2021"
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolox_blocks import BaseConv, DWConv

# Try to import torchvision NMS functions
try:
    from torchvision.ops import nms, batched_nms

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    # Fallback NMS implementation will be provided


class IOULoss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, reduction: str = "none", loss_type: str = "iou"):
        """Initialize IoU loss.

        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
            loss_type: Type of IoU loss ('iou', 'giou')
        """
        super(IOULoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            pred: Predicted boxes in (x, y, w, h) format
            target: Target boxes in (x, y, w, h) format

        Returns:
            IoU loss tensor
        """
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # Convert to (x1, y1, x2, y2) format
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            raise NotImplementedError

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def bboxes_iou(
    bboxes_a: torch.Tensor, bboxes_b: torch.Tensor, xyxy: bool = True
) -> torch.Tensor:
    """Compute IoU between two sets of bounding boxes.

    Args:
        bboxes_a: First set of boxes (N, 4)
        bboxes_b: Second set of boxes (M, 4)
        xyxy: Whether boxes are in (x1, y1, x2, y2) format

    Returns:
        IoU matrix of shape (N, M)
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


class YOLOXHead(nn.Module):
    """YOLOX detection head with decoupled classification and regression branches."""

    def __init__(
        self,
        num_classes: int = 2,  # pedestrian + cyclist (no background class in YOLO)
        strides: Tuple[int, int, int] = (8, 16, 32),
        in_channels: Tuple[int, int, int] = (256, 512, 1024),
        act: str = "silu",
        depthwise: bool = False,
    ):
        """Initialize YOLOX head.

        Args:
            num_classes: Number of object classes (without background)
            strides: Strides of the feature maps from FPN
            in_channels: Input channel dimensions from FPN
            act: Activation function name
            depthwise: Whether to use depthwise separable convolutions
        """
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # Whether to decode outputs during inference
        self.strides = strides

        # Initialize module lists for multi-scale heads
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        # Choose convolution type
        Conv = DWConv if depthwise else BaseConv

        # Cache for grid generation
        self.output_strides = None
        self.output_grids = None
        self.grids = [torch.zeros(1)] * len(in_channels)

        # Calculate hidden dimension based on largest input channel
        # YOLOX uses width scaling: out = in[-1]/4 for base model
        largest_base_dim_yolox = 1024
        largest_base_dim_from_input = in_channels[-1]
        width_scale = largest_base_dim_from_input / largest_base_dim_yolox
        hidden_dim = int(256 * width_scale)

        # Build heads for each scale
        for i in range(len(in_channels)):
            # Stem layer to adjust input channels
            self.stems.append(
                BaseConv(
                    in_channels=in_channels[i],
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                    act=act,
                )
            )

            # Classification branch (2 conv layers)
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )

            # Regression branch (2 conv layers)
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )

            # Prediction heads
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=4,  # (x, y, w, h)
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=1,  # objectness
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        # Loss functions
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOULoss(reduction="none")

        # Initialize biases according to Focal Loss paper
        self.initialize_biases(prior_prob=0.01)

    def initialize_biases(self, prior_prob: float):
        """Initialize prediction head biases.

        Args:
            prior_prob: Prior probability for positive samples
        """
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(
        self, fpn_features: List[torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass through detection head.

        Args:
            fpn_features: List of feature maps from FPN [P3, P4, P5]
            labels: Ground truth labels for training (B, max_objects, 5)
                   Format: [class_id, x_center, y_center, width, height]

        Returns:
            Tuple of (predictions, losses)
            - predictions: Decoded detections (B, N, 5+num_classes)
            - losses: Dictionary of loss components (only during training)
        """
        train_outputs = []
        inference_outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        # Process each scale
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, fpn_features)
        ):
            # Apply stem convolution
            x = self.stems[k](x)

            # Split into classification and regression branches
            cls_x = x
            reg_x = x

            # Classification branch
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            # Regression branch
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # Prepare training outputs
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, fpn_features[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(fpn_features[0])
                )

                # Store original predictions for L1 loss if needed
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

                train_outputs.append(output)

            # Prepare inference outputs
            inference_output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            inference_outputs.append(inference_output)

        # Compute losses during training
        losses = None
        if self.training and labels is not None:
            losses = self.get_losses(
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(train_outputs, 1),
                origin_preds,
                dtype=fpn_features[0].dtype,
            )
            losses = {
                "loss": losses[0],
                "iou_loss": losses[1],
                "conf_loss": losses[2],
                "cls_loss": losses[3],
                "l1_loss": losses[4],
                "num_fg": losses[5],
            }

        # Prepare final outputs
        self.hw = [x.shape[-2:] for x in inference_outputs]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in inference_outputs], dim=2
        ).permute(0, 2, 1)

        # Decode outputs during inference
        if self.decode_in_inference:
            return self.decode_outputs(outputs), losses
        else:
            return outputs, losses

    def get_output_and_grid(
        self, output: torch.Tensor, k: int, stride: int, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get output with grid coordinates for loss computation."""
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]

        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid(
                [torch.arange(hsize), torch.arange(wsize)], indexing="ij"
            )
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, grid

    def decode_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Decode raw outputs to final detections."""
        if self.output_grids is None:
            assert self.output_strides is None
            dtype = outputs.dtype
            device = outputs.device
            grids = []
            strides = []

            for (hsize, wsize), stride in zip(self.hw, self.strides):
                yv, xv = torch.meshgrid(
                    [
                        torch.arange(hsize, device=device, dtype=dtype),
                        torch.arange(wsize, device=device, dtype=dtype),
                    ],
                    indexing="ij",
                )
                grid = torch.stack((xv, yv), 2).view(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                strides.append(
                    torch.full((*shape, 1), stride, device=device, dtype=dtype)
                )

            self.output_grids = torch.cat(grids, dim=1)
            self.output_strides = torch.cat(strides, dim=1)

        outputs = torch.cat(
            [
                (outputs[..., 0:2] + self.output_grids) * self.output_strides,
                torch.exp(outputs[..., 2:4]) * self.output_strides,
                outputs[..., 4:],
            ],
            dim=-1,
        )

        return outputs

    def get_losses(
        self,
        x_shifts: List[torch.Tensor],
        y_shifts: List[torch.Tensor],
        expanded_strides: List[torch.Tensor],
        labels: torch.Tensor,
        outputs: torch.Tensor,
        origin_preds: List[torch.Tensor],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute training losses."""
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # Calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt

            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory. " not in str(e):
                        raise

                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx: int,
        num_gt: int,
        gt_bboxes_per_image: torch.Tensor,
        gt_classes: torch.Tensor,
        bboxes_preds_per_image: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        cls_preds: torch.Tensor,
        obj_preds: torch.Tensor,
        mode: str = "gpu",
    ) -> Tuple[torch.Tensor, ...]:
        """Assign ground truth to predictions using SimOTA."""
        if mode == "cpu":
            print("Using CPU for assignment")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = F.one_hot(
            gt_classes.to(torch.int64), self.num_classes
        ).float()
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.amp.autocast("cuda", enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none",
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self,
        gt_bboxes_per_image: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply geometry constraint for anchor assignment."""
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = (
            (x_shifts[0] + 0.5) * expanded_strides_per_image
        ).unsqueeze(0)
        y_centers_per_image = (
            (y_shifts[0] + 0.5) * expanded_strides_per_image
        ).unsqueeze(0)

        # Fixed center radius
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(
        self,
        cost: torch.Tensor,
        pair_wise_ious: torch.Tensor,
        gt_classes: torch.Tensor,
        num_gt: int,
        fg_mask: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SimOTA matching algorithm."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1

        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def get_l1_target(
        self,
        l1_target: torch.Tensor,
        gt: torch.Tensor,
        stride: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Get L1 regression targets."""
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target


def postprocess(
    prediction: torch.Tensor,
    num_classes: int,
    conf_thre: float = 0.7,
    nms_thre: float = 0.45,
    class_agnostic: bool = False,
) -> List[Optional[torch.Tensor]]:
    """Post-process YOLOX predictions with NMS.

    Args:
        prediction: Raw predictions (B, N, 5+num_classes)
        num_classes: Number of object classes
        conf_thre: Confidence threshold
        nms_thre: NMS threshold
        class_agnostic: Whether to apply class-agnostic NMS

    Returns:
        List of detection results per image
    """
    # Convert center format to corner format
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        if not detections.size(0):
            continue

        if TORCHVISION_AVAILABLE:
            if class_agnostic:
                nms_out_index = nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )
        else:
            # Fallback: simple confidence-based filtering (no proper NMS)
            print(
                "Warning: torchvision not available, using simple confidence filtering instead of NMS"
            )
            scores = detections[:, 4] * detections[:, 5]
            sorted_indices = torch.argsort(scores, descending=True)
            # Keep top 100 detections as a simple fallback
            nms_out_index = sorted_indices[:100]

        detections = detections[nms_out_index]

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output
