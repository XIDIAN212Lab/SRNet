import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from mmdet.datasets.pipelines.formatting import to_tensor
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmrotate.core import obb2hbb, rotated_anchor_inside_flags
from .rotated_anchor_head import RotatedAnchorHead
from .rotated_retina_head import RotatedRetinaHead
from .utils import get_num_level_anchors_inside
from .SRNet_utils import sample_feature_and_gt_masks, sample_feature, get_mask, sample_feature_and_gt_masksV2
from ..builder import ROTATED_HEADS, build_loss


@ROTATED_HEADS.register_module()
class SRCoarseHead(RotatedAnchorHead):
    r"""An anchor-based head used in `RotatedRetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 align_conv_size=3,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.align_con_size = align_conv_size
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(SRCoarseHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.offset_size = 2 * self.align_con_size ** 2
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)
        self.retina_offset = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.offset_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(self.num_anchors * self.offset_size)
        self.sigmoid = nn.Sigmoid()

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        offset_pred = self.retina_offset(reg_feat)
        offset_pred = self.bn(offset_pred)
        offset_pred = self.sigmoid(offset_pred)
        return cls_score, bbox_pred, offset_pred

    def loss_single(self, cls_score, bbox_pred, offset_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            offset_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'offset_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds, offset_preds):
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds) == len(offset_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0) == offset_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]
        offsets_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 5)
            anchors = mlvl_anchors[lvl]
            offset_pred = offset_preds[lvl]
            offset_pred = offset_pred.permute(0, 2, 3, 1)
            offset_pred = offset_pred.reshape(num_imgs, -1, self.offset_size)

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(anchors, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())
                offset_pred_i = offset_pred[img_id]
                offsets_list[img_id].append(offset_pred_i)

        return bboxes_list, offsets_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            num_level_anchors,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape \
                (num_anchors, 5)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of \
                shape (num_anchors,).
            num_level_anchors (torch.Tensor): Number of anchors of each \
                scale level
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be \
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original \
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of all anchor
                label_weights_list (list[Tensor]): Label weights of all anchor
                bbox_targets_list (list[Tensor]): BBox targets of all anchor
                bbox_weights_list (list[Tensor]): BBox weights of all anchor
                pos_inds (int): Indices of positive anchor
                neg_inds (int): Indices of negative anchor
                sampling_result: object `SamplingResult`, sampling result.
        """
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        if self.assign_by_circumhbbox is not None:
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, num_level_anchors_inside, gt_bboxes_assign,
                gt_bboxes_ignore, None if self.sampling else gt_labels)
        else:
            assign_result = self.assigner.assign(
                anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each \
                image. The outer list indicates images, and the inner list \
                corresponds to feature levels of the image. Each element of \
                the inner list is a tensor of shape (num_anchors, 5).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of \
                each image. The outer list indicates images, and the inner \
                list corresponds to feature levels of the image. Each element \
                of the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be \
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original \
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of \
                    each level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
            additional_returns: This function enables user-defined returns \
                from self._get_targets_single`. These returns are currently \
                refined to properties at each feature map (HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)


@ROTATED_HEADS.register_module()
class BaseRefineHead(RotatedRetinaHead):
    """Rotated Anchor-based refine head. It's a part of the Oriented Detection
    Module (ODM), which produces orientation-sensitive features for
    classification and orientation-invariant features for localization.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='PseudoAnchorGenerator',
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01),
                 **kwargs):
        self.bboxes_as_anchors = None
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(BaseRefineHead, self).__init__(
            num_classes,
            in_channels,
            stacked_convs=2,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.cls_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 5.
        """
        cls_feats, reg_feats = feats
        return multi_apply(self.forward_single, cls_feats, reg_feats)

    def forward_single(self, cls_feat, reg_feat):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 4.
        """
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.cls(cls_feat)
        bbox_pred = self.reg(reg_feat)
        return cls_score, bbox_pred

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            bboxes_as_anchors (list[list[Tensor]]) bboxes of levels of images.
                before further regression just like anchors.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple (list[Tensor]):

                - anchor_list (list[Tensor]): Anchors of each image
                - valid_flag_list (list[Tensor]): Valid flags of each image
        """
        anchor_list = [[
            bboxes_img_lvl.clone().detach() for bboxes_img_lvl in bboxes_img
        ] for bboxes_img in self.bboxes_as_anchors]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             rois=None,
             gt_bboxes_ignore=None):
        """Loss function of ODMRefineHead."""
        assert rois is not None
        self.bboxes_as_anchors = rois[0]
        return super(BaseRefineHead, self).loss(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   rois=None):
        """Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): size / scale info for each image
            cfg (mmcv.Config): test / postprocessing configuration
            rois (list[list[Tensor]]): input rbboxes of each level of
            each image. rois output by former stages and are to be refined
            rescale (bool): if True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (xc, yc, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.
        """
        num_levels = len(cls_scores)
        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None
        # rois = rois[0]
        result_list = []

        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                rois[img_id], img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list


@ROTATED_HEADS.register_module()
class SRRefineHead(BaseRefineHead):

    def forward(self, feats):
        cls_feats, reg_feats = feats
        outs = multi_apply(self.forward_single, cls_feats, reg_feats)
        return outs

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss_box(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            return_sampling_results=True)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, sampling_results_list) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox), sampling_results_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             rois=None,
             gt_bboxes_ignore=None):
        """Loss function of ODMRefineHead."""
        assert rois is not None
        self.bboxes_as_anchors = rois
        return self.loss_box(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds, rois):
        '''
        Args:
            cls_scores (list[Tensors]): 分类得分 per level
            bbox_preds (list[Tensors]): bbox per level
            rois (list[Tensor]): coarse head -> refine bboxes per image
        Returns:
            list[Tensor]: refined bboxes     per image
        '''
        num_levels = len(cls_scores)
        num_imgs = len(rois)

        # image per to level per
        rois_ = [[] for _ in range(num_levels)]
        for i in range(num_levels):
            for j in range(num_imgs):
                rois_[i].append(rois[j][i])
        bboxes_list = [[] for _ in range(num_imgs)]
        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 5)
            anchors = torch.stack(rois_[lvl], 0)
            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                anchors_i = anchors[img_id]
                decode_bbox_i = self.bbox_coder.decode(anchors_i, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())
        return bboxes_list


@ROTATED_HEADS.register_module()
class SRMaskHead(BaseDenseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 num_levels=4,
                 version='oc',
                 output_size=28,
                 finest_scale=56,
                 stacked_convs=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 masks_loss=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super(SRMaskHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.finest_scale = finest_scale
        self.num_levels = num_levels
        self.version = version
        self.output_size = output_size
        self.masks_loss = build_loss(masks_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.feat_channels,
                               out_channels=self.feat_channels,
                               kernel_size=(4, 4),
                               stride=(2, 2),
                               padding=(1, 1)),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.feat_channels,
                      out_channels=self.feat_channels,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)))
        self.logit = nn.Conv2d(self.feat_channels, 1, kernel_size=(1, 1))

    def forward(self, x):
        features = x[: self.num_levels]
        return multi_apply(self.forward_single, features)

    def forward_single(self, x):
        feat = x
        for mask_conv in self.mask_convs:
            feat = mask_conv(feat)
        feat = self.upsample(feat)
        return (feat, )

    def map_roi_levels(self, rois):
        scale = torch.sqrt(rois[:, 2] * rois[:, 3])
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=self.num_levels - 1).long()
        return target_lvls

    def loss(self,
             features,
             rois,
             sampling_results,
             gt_masks,
             img_metas):
        ''' OASDet的mask head训练
        Args:
            features (list[torch.Tensor]): per levels 输入特征图
            sampling_results (list[sampling_result]): per images 采样结果
            gt_masks (list[BitmapMasks]): per images ground truth
            img_metas(list[dict]): per images 输入图像信息
        Returns:
            dict: 损失
        '''
        assert rois is not None
        num_imgs = len(img_metas)
        masks_loss = []
        for idx in range(num_imgs):
            cur_img_metas = img_metas[idx]
            cur_feature = [feature[idx] for feature in features]
            cur_sampling_results = sampling_results[idx]
            cur_pos_assigned_gt_inds = cur_sampling_results.pos_assigned_gt_inds
            # 这里有个问题,实际上这里的pos_bboxes是coarse head的输出,即refine head的正样本anchor
            # 后续将其修改为refine head输出的旋转框
            cur_pos_bboxes = cur_sampling_results.pos_bboxes
            cur_pos_gt_labels = cur_sampling_results.pos_gt_labels
            target_lvls = self.map_roi_levels(cur_pos_bboxes)
            cur_gt_masks = to_tensor(gt_masks[idx].masks).float()[cur_pos_assigned_gt_inds.long(), ...].to("cuda")
            crop_feature_list = []
            crop_gt_masks_list = []
            for i in range(self.num_levels):
                mask_lvl = target_lvls == i
                bboxes_i = cur_pos_bboxes.clone().detach()
                bboxes_i = bboxes_i[mask_lvl]
                if bboxes_i.shape[0] == 0:
                    continue
                cur_feature_i = cur_feature[i]
                crop_feature, crop_gt_masks = sample_feature_and_gt_masks(bboxes_i,
                                                                          cur_feature_i,
                                                                          cur_gt_masks,
                                                                          cur_img_metas,
                                                                          self.version,
                                                                          self.output_size)
                crop_feature_list.append(crop_feature)
                crop_gt_masks_list.append(crop_gt_masks)
            crop_feature_list = torch.cat(crop_feature_list, 0)
            crop_masks_list = self.logit(crop_feature_list)
            crop_gt_masks_list = torch.cat(crop_gt_masks_list, 0)
            masks_loss.append(self.masks_loss(crop_masks_list, crop_gt_masks_list, cur_pos_gt_labels))
        return dict(masks_loss=masks_loss)

    def get_masks(self,
                  features,
                  rois,
                  img_metas,
                  rescale=True,
                  thr=0.5):
        num_imgs = len(img_metas)
        img_bboxes = []
        img_labels = []
        img_masks = []
        for i in range(num_imgs):
            roi = rois[i]
            det_bboxes = roi[0]
            det_labels = roi[1]
            feature_img = [feature[i] for feature in features]
            img_meta = img_metas[i]

            if roi[0].shape[0] == 0:
                img_bboxes.append(det_bboxes)
                img_labels.append(det_labels)
                # img_masks.append([[torch.zeros(img_meta['ori_shape'][:2]).to(torch.bool)]])
                img_masks.append([[]])
                continue
            target_lvls = self.map_roi_levels(det_bboxes)
            bbox_out = []
            label_out = []
            crop_feature_list = []
            for i in range(self.num_levels):
                mask_lvl = target_lvls == i
                bbox_i = det_bboxes[mask_lvl]
                if bbox_i.shape[0] == 0:
                    continue
                label_i = det_labels[mask_lvl]
                bbox_out.append(bbox_i)
                label_out.append(label_i)
                feature_i = feature_img[i]
                crop_feature = sample_feature(bbox_i,
                                              feature_i,
                                              img_meta,
                                              self.version,
                                              self.output_size)
                crop_feature_list.append(crop_feature)
            crop_feature_list = torch.cat(crop_feature_list, 0)
            bbox_out = torch.cat(bbox_out, 0)
            label_out = torch.cat(label_out, 0)
            crop_mask_list = self.logit(crop_feature_list).sigmoid()
            # 需要一个将mask恢复到原位置的函数
            # 还需要将bbox进行rescale,mask也要进行rescale
            mask_out = get_mask(bbox_out,
                                label_out,
                                crop_mask_list,
                                self.version,
                                self.output_size,
                                img_meta,
                                thr,
                                self.num_classes,
                                rescale)
            if rescale:
                scale_factor = img_meta['scale_factor']
                bbox_out[:, :4] = bbox_out[:, :4] / bbox_out.new_tensor(scale_factor)
            img_bboxes.append(bbox_out)
            img_labels.append(label_out)
            img_masks.append(mask_out)
        return list(zip(img_bboxes, img_labels, img_masks))
