from mmrotate.core import rbbox2result, imshow_det_rbboxes
from mmrotate.models.builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from mmrotate.models.detectors.base import RotatedBaseDetector
from .utils import DoubleAlignModule


@ROTATED_DETECTORS.register_module()
class SRNet(RotatedBaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 coarse_head=None,
                 refine_head=None,
                 mask_head=None,
                 align_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SRNet, self).__init__()
        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            coarse_head.update(train_cfg=train_cfg['coarse_cfg'])
            refine_head.update(train_cfg=train_cfg['refine_cfg'])
        coarse_head.update(test_cfg=test_cfg)
        refine_head.update(test_cfg=test_cfg)
        mask_head.update(test_cfg=test_cfg)
        self.coarse_head = build_head(coarse_head)
        self.refine_head = build_head(refine_head)
        self.mask_head = build_head(mask_head)
        self.align_conv_type = align_cfg['type']
        self.align_conv_size = align_cfg['kernel_size']
        self.feat_channels = align_cfg['channels']
        self.featmap_strides = align_cfg['featmap_strides']

        if self.align_conv_type == 'DoubleAlignModule':
            self.align_conv = DoubleAlignModule(self.feat_channels,
                                                self.featmap_strides,
                                                self.align_conv_size)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        ''' 从backbone和neck提取特征
        Args:
            img (torch.Tensor): (batch, 3, img_w, img_h)
        Returns:
            tuple : 特征图,最细尺度用于分割,其他尺度用于旋转框检测
        '''
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x[0], x[1:]

    def forward_train(self,
                      imgs,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        '''训练模型
        Args:
            imgs (torch.Tensor): 输入图像
            img_metas (list[dict]): 输入图像的信息
            gt_bboxes (list[torch.Tensor]): 输入图像对应的旋转框
            gt_labels (list[torch.Tensor]): 旋转框对应的类别
            gt_masks (list[BitmapMasks]): 旋转框对应的mask,有masks属性
            gt_bboxes_ignore (list[torch.Tensor] or None): 需要忽略的旋转框
        Returns:
            dict: 损失
        '''
        losses = dict()
        base_feature, x = self.extract_feat(imgs)
        # outs (tuple[list, list, list]): 分类,回归,回归对齐偏移量预测
        outs = self.coarse_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        # coarse head计算损失
        loss_base = self.coarse_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f'coarse.{name}'] = value
        # coarse head阶段所有的旋转框作为refine head的anchor
        # 在双对齐模块中,回归特征对齐可变卷积偏移量来自coarse head检测框,
        # 分类特征对齐可变卷积偏移量来自coarse head中通过分类特征计算的自适应偏移量
        # rois (tuple[list, list]) 回归对齐,分类对齐
        rois = self.coarse_head.refine_bboxes(*outs)
        # align_feat tuple(list list): 对齐的回归特征, 对齐的分类特征
        align_feat = self.align_conv(x, rois)
        # out (tuple[list, list]) 分类, 回归
        outs = self.refine_head(align_feat)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        # 计算损失并返回refine head的采样结果
        loss_refine, sampling_results = self.refine_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois[0])
        for name, value in loss_refine.items():
            losses[f'refine.{name}'] = value
        # mask head需要rois进行裁剪和扣取
        rois = self.refine_head.refine_bboxes(*outs, rois=rois[0])
        # mask head需要最细尺度的特征图
        feature = (base_feature, ) + x
        # outs (tuple[list]): mask head输出特征图,
        # 此时特征图上采样了2倍,但是还不是分割结果
        # 分割结果在mask head的loss和get_masks计算
        outs = self.mask_head(feature)
        loss_inputs = outs + (rois, sampling_results, gt_masks, img_metas)
        loss_mask = self.mask_head.loss(*loss_inputs)
        for name, value in loss_mask.items():
            losses[f'mask.{name}'] = value
        return losses

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError

    def simple_test(self, img, img_metas, rescale=True, thr=0.5):
        base_feature, x = self.extract_feat(img)
        outs = self.coarse_head(x)
        rois = self.coarse_head.refine_bboxes(*outs)
        align_feat = self.align_conv(x, rois)
        outs = self.refine_head(align_feat)

        bbox_inputs = outs + (img_metas, self.test_cfg)
        bbox_list = self.refine_head.get_bboxes(*bbox_inputs, rois=rois[0])
        feature = (base_feature,) + x
        outs = self.mask_head(feature)
        mask_inputs = outs + (bbox_list, img_metas, rescale, thr)
        results_list = self.mask_head.get_masks(*mask_inputs)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.refine_head.num_classes)
            for det_bboxes, det_labels, det_masks in results_list
        ]
        mask_results = [
            det_masks
            for det_bboxes, det_labels, det_masks in results_list
        ]
        return list(zip(bbox_results, mask_results))
