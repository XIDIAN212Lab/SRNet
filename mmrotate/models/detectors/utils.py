# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.ops import DeformConv2d, rotated_feature_align


class AlignConv(nn.Module):
    """Align Conv of `S2ANet`.

    Args:
        in_channels (int): Number of input channels.
        featmap_strides (list): The strides of featmap.
        kernel_size (int, optional): The size of kernel.
        stride (int, optional): Stride of the convolution. Default: None
        deform_groups (int, optional): Number of deformable group partitions.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=None,
                 deform_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.deform_conv, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        """Get the offset of AlignConv."""
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h = \
            x_ctr / stride, y_ctr / stride, \
            w / stride, h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        offset = offset.reshape(anchors.size(0),
                                -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def forward(self, x, anchors):
        """Forward function of AlignConv."""
        anchors = anchors.reshape(x.shape[0], x.shape[2], x.shape[3], 5)
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), self.stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        x = self.relu(self.deform_conv(x, offset_tensor.detach()))
        return x


class AlignConvModule(nn.Module):
    """The module of AlignConv.

    Args:
        in_channels (int): Number of input channels.
        featmap_strides (list): The strides of featmap.
        align_conv_size (int): The size of align convolution.
    """

    def __init__(self, in_channels, featmap_strides, align_conv_size):
        super(AlignConvModule, self).__init__()
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.align_conv_size = align_conv_size
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.ac = nn.ModuleList([
            AlignConv(
                self.in_channels,
                self.in_channels,
                kernel_size=self.align_conv_size,
                stride=s) for s in self.featmap_strides
        ])

    def forward(self, x, rbboxes):
        """
        Args:
            x (list[Tensor]):
                feature maps of multiple scales
            best_rbboxes (list[list[Tensor]]):
                best rbboxes of multiple scales of multiple images
        """
        mlvl_rbboxes = [torch.cat(rbbox) for rbbox in zip(*rbboxes)]
        out = []
        for x_scale, rbboxes_scale, ac_scale in zip(x, mlvl_rbboxes, self.ac):
            feat_refined_scale = ac_scale(x_scale, rbboxes_scale)
            out.append(feat_refined_scale)
        return out


class FeatureRefineModule(nn.Module):
    """Feature refine module for `R3Det`.

    Args:
        in_channels (int): Number of input channels.
        featmap_strides (list): The strides of featmap.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 featmap_strides,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FeatureRefineModule, self).__init__()
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of feature refine module."""
        self.conv_5_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0))
        self.conv_1_5 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2))
        self.conv_1_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1)

    def init_weights(self):
        """Initialize weights of feature refine module."""
        normal_init(self.conv_5_1, std=0.01)
        normal_init(self.conv_1_5, std=0.01)
        normal_init(self.conv_1_1, std=0.01)

    def forward(self, x, best_rbboxes):
        """
        Args:
            x (list[Tensor]):
                feature maps of multiple scales
            best_rbboxes (list[list[Tensor]]):
                best rbboxes of multiple scales of multiple images
        """
        mlvl_rbboxes = [
            torch.cat(best_rbbox) for best_rbbox in zip(*best_rbboxes)
        ]
        out = []
        for x_scale, best_rbboxes_scale, fr_scale in zip(
                x, mlvl_rbboxes, self.featmap_strides):
            feat_scale_1 = self.conv_5_1(self.conv_1_5(x_scale))
            feat_scale_2 = self.conv_1_1(x_scale)
            feat_scale = feat_scale_1 + feat_scale_2
            feat_refined_scale = rotated_feature_align(feat_scale,
                                                       best_rbboxes_scale,
                                                       1 / fr_scale)
            out.append(x_scale + feat_refined_scale)
        return out


class DoubleAlignConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=None,
                 deform_groups=1):
        super(DoubleAlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.deform_conv_with_grid = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups
        )
        self.deform_conv_inregular = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups
        )
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.deform_conv_with_grid, std=0.01)
        normal_init(self.deform_conv_inregular, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, feat, featmap_size, stride):
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        # 获得每个特征点对应的卷积采样位置
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        # 增加一维，这一维放的每个卷积核相对于中心点的相对位置
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy
        # 根据输入的box计算采样位置
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h, a = x_ctr / stride, y_ctr / stride, w / stride, h / stride, a
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr_anchor = cos[:, None] * x - sin[:, None] * y
        yr_anchor = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr_anchor + x_ctr[:, None], yr_anchor + y_ctr[:, None]

        offset_x_anchor = x_anchor - x_conv
        offset_y_anchor = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset_anchor = torch.stack([offset_y_anchor, offset_x_anchor], dim=-1)
        offset_anchor = offset_anchor.reshape(anchors.size(0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)

        # sigmoid
        offset_length = self.kernel_size ** 2
        x, y = dw[:, None] * (xx - 1 / 2) + dw[:, None] * feat[:, :offset_length], \
               dh[:, None] * (yy - 1 / 2) + dh[:, None] * feat[:, offset_length:]
        xr_feat = cos[:, None] * x - sin[:, None] * y
        yr_feat = sin[:, None] * x + cos[:, None] * y

        x_feat, y_feat = xr_feat + x_ctr[:, None], yr_feat + y_ctr[:, None]
        # get offset filed
        offset_x_feat = x_feat - x_conv
        offset_y_feat = y_feat - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset_feat = torch.stack([offset_y_feat, offset_x_feat], dim=-1)
        offset_feat = offset_feat.reshape(anchors.size(0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return [offset_anchor, offset_feat]

    def forward(self, x, anchors, feats):
        """Forward function of AlignConv."""
        anchors = anchors.reshape(x.shape[0], x.shape[2], x.shape[3], 5)
        feats = feats.reshape(x.shape[0], x.shape[2], x.shape[3], 2 * self.kernel_size ** 2)
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), feats[i].reshape(-1, 2 * self.kernel_size ** 2), (H, W),
                            self.stride)
            for i in range(num_imgs)
        ]
        anchor_offset_list = [offset[0] for offset in offset_list]
        feat_offset_list = [offset[1] for offset in offset_list]
        anchor_offset_tensor = torch.stack(anchor_offset_list, dim=0)
        feat_offset_tensor = torch.stack(feat_offset_list, dim=0)
        x_anchor = self.relu(self.deform_conv_with_grid(x, anchor_offset_tensor.detach()))
        x_feat = self.relu(self.deform_conv_inregular(x, feat_offset_tensor.detach()))

        return x_feat, x_anchor


class DoubleAlignModule(nn.Module):
    def __init__(self,
                 in_channels,
                 featmap_strides,
                 conv_size):
        super(DoubleAlignModule, self).__init__()
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.conv_size = conv_size
        self._init_layers()

    def _init_layers(self):
        self.conv = nn.ModuleList([
                DoubleAlignConv(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=self.conv_size,
                    stride=s)
                for s in self.featmap_strides
            ])

    def forward(self, x, rois):
        rbboxes, feats = rois
        mlvl_rbboxes = [torch.cat(rbbox) for rbbox in zip(*rbboxes)]
        mlvl_feats = [torch.cat(feat) for feat in zip(*feats)]
        out_cls = []
        out_reg = []
        for x_scale, rbboxes_scale, feat_scale, cc_scale in zip(x, mlvl_rbboxes, mlvl_feats, self.conv):
            feat_cls, feat_reg = cc_scale(x_scale, rbboxes_scale, feat_scale)
            out_cls.append(feat_cls)
            out_reg.append(feat_reg)
        return out_cls, out_reg
