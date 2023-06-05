# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.ops import convex_iou
from mmrotate.core import obb2poly


def points_center_pts(RPoints, y_first=True):
    """Compute center point of Pointsets.

    Args:
        RPoints (torch.Tensor): the  lists of Pointsets, shape (k, 18).
        y_first (bool, optional): if True, the sequence of Pointsets is (y,x).

    Returns:
        center_pts (torch.Tensor): the mean_center coordination of Pointsets,
            shape (k, 18).
    """
    RPoints = RPoints.reshape(-1, 9, 2)

    if y_first:
        pts_dy = RPoints[:, :, 0::2]
        pts_dx = RPoints[:, :, 1::2]
    else:
        pts_dx = RPoints[:, :, 0::2]
        pts_dy = RPoints[:, :, 1::2]
    pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
    pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
    center_pts = torch.cat([pts_dx_mean, pts_dy_mean], dim=1).reshape(-1, 2)
    return center_pts


def convex_overlaps(gt_bboxes, points):
    """Compute overlaps between polygons and points.

    Args:
        gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).
        points (torch.Tensor): Points to be assigned, shape(n, 18).

    Returns:
        overlaps (torch.Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
    """
    overlaps = convex_iou(points, gt_bboxes)
    overlaps = overlaps.transpose(1, 0)
    return overlaps


def levels_to_images(mlvl_tensor, flatten=False):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)
        flatten (bool, optional): if shape of mlvl_tensor is (N, C, H, W)
            set False, if shape of mlvl_tensor is  (N, H, W, C) set True.

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    if flatten:
        channels = mlvl_tensor[0].size(-1)
    else:
        channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        if not flatten:
            t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


def get_num_level_anchors_inside(num_level_anchors, inside_flags):
    """Get number of every level anchors inside.

    Args:
        num_level_anchors (List[int]): List of number of every level's anchors.
        inside_flags (torch.Tensor): Flags of all anchors.

    Returns:
        List[int]: List of number of inside anchors.
    """
    split_inside_flags = torch.split(inside_flags, num_level_anchors)
    num_level_anchors_inside = [
        int(flags.sum()) for flags in split_inside_flags
    ]
    return num_level_anchors_inside


def sample_points(rois, version='oc', output_size=14):
    device, dtype = rois.device, rois.dtype
    num_rois = rois.shape[0]
    poly = obb2poly(rois, version)
    cx, cy, w, h, theta = rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
    start_points = poly[:, :2]
    # 这里采样点有一个问题,并未采样到框的轮廓,而是框的内部
    # 后续可以改进
    idx = torch.arange(1, output_size+1, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(idx, idx)
    w_delta = w / (output_size + 1)
    h_delta = h / (output_size + 1)
    yy = yy[None] * h_delta[:, None, None].repeat(1, output_size, output_size)
    xx = xx[None] * w_delta[:, None, None].repeat(1, output_size, output_size)
    yy = yy.reshape(num_rois, -1)
    xx = xx.reshape(num_rois, -1)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    sample_xx = cos[:, None] * xx - sin[:, None] * yy + start_points[:, [0]]
    sample_yy = cos[:, None] * yy + sin[:, None] * xx + start_points[:, [1]]
    points = torch.stack([sample_xx, sample_yy], dim=-1)
    return points


def sample_feature_and_gt_masks(rois, masks, gt_masks, img_metas, version='oc', output_size=28):
    img_w, img_h, _ = img_metas['pad_shape']
    num_rois = rois.shape[0]
    sample_result = sample_points(rois, version, output_size)
    sample_result[:, :, 0] = sample_result[:, :, 0] / (img_w / 2) - 1
    sample_result[:, :, 1] = sample_result[:, :, 1] / (img_h / 2) - 1
    sample_masks = []
    sample_gt_masks = []
    for i in range(num_rois):
        sample = F.grid_sample(masks[None], sample_result[i][None, None],
                               align_corners=False)[0].squeeze(1).reshape(-1, output_size, output_size)
        # sample_debug1 = sample.cpu().detach().numpy() > 0.5
        sample_masks.append(sample)
        sample = F.grid_sample(gt_masks[i][None, None], sample_result[i][None, None], align_corners=False)[
            0].squeeze().reshape(-1, output_size, output_size)
        sample[sample < 0.5] = 0.0
        sample[sample >= 0.5] = 1.0
        # sample_debug2 = sample.cpu().detach().numpy()
        sample_gt_masks.append(sample)
    sample_masks = torch.stack(sample_masks, 0)
    sample_gt_masks = torch.cat(sample_gt_masks, 0)
    return sample_masks, sample_gt_masks


def sample_feature(rois, feature, img_metas, version='oc', output_size=28):
    img_w, img_h, _ = img_metas['pad_shape']
    num_rois = rois.shape[0]
    sample_result = sample_points(rois, version, output_size)
    sample_result[:, :, 0] = sample_result[:, :, 0] / (img_w / 2) - 1
    sample_result[:, :, 1] = sample_result[:, :, 1] / (img_h / 2) - 1
    sample_masks = []
    for i in range(num_rois):
        sample = F.grid_sample(feature[None], sample_result[i][None, None],
                               align_corners=False)[0].squeeze(1).reshape(-1, output_size, output_size)
        sample_masks.append(sample)
    sample_masks = torch.stack(sample_masks, 0)
    return sample_masks


def get_sample_points(rois, version, output_size, img_size):
    N, dtype, device = rois.shape[0], rois.dtype, rois.device
    cx, cy, w, h, theta = rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
    cos = torch.cos(-theta)
    sin = torch.sin(-theta)
    poly = obb2poly(rois, version)
    start_point = poly[:, :2]
    idx = torch.arange(0, img_size, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(idx, idx)
    # 平移坐标系
    s_xx = xx[None] - start_point[:, 0][:, None, None]
    s_yy = yy[None] - start_point[:, 1][:, None, None]
    # 旋转坐标系
    r_xx = (s_xx * cos[:, None, None] - s_yy * sin[:, None, None]) / (w[:, None, None] / output_size)
    r_yy = (s_yy * cos[:, None, None] + s_xx * sin[:, None, None]) / (h[:, None, None] / output_size)
    return torch.stack([r_xx, r_yy], dim=-1)


def get_mask(rois, labels, crop_masks, version='oc', output_size=28, img_meta=None, thr=0.5, num_classes=1, rescale=True):
    assert img_meta is not None
    pad_shape = img_meta['pad_shape']
    img_shape = img_meta['img_shape']
    ori_shape = img_meta['ori_shape']
    grid = get_sample_points(rois, version, output_size, pad_shape[0])
    grid[..., 0] = grid[..., 0] / (output_size / 2) - 1
    grid[..., 1] = grid[..., 1] / (output_size / 2) - 1
    masks = F.grid_sample(crop_masks, grid, align_corners=False)[:, 0]
    masks = masks[:, :img_shape[0], :img_shape[1]]
    # 这里是错误的不能直接插值改变mask的大小, 因为mask是被pad过的,需要先还原到img_shape,再线性插值到ori_shape
    if rescale:
        masks = F.interpolate(masks[None], ori_shape[:2], mode='bilinear').squeeze(0)
    masks = (masks > thr).to(torch.bool)
    N = rois.shape[0]
    cls_segms = [[] for _ in range(num_classes)]
    for i in range(N):
        cls_segms[labels[i]].append(masks[i].detach().cpu().numpy())
    return cls_segms
