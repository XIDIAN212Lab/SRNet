# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmrotate.core import obb2poly


def sample_points_train(rois, version='oc', output_size=14):
    device, dtype = rois.device, rois.dtype
    num_rois = rois.shape[0]
    poly = obb2poly(rois, version)
    cx, cy, w, h, theta = rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
    start_points = poly[:, :2]
    # 这里采样点有一个问题,并未采样到框的轮廓,而是框的内部
    # 后续可以改进
    idx = torch.arange(1, output_size + 1, dtype=dtype, device=device)
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
    sample_result = sample_points_train(rois, version, output_size)
    sample_result[:, :, 0] = sample_result[:, :, 0] / (img_w / 2) - 1
    sample_result[:, :, 1] = sample_result[:, :, 1] / (img_h / 2) - 1
    sample_masks = []
    sample_gt_masks = []
    for i in range(num_rois):
        sample = F.grid_sample(masks[None], sample_result[i][None, None],
                               align_corners=False)[0].squeeze(1).reshape(-1, output_size, output_size)
        sample_masks.append(sample)
        sample = F.grid_sample(gt_masks[i][None, None], sample_result[i][None, None], align_corners=False)[
            0].squeeze().reshape(-1, output_size, output_size)
        sample[sample < 0.5] = 0.0
        sample[sample >= 0.5] = 1.0
        sample_gt_masks.append(sample)
    sample_masks = torch.stack(sample_masks, 0)
    sample_gt_masks = torch.cat(sample_gt_masks, 0)
    return sample_masks, sample_gt_masks


def sample_points_test(rois, version='oc', output_size=14):
    device, dtype = rois.device, rois.dtype
    num_rois = rois.shape[0]
    poly = obb2poly(rois, version)
    cx, cy, w, h, theta = rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
    start_points = poly[:, :2]
    # 这里采样点有一个问题,并未采样到框的轮廓,而是框的内部
    # 后续可以改进
    idx = torch.arange(0, output_size, dtype=dtype, device=device) + 0.5
    yy, xx = torch.meshgrid(idx, idx)
    w_delta = w / output_size
    h_delta = h / output_size
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


def sample_feature(rois, feature, img_metas, version='oc', output_size=28):
    img_w, img_h, _ = img_metas['pad_shape']
    num_rois = rois.shape[0]
    sample_result = sample_points_test(rois, version, output_size)
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
    idx = torch.arange(0, img_size, dtype=dtype, device=device) + 0.5
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


def sample_feature_and_gt_masksV2(rois, masks, gt_masks, img_metas, version='oc', output_size=28):
    img_w, img_h, _ = img_metas['pad_shape']
    num_rois = rois.shape[0]

    sample_result_gt = sample_points_train(rois, version, output_size)
    sample_result_gt[:, :, 0] = sample_result_gt[:, :, 0] / (img_w / 2) - 1
    sample_result_gt[:, :, 1] = sample_result_gt[:, :, 1] / (img_h / 2) - 1

    sample_result = sample_points_train(rois, version, int(output_size/2))
    sample_result[:, :, 0] = sample_result[:, :, 0] / (img_w / 2) - 1
    sample_result[:, :, 1] = sample_result[:, :, 1] / (img_h / 2) - 1

    sample_masks = []
    sample_gt_masks = []
    for i in range(num_rois):
        sample = F.grid_sample(masks[None], sample_result[i][None, None],
                               align_corners=False)[0].squeeze(1).reshape(-1, int(output_size/2), int(output_size/2))
        sample_masks.append(sample)
        sample = F.grid_sample(gt_masks[i][None, None], sample_result_gt[i][None, None], align_corners=False)[
            0].squeeze().reshape(-1, output_size, output_size)
        sample[sample < 0.5] = 0.0
        sample[sample >= 0.5] = 1.0
        sample_gt_masks.append(sample)
    sample_masks = torch.stack(sample_masks, 0)
    sample_gt_masks = torch.cat(sample_gt_masks, 0)
    return sample_masks, sample_gt_masks
