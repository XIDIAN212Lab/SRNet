# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import mmcv
from mmrotate.core import obb2poly_np, poly2obb_np
from .transforms import PolyRandomRotate
from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class RRandomRotate(PolyRandomRotate):
    """Rotate img & bbox.
    Reference: https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA

    Args:
        rate (bool): (float, optional): The rotating probability.
            Default: 0.5.
        angles_range(int, optional): The rotate angle defined by random
            (-angles_range, +angles_range).
        auto_bound(bool, optional): whether to find the new width and height
            bounds.
        rect_classes (None|list, optional): Specifies classes that needs to
            be rotated by a multiple of 90 degrees.
        version  (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self,
                 **kwargs):
        super(RRandomRotate, self).__init__(**kwargs)

    def __call__(self, results):
        """Call function of PolyRandomRotate."""
        if not self.is_rotate:
            results['rotate'] = False
            angle = 0
        else:
            angle = 2 * self.angles_range * np.random.rand() - \
                    self.angles_range
            results['rotate'] = True

            class_labels = results['gt_labels']
            for classid in class_labels:
                if self.rect_classes:
                    if classid in self.rect_classes:
                        np.random.shuffle(self.discrete_range)
                        angle = self.discrete_range[0]
                        break

        h, w, c = results['img_shape']
        img = results['img']
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = abs(np.cos(angle)), abs(np.sin(angle))
        if self.auto_bound:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos,
                 h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle,
                                                     bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)

        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)
        gt_masks = results.get('gt_masks', [])
        for i in range(gt_masks.masks.shape[0]):
            gt_masks.masks[i] = self.apply_image(gt_masks.masks[i], bound_h, bound_w)
        results['gt_masks'] = gt_masks

        gt_bboxes = results.get('gt_bboxes', [])
        labels = results.get('gt_labels', [])
        # 这里贴0是因为obb2poly_np的最后一维是score得分
        gt_bboxes = np.concatenate(
            [gt_bboxes, np.zeros((gt_bboxes.shape[0], 1))], axis=-1)
        if gt_bboxes.shape[0] == 0:
            return None
        polys = obb2poly_np(gt_bboxes, self.version)[:, :-1].reshape(-1, 2)
        polys = self.apply_coords(polys).reshape(-1, 8)
        gt_bboxes = []
        for pt in polys:
            pt = np.array(pt, dtype=np.float32)
            obb = poly2obb_np(pt, self.version) \
                if poly2obb_np(pt, self.version) is not None\
                else [0, 0, 0, 0, 0]
            gt_bboxes.append(obb)
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
        gt_bboxes = gt_bboxes[keep_inds, :]
        labels = labels[keep_inds]
        gt_masks.masks = gt_masks.masks[keep_inds]
        if len(gt_bboxes) == 0:
            return None
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = labels
        return results

