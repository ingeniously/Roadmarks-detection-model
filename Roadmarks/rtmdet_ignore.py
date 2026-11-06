"""RTMDet-Ins head that respects iscrowd=1 "ignore" regions (marks too far / too hidden / rare).

Stock RTMDet (DynamicSoftLabelAssigner) drops the ignore regions, so their pixels become background and the model is
trained to NOT detect real road markings there. Here the background (negative) points whose location falls inside an
ignore mask get classification weight 0: neither rewarded nor punished, like ignore_iof_thr in Mask R-CNN and like
the COCO evaluation (detections on iscrowd regions are not counted). Positive points are unchanged.

Import before building the model (configs use custom_imports; scripts: `import rtmdet_ignore`).
"""
import torch
from mmdet.models.dense_heads import RTMDetInsSepBNHead
from mmdet.registry import MODELS


@MODELS.register_module()
class RTMDetInsSepBNHeadIgnore(RTMDetInsSepBNHead):

    def _get_targets_single(self, cls_scores, bbox_preds, flat_anchors, valid_flags, gt_instances, img_meta,
                            gt_instances_ignore=None, unmap_outputs=True):
        out = super()._get_targets_single(cls_scores, bbox_preds, flat_anchors, valid_flags, gt_instances,
                                          img_meta, gt_instances_ignore, unmap_outputs)
        if out[0] is None or gt_instances_ignore is None or len(gt_instances_ignore) == 0 \
                or 'masks' not in gt_instances_ignore:
            return out
        anchors, labels, label_weights = out[0], out[1], out[2]
        ign = gt_instances_ignore.masks.to_tensor(dtype=torch.bool, device=labels.device).any(0)  # H x W
        h, w = ign.shape
        # point priors: (x, y, stride_w, stride_h)
        x = anchors[:, 0].long().clamp_(0, w - 1)
        y = anchors[:, 1].long().clamp_(0, h - 1)
        in_ign = ign[y, x] & (labels == self.num_classes)
        label_weights = label_weights.clone()
        label_weights[in_ign] = 0.0
        return (anchors, labels, label_weights) + tuple(out[3:])
