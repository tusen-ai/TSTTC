import numpy as np
from tqdm import tqdm
import math

from .utils import (scale_ratio_to_ttc, reweight)
from .cuda_ops import ScaleShiftTorch
from data.ttc_dataset import ttc_to_scale_ratio

def mse_baseline(expand_ratio=1.0, scale_range=[0.8, 1.2], num_scale=13, win_size=3, topk=3, dataloader=None,
                 seq_len=6,bbox_thrs=200):
    scale_shift_op_to = ScaleShiftTorch(expand_ratio=expand_ratio, scale_range=scale_range, num_scale=num_scale,
                                        win_size=win_size,topk=topk,bbox_thrs=bbox_thrs)
    ttc_error_list, scale_error_list = [], []
    progress_bar = tqdm
    fps = 10 / (seq_len - 1)
    for cur_iter, (imgs, boxes, enlarge_boxes, ttcs, annos) in enumerate(
            progress_bar(dataloader)
    ):
        con_flag = False
        for tmp_box in enlarge_boxes:
            if None in tmp_box or tmp_box == []:
                con_flag = True
                break
        if con_flag: continue
        tar_img = np.array(imgs[1])
        ref_img = np.array(imgs[0])

        tmp_ttc_pred, tmp_scale_pred = [], []
        ttcs = ttcs['ttc_gts'] #TODO
        scale_gt = ttc_to_scale_ratio(np.array(ttcs, ), fps)
        ttc_gt = np.array(ttcs)

        tmp_ttc_gt, tmp_scale_gt, tmp_bbox_enlarge = [], [], []
        idx = 0

        for boxes_frame, annos_frame in zip(enlarge_boxes, annos): #batch dim
            for box_frame, anno_frame in zip(boxes_frame, annos_frame): #box dim
                enlarge_bbox, box, cur_bbox = box_frame
                topk_scales_score, topk_scales = scale_shift_op_to(ref_img, tar_img, box, cur_bbox)
                if topk_scales_score is None:
                    print('iter: ', cur_iter, 'box: ', idx, 'topk_scales_score is None')
                    continue  # or give invalid flag
                tmp_bbox_enlarge.append(box_frame)
                mean_scale = sum(topk_scales * reweight(np.array(topk_scales_score)))
                ttc = scale_ratio_to_ttc(mean_scale, fps=fps)  # hard choose top-3
                tmp_ttc_pred.append(ttc)
                tmp_scale_pred.append(mean_scale)
                tmp_ttc_gt.append(ttc_gt[idx])
                tmp_scale_gt.append(scale_gt[idx])
                idx += 1

        tmp_ttc, ttcs_abs_error = compute_error_rate(np.array(tmp_ttc_pred), np.array(tmp_ttc_gt))
        tmp_scale, scale_abs_error = compute_error_rate(np.array(tmp_ttc_pred),np.array(tmp_ttc_gt),mid=True)
        ttc_error_list.extend(tmp_ttc)
        scale_error_list.extend(tmp_scale)

    return sum(ttc_error_list) / len(ttc_error_list), sum(scale_error_list) / (len(scale_error_list))



def compute_error_rate(pred, gt, mid=False):
    assert pred.shape[0] == gt.shape[0]
    relative_errors, abs_error = [], []
    for i in range(pred.shape[0]):

        if mid:
            pred_ttc = max(min(pred[i], 20), -20)
            pred_tmp = ttc_to_scale_ratio(pred_ttc)
            gt_tmp = ttc_to_scale_ratio(gt[i])
            error = float(abs(math.log(pred_tmp) - math.log(gt_tmp)) * 10 ** 4)
        else:
            pred_tmp = max(min(pred[i], 20), -20)
            gt_tmp = gt[i]
            error = float(abs((pred_tmp - gt_tmp) / abs(gt_tmp)) * 100)
        abs_error.append(abs(pred_tmp - gt_tmp))
        relative_errors.append(error)
    return relative_errors, abs_error
