import torch
import custom_gridsample as gscuda
import time
import warnings
import sys
sys.path.append("..")
from .utils import resize_img, reweight, bbox_xyxy2xywh, bbox_xywh2xyxy, cat_2img
import numpy as np
import cv2


def make_ctr_scale_grid(h_in, w_in, h_out, w_out, scale_list):
    # assert h_in >= h_out
    # assert w_in >= w_out
    num_scale = len(scale_list)
    scale_arr = torch.Tensor(scale_list)
    x = torch.linspace(-1, 1, w_in)[(w_in - w_out) // 2: (w_in - w_out) // 2 + w_out]
    y = torch.linspace(-1, 1, h_in)[(h_in - h_out) // 2: (h_in - h_out) // 2 + h_out]
    meshy, meshx = torch.meshgrid((y, x))  # HxW, HxW
    grid = torch.stack((meshx, meshy), 2)  # W(x), H(y), 2
    grid = grid.unsqueeze(-1).repeat(1, 1, 1, num_scale)  # hxwx2xN
    grid = grid * 1 / scale_arr
    # scale = 0.5, means cur frame object is expanding compare to last frame
    # thus we should zoom out to match last frame img.
    # scale = 0.5 means zoon out, exactly.
    grid = grid.permute(3, 0, 1, 2)  # Nxhxwx2
    return grid


def make_ctr_shift_grid(h_in, w_in, h_out, w_out, scale_list):
    # assert h_in >= h_out
    # assert w_in >= w_out
    raw_x_grid = torch.linspace(-1, 1, w_in)
    raw_y_grid = torch.linspace(-1, 1, h_in)
    all_grid = []
    _x = torch.ones(w_out)
    _y = torch.ones(h_out)
    for scale in scale_list:
        expand_w = w_out / scale
        expand_h = h_out / scale

        start_x = int((w_in - expand_w) / 2)
        start_y = int((h_in - expand_h) / 2)
        if start_x < 0 or start_y < 0:
            start_x = 0
            start_y = 0
            warnings.warn("lower bound of shift moving is  out-of-boundary,\
                hyper-parameters may not suitable. The process is keep running.")

        x = raw_x_grid[start_x: start_x + w_out]
        y = raw_y_grid[start_y: start_y + h_out]
        _x[:x.shape[0]] = x
        _y[:y.shape[0]] = y

        meshy, meshx = torch.meshgrid((_y, _x))
        grid = torch.stack((meshx, meshy), 2)
        all_grid.append(grid.unsqueeze(0))
    all_grid = torch.cat(all_grid, 0)  # NxHxWx2
    return all_grid


def expand_bbox_xyxy(bbox, ratio):
    # wont check Out-of-Range
    cx, cy, w, h = bbox_xyxy2xywh(bbox)
    w *= ratio
    h *= ratio
    bbox = bbox_xywh2xyxy([cx, cy, w, h])
    return bbox


def expand_bboxes_v2(bbox1, bbox2, imgh, imgw, ratio):
    cx1, cy1, w1, h1 = bbox_xyxy2xywh(bbox1)
    cx2, cy2, w2, h2 = bbox_xyxy2xywh(bbox2)

    bound_w1 = min(imgw - cx1, cx1)
    bound_h1 = min(imgh - cy1, cy1)
    max_ratio1 = min(2 * bound_w1 / w1, 2 * bound_h1 / h1)

    bound_w2 = min(imgw - cx2, cx2)
    bound_h2 = min(imgh - cy2, cy2)
    max_ratio2 = min(2 * bound_w2 / w2, 2 * bound_h2 / h2)
    ratio = min(min(max_ratio1, max_ratio2), ratio)
    w1 *= ratio
    h1 *= ratio

    w2 *= ratio
    h2 *= ratio

    new_bbox1 = bbox_xywh2xyxy([cx1, cy1, w1, h1])
    new_bbox2 = bbox_xywh2xyxy([cx2, cy2, w2, h2])

    base_max_ratio = min(max_ratio1, max_ratio2)
    return new_bbox1, new_bbox2, ratio, base_max_ratio


def expand_bbox_same_size(bbox_f1, bbox_f2, imgh, imgw, ratio):
    new_bbox_f1 = expand_bbox_xyxy(bbox_f1, ratio)
    new_bbox_f2 = expand_bbox_xyxy(bbox_f2, ratio)

    f1_x1, f1_y1, f1_x2, f1_y2 = new_bbox_f1
    f2_x1, f2_y1, f2_x2, f2_y2 = new_bbox_f2

    f1_x1 = max(0, f1_x1)
    f2_x1 = max(0, f2_x1)

    f1_y1 = max(0, f1_y1)
    f2_y1 = max(0, f2_y1)

    f1_x2 = min(f1_x2, imgw)
    f2_x2 = min(f2_x2, imgw)

    f1_y2 = min(f1_y2, imgh)
    f2_y2 = min(f2_y2, imgh)

    f1w, f1h = f1_x2 - f1_x1, f1_y2 - f1_y1
    f2w, f2h = f2_x2 - f2_x1, f2_y2 - f2_y1

    h = f1h if (f2_y1 + f1h) < imgh else imgh - f2_y1
    w = f1w if (f2_x1 + f1w) < imgw else imgw - f2_x1

    # print(h, w, new_bbox_f1, new_bbox_f2)
    new_bbox_f1 = f1_x1, f1_y1, f1_x1 + w, f1_y1 + h
    new_bbox_f2 = f2_x1, f2_y1, f2_x1 + w, f2_y1 + h
    return new_bbox_f1, new_bbox_f2


def crop_bbox_img(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[int(y1):int(y2), int(x1):int(x2)]


def get_topk_scale(scale_outputs, shift_outputs, scale_list, topk=1, return_out=True):
    """Get Topk-scale and scores.
    Args:
        scale_outputs (torch.Tensor): NxCxHxW
        shift_outputs (torch.Tensor): NxMxCxHxW
    """
    N, M = shift_outputs.shape[:2]
    shift_outputs = shift_outputs.transpose(1, 0)  # MxNxCxHxW
    # avoid oom
    if scale_outputs.shape[-1] * scale_outputs.shape[-2] > 200 * 200:
        scale_outputs = scale_outputs.cpu()
        shift_outputs = shift_outputs.cpu()
    diff = scale_outputs - shift_outputs

    diff = diff ** 2
    diff = torch.mean(diff.reshape(M, N, -1), dim=-1)  # M x N
    score = 1 / diff / 255
    best_score_under_shift, best_score_under_shift_idx = torch.max(score, dim=0)  # N
    topk_score, topk_score_idx = torch.topk(best_score_under_shift, topk)
    top_same = len(torch.where(best_score_under_shift==topk_score[0])[0])
    if top_same > topk:
        topk_score, topk_score_idx = torch.topk(best_score_under_shift, top_same)

    if not return_out:
        return topk_score, [scale_list[i] for i in topk_score_idx]

    topk_scale_outputs = scale_outputs[topk_score_idx, ...]
    topk_shift_outputs = torch.zeros_like(topk_scale_outputs)
    for i, idx in enumerate(topk_score_idx):
        topk_shift_outputs[i] = shift_outputs[best_score_under_shift_idx[idx], idx, ...]
    return topk_score, [scale_list[i] for i in topk_score_idx], topk_scale_outputs, topk_shift_outputs


class GridScaleShift(object):
    # Defination in CPP:
    _INTER_MAP = {
        'nearst': 0,
        'bilinear': 1,
    }
    _PADDING_MAP = {
        'zeros': 0,
        'border': 1,
    }

    def __init__(self, interp='nearst', padding='zeros', verbose=False):
        self._interp = interp
        self._padding = padding
        self.interp_mode = self._INTER_MAP[interp]
        self.padding_mode = self._PADDING_MAP[padding]
        self.verbose = verbose
        self.device = "cuda:0"
        assert torch.cuda.is_available()

    def __call__(self, scale_inputs, shift_inputs, scale_grid, shift_grid, win_size):
        # scale & shift : NxHxWx2, input: CxHxW;
        assert scale_inputs.shape == shift_inputs.shape, f"{scale_inputs.shape}, {shift_inputs.shape}"
        assert scale_grid.shape == shift_grid.shape, f"{scale_grid.shape}, {shift_grid.shape}"

        scale_out, shift_out = gscuda.forward_2d_scale_shift(scale_inputs, shift_inputs, scale_grid, \
                                                             shift_grid, win_size, self.interp_mode, self.padding_mode,
                                                             True)  # align corners
        _num_scale = scale_out.shape[0]
        _h, _w = scale_out.shape[-2], scale_out.shape[-1]
        shift_out = shift_out.reshape(_num_scale, -1, 3, _h, _w).to(device=self.device)  # NxMx3xHxW
        scale_out = scale_out.to(device=self.device)

        del scale_grid
        del scale_inputs
        del shift_grid
        del shift_inputs
        return scale_out, shift_out

    def __repr__(self):
        _msg = 'Custom Grid Sample Forward <CUDA> ' + \
               f'Interpolation = {self._interp}' + \
               f'Padding = {self._padding}'
        return _msg


class ScaleShiftTorch(object):
    def __init__(self, scale_range=[0.8, 1.1], num_scale=30, win_size=3, topk=3, expand_ratio=1.1,
                 bbox_thrs=128, device="cuda:0"):
        self.scale_range = scale_range
        self.num_scale = num_scale
        self.win_size = win_size
        self.topk = topk
        self.bg_ratio = expand_ratio
        self.bbox_thrs = bbox_thrs  # avoid oom
        self.cuda_op = GridScaleShift()
        self.device = device

    def __call__(self, last_img,img, last_bbox, bbox, ts=None, cam=None, npc_id=None):
        imgh, imgw = img.shape[:2]
        # take bg and get boundary ratio
        ibbox = [bbox[0]*imgw,bbox[1]*imgh,bbox[2]*imgw,bbox[3]*imgh]
        ilast_bbox = [last_bbox[0]*imgw,last_bbox[1]*imgh,last_bbox[2]*imgw,last_bbox[3]*imgh]

        raw_bbox, raw_last_bbox = ibbox, ilast_bbox
        bbox, last_bbox, ratio, base_max_expand_ratio = expand_bboxes_v2(raw_bbox, raw_last_bbox, imgh=imgh, imgw=imgw,
                                                                         ratio=self.bg_ratio)
        # print("try", ts, cam, npc_id, ratio, base_max_expand_ratio, img.shape, bbox)
        if base_max_expand_ratio > self.scale_range[1] and base_max_expand_ratio < self.scale_range[1] * ratio:
            # not allow enough bg
            bbox, last_bbox, ratio, base_max_expand_ratio = expand_bboxes_v2(raw_bbox, raw_last_bbox, imgh=imgh,
                                                                             imgw=imgw, ratio=max(
                    base_max_expand_ratio / self.scale_range[1], 1.0))

        base_bbox_h = int(bbox[3] - bbox[1])
        base_bbox_w = int(bbox[2] - bbox[0])

        still_expand_ratio = base_max_expand_ratio / ratio

        if (still_expand_ratio) < 1.1:  # hard threshold
            return None, None
        elif (1 / self.scale_range[0]) > still_expand_ratio:
            # cur range is need changes:
            if self.scale_range[1] > still_expand_ratio:
                scale_list = np.linspace(1 / still_expand_ratio, still_expand_ratio, self.num_scale)
            else:
                scale_list = np.linspace(1 / still_expand_ratio, self.scale_range[1], self.num_scale)
        else:
            scale_list = np.linspace(self.scale_range[0], self.scale_range[1], self.num_scale)
        # avoid zero-padding
        zero_padding_expand_ratio = max(1 / scale_list[0],scale_list[-1])
        expand_bbox, expand_last_bbox = expand_bbox_same_size(bbox, last_bbox, imgh, imgw, zero_padding_expand_ratio)

        bbox_img = crop_bbox_img(img, expand_bbox)
        bbox_imgh, bbox_imgw = bbox_img.shape[:2]
        last_bbox_img = crop_bbox_img(last_img, [expand_last_bbox[0], expand_last_bbox[1], \
                                                 expand_last_bbox[0] + bbox_imgw, expand_last_bbox[1] + bbox_imgh])
        assert bbox_img.shape == last_bbox_img.shape, "{}, {}".format(bbox_img.shape, last_bbox_img.shape)
        if bbox_imgh < base_bbox_h or bbox_imgw < base_bbox_w:
            # need check this case
            return None, None

        if bbox_imgh * bbox_imgw > self.bbox_thrs ** 2:
            # resize bbox imgs
            _ratio = max(bbox_imgh / self.bbox_thrs, bbox_imgw / self.bbox_thrs)
            _new_imgh = int(bbox_imgh / _ratio)
            _new_imgw = int(bbox_imgw / _ratio)
            bbox_img = resize_img(bbox_img, _new_imgh, _new_imgw)
            last_bbox_img = resize_img(last_bbox_img, _new_imgh, _new_imgw)
            base_bbox_h = int(base_bbox_h / _ratio)
            base_bbox_w = int(base_bbox_w / _ratio)
            bbox_imgh = _new_imgh
            bbox_imgw = _new_imgw

        scale_grid \
            = make_ctr_scale_grid(bbox_imgh, bbox_imgw, base_bbox_h, base_bbox_w, scale_list)
        shift_grid \
            = make_ctr_shift_grid(bbox_imgh, bbox_imgw, base_bbox_h, base_bbox_w, scale_list)
        scale_grid = scale_grid.to(device=self.device).contiguous().float()
        shift_grid = shift_grid.to(device=self.device).contiguous().float()
        bbox_img = torch.Tensor(np.transpose(bbox_img, (2, 0, 1)) / 255).to(device=self.device).contiguous().float()
        last_bbox_img = torch.Tensor(np.transpose(last_bbox_img, (2, 0, 1)) / 255).to(
            device=self.device).contiguous().float()
        scale_out, shift_out = self.cuda_op(bbox_img, last_bbox_img, scale_grid, shift_grid, self.win_size)
        topk_scales_score, topk_scales = get_topk_scale(scale_out, shift_out, scale_list, self.topk, False)

        del scale_grid
        del shift_grid
        del bbox_img
        del last_bbox_img

        torch.cuda.empty_cache()

        return topk_scales_score.cpu().numpy(), topk_scales
