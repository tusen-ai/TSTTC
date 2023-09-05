from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import numpy as np
import torch


def get_crop_size(bbox, cur_bbox, scale_beta=0.1, expand_ratio=1.1, max_scale=1.26):
    eps = 0.002
    bbox, cur_bbox = expand_bbox_for_bg(bbox, cur_bbox, expand_ratio)

    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    cur_bbox_h = cur_bbox[3] - cur_bbox[1]
    cur_bbox_w = cur_bbox[2] - cur_bbox[0]
    cur_center_x = (cur_bbox[2] + cur_bbox[0]) / 2
    cur_center_y = (cur_bbox[3] + cur_bbox[1]) / 2

    scale_ratio = max_scale  # max(1 / min_scale, max_scale)
    crop_h = scale_ratio * cur_bbox_h
    crop_w = scale_ratio * cur_bbox_w

    if (center_x - crop_w / 2 > eps and center_x + crop_w / 2 + eps < 1 and
            center_y - crop_h / 2 > eps and center_y + crop_h / 2 + eps < 1):
        enlarge_bbox = [center_x - crop_w / 2, center_y - crop_h / 2, center_x + crop_w / 2, center_y + crop_h / 2]
        enlarge_cur_bbox = [cur_center_x - crop_w / 2, cur_center_y - crop_h / 2, cur_center_x + crop_w / 2,
                            cur_center_y + crop_h / 2]

        return enlarge_bbox, enlarge_cur_bbox, bbox, cur_bbox
    else:
        return None


def expand_bbox_for_bg(bbox, cur_bbox, ratio):
    bbox_ratio = get_valid_ratio(bbox, ratio)
    last_bbox_ratio = get_valid_ratio(cur_bbox, ratio)

    ratio = min(bbox_ratio, last_bbox_ratio)
    bbox = expand_box(bbox, ratio)
    cur_bbox = expand_box(cur_bbox, ratio)
    return bbox, cur_bbox


def get_valid_ratio(bbox, ratio):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    ctr_x = (x1 + x2) / 2
    ctr_y = (y1 + y2) / 2
    delta_x = (x2 - x1)
    delta_y = (y2 - y1)

    max_ratio_x = min(ctr_x, 1 - ctr_x) / (delta_x / 2)
    max_ratio_y = min(ctr_y, 1 - ctr_y) / (delta_y / 2)

    max_ratio = min(max_ratio_x, max_ratio_y)
    bg_ratio = min(ratio, max_ratio)
    return bg_ratio


def expand_box(bbox, ratio):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    ctr_x = (x1 + x2) / 2
    ctr_y = (y1 + y2) / 2
    delta_x = (x2 - x1)
    delta_y = (y2 - y1)
    ndelta_x = delta_x * ratio
    ndelta_y = delta_y * ratio

    nx1 = ctr_x - ndelta_x / 2
    nx2 = ctr_x + ndelta_x / 2
    ny1 = ctr_y - ndelta_y / 2
    ny2 = ctr_y + ndelta_y / 2

    return [nx1, ny1, nx2, ny2]

def crop_bbox_img(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[int(y1):int(y2), int(x1):int(x2)]


def downsample_img(img, rate=0.5):
    _img = cv2.resize(
        img,
        (int(img.shape[1] * rate), int(img.shape[0] * rate)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    return _img


def get_cropped_imgs(ref_img, tar_img, ref_box, tar_box, receptive_filed=32, max_unit_size=500):
    H, W = ref_img.shape[:2]
    tar_padding = [min(int(tar_box[0] * W),receptive_filed),min(int(tar_box[1] * H),receptive_filed),
                   min(int(W-tar_box[2] * W),receptive_filed),min(int(H-tar_box[3] * H),receptive_filed)]
    ref_padding = [min(int(ref_box[0] * W), receptive_filed), min(int(ref_box[1] * H), receptive_filed),
                   min(int(W - ref_box[2] * W), receptive_filed), min(int(H - ref_box[3] * H), receptive_filed)]
    _ref_box = [int(ref_box[0] * W)-ref_padding[0], int(ref_box[1] * H)-ref_padding[1],
                int(ref_box[2] * W)+ref_padding[2], int(ref_box[3] * H)+ref_padding[3]]
    _tar_box = [int(tar_box[0] * W)-tar_padding[0], int(tar_box[1] * H)-tar_padding[1],
                int(tar_box[2] * W)+tar_padding[2], int(tar_box[3] * H)+tar_padding[3]]

    ref, tar = crop_bbox_img(ref_img, _ref_box), crop_bbox_img(tar_img, _tar_box)

    if max(ref.shape) > max_unit_size:
        ref = downsample_img(ref, )
        tar = downsample_img(tar, )
        ref_padding = [int(ele / 2) for ele in ref_padding]
        tar_padding = [int(ele / 2) for ele in tar_padding]
    else:
        ref_padding = [int(ele) for ele in ref_padding]
        tar_padding = [int(ele) for ele in tar_padding]

    return [ref, tar], ref_padding, tar_padding


def padding_image_to_same(imgs, boxes_pd, swap=(2, 0, 1), dst_size=[200, 200]):
    '''
    :param imgs: a list of imgs with different size
    :return: list of img with the max size of given imgs, constant padding. list of original img size
    '''
    padded_img_list = []
    orininal_box_list = []
    for img, box_pd in zip(imgs, boxes_pd):
        H, W = img.shape[:2]
        dst_size[0], dst_size[1] = max(dst_size[0], H), max(dst_size[1], W)
        orininal_box_list.append([box_pd[0], box_pd[1], W - box_pd[2], H - box_pd[3]])
    for img in imgs:
        H, W = img.shape[:2]
        ph, pw = dst_size[0] - H, dst_size[1] - W
        padded_img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        # draw_and_show_boxes([], padded_img, )
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        padded_img_list.append(padded_img)
    return padded_img_list, orininal_box_list


def bbox_norm2abs(bbox, img_shape):
    """Conovert normalized bbox corrdinate to absolute pixel coordinate.

    Args:
        bbox (list | tuple | np.ndarray): Normalized bbox.
        img_shape (tuple | list): Image shape.

    Returns:
        Bbox in absolute pixel coordinate.
    """

    assert isinstance(bbox, (list, tuple, np.ndarray)), "Bbox should be list or tuple or np.ndarray!"
    bbox = np.array(bbox, dtype=np.float32)

    assert bbox.ndim <= 2
    if bbox.ndim == 1:
        assert bbox.size >= 4
    elif bbox.ndim == 2:
        assert bbox.shape[1] >= 4
    img_shape = np.array(img_shape)  # w, h
    bbox *= np.tile(img_shape, 2)
    bbox = np.array(bbox, dtype=np.int)
    return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]


# for saving files
import os


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # mp


import pickle


def save_pkl(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(name):
    return pickle.load(open(name, 'rb'))


def load_json(name):
    return json.load(open(name, 'r'))


def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

