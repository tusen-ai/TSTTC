import numpy as np
from collections import defaultdict
import cv2
# import yaml
# import os.path as osp
#from perception_msgs.msg import TTCObject


def cat_2img(img1, img2):
    """ along x-axis cat
    Args:
        img1 (_type_): imread img1
        img2 (_type_): imread img2
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = max(w1, w2)
    new_ = np.zeros((h, w * 2, 3))

    new_[:h1, :w1, :] = img1
    new_[:h2, w:w + w2, :] = img2
    return new_


# def load_cfg(cfg_name):
#     if not isinstance(cfg_name, str):
#         return None
#     if cfg_name.endswith('.yaml'):
#         with open(osp.join(CONFIG_DIR, cfg_name), 'r') as f:
#             cfg = yaml.safe_load(f)
#     else:
#         raise NotImplementedError("Unsupported config format.")
#     return cfg


# def wrap_ttc_msg(ttc_msg, res):
#     for npc_res in res:
#         ttc_obj = TTCObject()
#         ttc_obj.scale_val = npc_res["scale_val"]
#         ttc_obj.scale_score = npc_res["scale_score"]
#         ttc_obj.scale_var = npc_res["scale_var"]
#         ttc_obj.object_id = npc_res["object_id"]
#         ttc_obj.cam_id = npc_res["cam_id"]
#         ttc_obj.flag = npc_res["flag"]
#         ttc_obj.ttc = npc_res["ttc"]
#         # ttc_obj.header = npc_res["header"]
#         ttc_msg.ttc_status.append(ttc_obj)


def scale_ratio_to_ttc(scale_ratio, fps=10):
    ttc = 1 / ((fps * (1 / scale_ratio - 1)) + 1e-6)
    return ttc


def ttc_to_scale_ratio(ttc, fps=10):
    scale_ratio = 1 / ((1 / (ttc * fps) + 1) + 1e-6)
    return scale_ratio


def is_partial(bbox, h, w, ratio=0.02):
    x1, y1, x2, y2 = bbox[:4]
    partial_flag = x1 < w * ratio or x2 > w * (1 - ratio) or y1 < h * ratio or y2 > h * (1 - ratio)
    return partial_flag


def bbox_size_xyxy(bbox):
    x1, y1, x2, y2 = bbox
    return (y2 - y1) * (x2 - x1)


def bbox_xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    center_x = (x1 + x2) / 2.
    center_y = (y1 + y2) / 2.
    return center_x, center_y, w, h


def bbox_xywh2xyxy(bbox):
    cx, cy, w, h = bbox
    x1 = cx - w / 2.
    x2 = cx + w / 2.
    y1 = cy - h / 2.
    y2 = cy + h / 2.
    return x1, y1, x2, y2


def expand_bbox_xyxy(bbox, ratio):
    # wont check Out-of-Range
    cx, cy, w, h = bbox_xyxy2xywh(bbox)
    w *= ratio
    h *= ratio
    bbox = bbox_xywh2xyxy([cx, cy, w, h])
    return bbox


def expand_bboxes(bbox1, bbox2, imgh, imgw, ratio):
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
    return new_bbox1, new_bbox2


def get_bboxxyxy(bbox_per, img):
    imgh, imgw = img.shape[:2]
    bbox = np.array(bbox_per)
    bbox[[0, 2]] *= imgw
    bbox[[1, 3]] *= imgh
    return bbox


def parse_header(header):
    sec = header.stamp.sec
    nanosec = header.stamp.nanosec
    return int(str(sec) + "{:09d}".format(nanosec))


def compute_box_3d(center, w, h, l, yaw):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''

    def rotz(t):
        ''' Rotation about the z-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    tx, ty, tz = center
    # compute rotational matrix around yaw axis
    R = rotz(yaw)

    # 3d bounding box corners
    x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    y_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + tx
    corners_3d[1, :] = corners_3d[1, :] + ty
    corners_3d[2, :] = corners_3d[2, :] + tz
    return np.transpose(corners_3d)


def resize_img(img, imgh, imgw, dtype=np.float32, inter="linear"):
    _map = {
        "cubic": cv2.INTER_CUBIC,
        "linear": cv2.INTER_LINEAR,
    }
    img = cv2.resize(img, (imgw, imgh), interpolation=_map[inter]).astype(dtype)
    return img


def mean_squared_error(p1, p2):
    err = np.average((p1 - p2) ** 2)
    # may do more op here
    return err


def get_score_between_patches(ref_patch, last_img, last_bbox, scale_h, scale_w, fix_size=None, metric="mse", ):
    imgh, imgw = last_img.shape[:2]
    last_x1, last_y1, last_x2, last_y2 = last_bbox[:4]
    shift_x1 = last_x1
    shift_y1 = last_y1
    shift_x2 = min(imgw, shift_x1 + scale_w)
    shift_y2 = min(imgh, shift_y1 + scale_h)
    crop_last_bbox_img = last_img[int(shift_y1):int(shift_y2), int(shift_x1):int(shift_x2)]
    crop_last_bbox_imgh, crop_last_bbox_imgw = crop_last_bbox_img.shape[:2]  # boundary may out-of-range

    if crop_last_bbox_img.shape != ref_patch.shape:
        # print(crop_last_bbox_imgh, crop_last_bbox_imgw, shift_y1,  shift_y2, \
        #     shift_x1, shift_x2)
        ref_patch = resize_img(ref_patch, imgh=crop_last_bbox_imgh, imgw=crop_last_bbox_imgw)

    if fix_size is not None and (crop_last_bbox_imgh * crop_last_bbox_imgw) > fix_size ** 2:
        small_ratio = max(crop_last_bbox_imgh / fix_size, crop_last_bbox_imgw / fix_size)
        small_h, small_w = int(crop_last_bbox_imgh / small_ratio), int(crop_last_bbox_imgw / small_ratio)
        crop_last_bbox_img = resize_img(crop_last_bbox_img, imgh=small_h, imgw=small_w)
        ref_patch = resize_img(ref_patch, imgh=small_h, imgw=small_w)

    if metric == "mse":
        val = 1 / (mean_squared_error(ref_patch, crop_last_bbox_img) / 255)
    else:
        raise

    return val, ref_patch, crop_last_bbox_img


def scale_shift(img, last_img, bbox, last_bbox, scale_list, topk=3, win=5, fix_size=128):
    imgh, imgw = img.shape[:2]
    ts_x1, ts_y1, ts_x2, ts_y2 = bbox[:4]
    last_x1, last_y1, last_x2, last_y2 = last_bbox[:4]
    cur_ts_bbox_img = img[int(ts_y1):int(ts_y2), int(ts_x1):int(ts_x2)]
    scale_vals = []
    cur_ts_bbox_imgh, cur_ts_bbox_imgw = cur_ts_bbox_img.shape[:2]
    for i, scale_ratio in enumerate(scale_list):
        raw_resize_ts_bbox_img = resize_img(cur_ts_bbox_img, imgh=int(cur_ts_bbox_imgh * scale_ratio),
                                            imgw=int(cur_ts_bbox_imgw * scale_ratio))
        scale_h, scale_w = raw_resize_ts_bbox_img.shape[:2]
        _best_shift = 0
        if win == 0:
            val, resize_ts_bbox_img, crop_last_bbox_img = get_score_between_patches(raw_resize_ts_bbox_img, last_img,
                                                                                    last_bbox, scale_h, scale_w,
                                                                                    fix_size)
        for shift_x1 in range(int(last_x1) - win, int(last_x1) + win):
            for shift_y1 in range(int(last_y1) - win, int(last_y1) + win):
                _shift_last_bbox = [shift_x1, shift_y1, None, None]  # only use top-left
                val, resize_ts_bbox_img, crop_last_bbox_img = get_score_between_patches(raw_resize_ts_bbox_img,
                                                                                        last_img, _shift_last_bbox,
                                                                                        scale_h, scale_w, fix_size)
                if val > _best_shift:
                    _best_shift = val
                    _best_scale = 1 / (mean_squared_error(resize_ts_bbox_img, crop_last_bbox_img) / 255)
        scale_vals.append(_best_scale)

    scale_vals = np.array(scale_vals)
    argmax_val = np.argsort(scale_vals)[::-1][:topk]

    max_val = [scale_vals[i] for i in argmax_val]
    max_val_scale = [scale_list[i] for i in argmax_val]
    return max_val, max_val_scale


def reweight(weight_list):
    weight_list = np.clip(weight_list, 0, 10000)
    weight_list = np.array(weight_list) / np.max(weight_list)  # max-norm to [0,1]
    new_weight = np.exp(weight_list) / sum(np.exp(weight_list))  # per weight of scale
    return new_weight


class NPC(object):
    def __init__(self, tracker_ts, tracker_object):
        super().__init__()
        self.ts = tracker_ts
        self.id = tracker_object.object_id

        self.image_bboxes = {}  # cam_id to image bbox
        self.lidar_bboxes = {}  # part id to lidar bbox
        self.assign_bboxes(tracker_object)
        self.image = {}

    def assign_bboxes(self, object):
        self.assign_image_bboxes(object)

    def assign_image_bboxes(self, object):
        for image_bbox in object.image_bboxes:
            source_topic = image_bbox.source_topic  # /detection/cam{}/pip0
            cam_id = int(source_topic.split('/')[-2][-1])
            self.image_bboxes["cam{}".format(cam_id)] = image_bbox

    def assign_bbox_image(self, cam_id, raw_image, process_func):
        bbox = self.image_bboxes["cam{}".format(cam_id)].bbox_per
        self.image["cam{}".format(cam_id)] = process_func(bbox, raw_image)

    def __repr__(self):
        _msg = f"timestamp = {self.ts}, " + \
               f"ID = {self.id}, " + \
               f"CAM = {list(self.image_bboxes.keys())}"
        return _msg


class NPCCache(object):
    def __init__(self, maxlen=50):
        self.cache = defaultdict(dict)
        self.maxlen = maxlen

    def update(self, ts, npc_id, data):
        self.cache[ts].update({npc_id: data})
        all_ts = sorted([t for t in self.cache])
        latest_ts = all_ts[-self.maxlen:]
        for ts in all_ts:
            if ts not in latest_ts:
                self.cache.pop(ts)

    def get_history(self, last=-5):
        if len(self.cache) < abs(last): return None, None
        all_ts = sorted([t for t in self.cache])
        last_ts = all_ts[last]
        return self.cache[last_ts], last_ts


class ImageCache(object):
    def __init__(self, maxlen=50):
        self.cache = defaultdict(dict)
        self.maxlen = maxlen

    def update(self, ts, data):
        self.cache[ts] = data
        all_ts = sorted([t for t in self.cache])
        latest_ts = all_ts[-self.maxlen:]
        for ts in all_ts:
            if ts not in latest_ts:
                self.cache.pop(ts)

    def get_history(self, last=-5):
        if len(self.cache) < abs(last): return None, None
        all_ts = sorted([t for t in self.cache])
        last_ts = all_ts[last]
        return self.cache[last_ts], last_ts


class ScaleShiftPy(object):
    def __init__(self, scale_range=[0.8, 1.1], num_scale=30, win_size=5, topk=3, expand_ratio=1.2):
        self.scale_list = np.linspace(scale_range[0], scale_range[1], num_scale)
        self.num_scale = num_scale
        self.win_size = win_size
        self.topk = 3
        self.expand_ratio = expand_ratio

    def __call__(self, img, last_img, bbox, last_bbox):
        imgh, imgw = img.shape[:2]
        # take bg to detect bbox
        bbox, last_bbox = expand_bboxes(bbox, last_bbox, imgh=imgh, imgw=imgw, ratio=self.expand_ratio)
        topk_scale_vals, topk_scale = scale_shift(img, last_img, bbox, last_bbox, self.scale_list)
        topk_scale_weight = reweight(topk_scale_vals)
        final_scale = sum(np.array(topk_scale) * topk_scale_weight)
        final_var = (np.array(topk_scale) - final_scale).var()
        return final_scale, final_var

    def __repr__(self):
        msg = "scale range = {}; num scale = {} ".format(self.scale_list, self.num_scale) + \
              "window size = {}; Topk = {}; BG Ratio = {}".format(self.win_szie, self.topk, self.expand_ratio - 1)
        return msg