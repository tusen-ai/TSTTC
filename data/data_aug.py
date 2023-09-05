import time

import cv2
import numpy as np
import math
import random

def xyxy2cxcywh(bboxes):
    bboxes[2] = bboxes[2] - bboxes[0]
    bboxes[3] = bboxes[3] - bboxes[1]
    bboxes[0] = bboxes[0] + bboxes[2] * 0.5
    bboxes[1] = bboxes[1] + bboxes[3] * 0.5
    return bboxes

def cxcywh2xyxy(bboxes):
    w,h = bboxes[2], bboxes[3]
    cx,cy =  bboxes[0], bboxes[1]
    bboxes[0] = cx - w * 0.5
    bboxes[1] = cy - h * 0.5
    bboxes[2] = cx + w * 0.5
    bboxes[3] = cy + h * 0.5
    return bboxes

def get_aug_params(value, center=0,seed = 0):
    random.seed(seed)
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates

def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
    seed = 0
):
    # targets = [cls, xyxy]
    random.seed(seed)
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        #M[0,-1],M[1,-1] = M[0,-1]/width,M[1,-1]/height
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy[:,0],xy[:,1] = xy[:,0]*width, xy[:,1]*height
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        xy[:,0::2],xy[:,1::2] = xy[:,0::2] / width, xy[:,1::2] / height
        # filter candidates
        # i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        # targets = targets[i]
        targets[:, :4] = xy#[i]

    return img, targets

def _mirror(image, boxes, prob=0.5,seed = 0):
    _, width, _ = image.shape
    random.seed(seed)
    if random.random() < prob:
        image = cv2.flip(image,1)
        boxes[:, [0,2]] = 1 - boxes[:, [2,0]]

    return image, boxes
def reverse_seq(objAnnoCur,objAnnoRef,prob):
    if random.random() < prob and objAnnoCur['ttc_imu']<0:
        objAnnoRef, objAnnoCur = objAnnoCur, objAnnoRef
        objAnnoCur['ttc_imu'] = -objAnnoCur['ttc_imu']
    return objAnnoCur,objAnnoRef

def nosiy_bbox(objAnno, nosiy_ratio):
    cxcywh = xyxy2cxcywh(objAnno['box2d'])
    cxcywh[0] += cxcywh[2] * random.uniform(-nosiy_ratio, nosiy_ratio)
    cxcywh[1] += cxcywh[3] * random.uniform(-nosiy_ratio, nosiy_ratio)
    objAnno['box2d'] = cxcywh2xyxy(cxcywh)
    return objAnno

def mannul_transform(img):
    H,W,C = img.shape
    res = np.zeros((C,H,W))
    for i in range(H):
        for k in range(C):
            for j in range(W):
                res[k,i,j] = img[i,j,k]
    return res
def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = cv2.cvtColor(padded_img, cv2.COLOR_RGB2BGR)
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

class TrainTransform:
    def __init__(self, flip_prob=0.5, hsv_prob=1.0,legacy = False):
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()

        image_o = image.copy()
        height_o, width_o, _ = image_o.shape
        # bbox_o: [xyxy] to [c_x,c_y,w,h]

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = image,boxes #_mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        #TODO: fix the resize code
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes *= r_
        targets_t = boxes

        return image_t, targets_t

class TrainTransformSeqLevel:
    def __init__(self, hsv_prob=1.0,legacy=False):
        self.hsv_prob = hsv_prob
        self.legacy = legacy
    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        if random.random() < self.hsv_prob:
            augment_hsv(image)

        if self.legacy:
            image = image[::-1, :, :].copy()
            image -= np.array([151.98584687, 162.89110864, 95.53919893]).reshape(3, 1, 1)
            image /= np.array([44.31244603, 48.31745843, 37.1085108]).reshape(3, 1, 1)


        image_t = image
        return image_t, boxes

class TrainTransformAug:
    def __init__(self, hsv_prob=1.0, degrees=10.0, translate=0.1,scale=(0.8,1.2),shear=2.0,flip = 0.5):
        self.hsv_prob = hsv_prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.flip = flip


    def __call__(self, image, targets, input_dim,seed):
        if targets != []:
            targets = np.array(targets)
            boxes = targets[:, :4].copy()
        else:
            boxes = np.array([[0,0,0,0]])

        image_t = image
        targets_t = boxes
        input_h, input_w = input_dim[0], input_dim[1]

        image_t, targets_t = random_perspective(
            image_t,
            targets_t,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            seed=seed
        )

        image_t, targets_t = _mirror(image_t, targets_t, seed)
        seed = time.time()
        random.seed(seed)
        if random.random() < self.hsv_prob:
            augment_hsv(image)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        return image_t, targets_t

class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network
    dimension -> tensorize -> color adj
    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, targets, input_size):
        img, r_ = preproc(img, input_size, self.swap)
        boxes = targets[:, :4].copy()
        if self.legacy:
            img = img[::-1, :, :].copy()
            img -= np.array([151.98584687, 162.89110864, 95.53919893]).reshape(3, 1, 1)
            img /= np.array([44.31244603, 48.31745843, 37.1085108]).reshape(3, 1, 1)
        boxes *= r_
        targets_t = boxes

        return img, targets_t