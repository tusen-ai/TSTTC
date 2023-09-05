import copy
import os
import random
import pickle
import uuid
import itertools
from loguru import logger

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data.sampler import Sampler, BatchSampler, SequentialSampler
import torch.distributed as dist
from .utils import get_crop_size, get_cropped_imgs,padding_image_to_same
from .dataset_api import TSTTC

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".JPEG"]
XML_EXT = [".xml"]
PKL_EXT = [".pkl"]


def load_pickle(f):
    return pickle.load(open(f, 'rb'))

def get_file_list(path, type_list):
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in type_list:
                file_names.append(apath)
    return file_names

def scale_ratio_to_ttc(scale_ratio, fps=10):
    ttc = 1 / ((fps * (1 / scale_ratio - 1)) + 1e-9)
    return ttc

def ttc_to_scale_ratio(ttc, fps=10):
    scale_ratio = 1 / ((1 / (ttc * fps) + 1) + 1e-6)
    return scale_ratio


def remove_useless_info(tsttc,first_last=True,seq_len=6):
    if isinstance(tsttc, TSTTC):
        annos = tsttc.annos
        if first_last:
            for key,val in annos.items():
                #set others to None except -1 and -seq_len
                for i in range(-len(val),-1):
                    if i != -seq_len: val[i] = None

            for key,val in tsttc.frameSeqs.items():
                for seqs in val:
                    for i in range(-len(val), -1):
                        if i != -seq_len: val[i] = None


class TSTTCDataset(torchDataset):
    def __init__(
            self,
            data_path = '',
            anno_path = '',
            img_size=(576, 1024),
            preproc=None,
            seq_len=6,
            first_last=True,
            training=True,
            whole_img = False,
            box_downsample_thresh = 300,
            receptive_filed = None,
            min_size_after_padding = 300,
            training_data_ratio = 1.0,
            expand_ratio = 1.1,
            default_max_scale = 1.25,
            nerf_path=None,
            nerf_seqs = 0,
            nerf_seed = 0,
            **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.anno_path = anno_path
        self.nerf_path = nerf_path
        self.img_size = img_size
        self.preproc = preproc
        self.first_last = first_last
        self.training = training
        self.seq_len = seq_len
        logger.info("Loading TSTTC dataset from {}...".format(self.data_path))
        self.tsttc = TSTTC(self.data_path, self.anno_path,
                           sequence_len=6)
        self.whole_img = whole_img
        self.box_downsample_thresh = box_downsample_thresh
        self.min_size_after_padding = min_size_after_padding
        self.receptive_filed = receptive_filed
        self.anno_ids = self.tsttc.getAnnoIds()
        if training and training_data_ratio < 1.0:
            random.shuffle(self.anno_ids)
            self.anno_ids = self.anno_ids[:int(len(self.anno_ids)*training_data_ratio)]
        self.annos = self.tsttc.loadAnnos(self.anno_ids)
        self.img_ids = self.tsttc.getImgSeqIds()
        self.imgSeqsAnnos = self.tsttc.loadImgSeqs(self.img_ids)
        if self.nerf_path is not None:
            #note: only support box level training!
            logger.info("Loading TSTTC dataset from {}...".format(self.nerf_path))
            self.nerfttc = TSTTC(self.nerf_path)
            nerf_ids = self.nerfttc.getAnnoIds()
            random.seed(nerf_seed)
            random.shuffle(nerf_ids)
            self.nerf_ids = nerf_ids[:nerf_seqs]
            self.annos = self.annos + self.nerfttc.loadAnnos(self.nerf_ids)

        self.expand_ratio = expand_ratio
        self.default_max_scale = default_max_scale
        self.grid_size = kwargs.get('grid_size',50)


    def __len__(self):
        if self.whole_img:
            return len(self.tsttc.frameSeqs)
        return len(self.annos)

    def resize_img(self, img,):
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img
    def pull_item(self, index):
        result_dict = {'imgPair':[],'refBoxAnnos':[],'curBoxAnnos':[],'ttc_imu':[],\
                       'curAnnos':[],'dynamicRanges':[],'frame_gap':[],'masks':[]}
        if self.first_last:
            cur_idx,ref_idx = -1,0
        else: #TODO fix this
            print('not implemented')
            exit(0)
        if self.whole_img:
            frameSeq = self.tsttc.frameSeqs[index]
            for i in range(len(frameSeq)):
                try:
                    objAnnoRef,objAnnoCur =  frameSeq[i][-self.seq_len:][ref_idx],frameSeq[i][-self.seq_len:][cur_idx]
                except Exception as e:
                    print('load: ',frameSeq[i])
                    logger.warning('fail to load image pair: %s' % (e))
                    return result_dict

                if 'ttc_imu' not in objAnnoCur:
                    logger.warning('no ttc imu: %s' % (objAnnoCur))
                    return result_dict
                if i == 0:
                    try:
                        imgRef = self.resize_img(cv2.imread(objAnnoRef['img_path']))
                        imgCur = self.resize_img(cv2.imread(objAnnoCur['img_path']))
                    except AttributeError:
                        logger.warning(
                            'fail to load image pair: %s or %s' % (objAnnoRef['img_path'], objAnnoCur['img_path']))
                        return result_dict
                    result_dict['imgPair'].extend([imgRef,imgCur])
                max_scale = self.default_max_scale
                candidate_boxes = get_crop_size(objAnnoRef['box2d'], objAnnoCur['box2d'],max_scale=max_scale,expand_ratio=self.expand_ratio)
                if candidate_boxes is not None:
                    result_dict['refBoxAnnos'].append(candidate_boxes[0])
                    result_dict['curBoxAnnos'].append(candidate_boxes[3])
                    result_dict['ttc_imu'].append(objAnnoCur['ttc_imu'])
                    result_dict['curAnnos'].append(objAnnoCur)
                    result_dict['frame_gap'].append(self.seq_len-ref_idx-1)
        else:
            result_dict['min_size_after_padding'] = self.min_size_after_padding
            try:
                objAnnoRef = copy.deepcopy(self.annos[index][-self.seq_len:][ref_idx])
                objAnnoCur = copy.deepcopy(self.annos[index][-self.seq_len:][cur_idx])
            except IndexError:
                logger.warning('fail to load image pair: %s' % index)
                return result_dict
            try:
                imgRef = self.resize_img(cv2.imread(objAnnoRef['img_path']))
                imgCur = self.resize_img(cv2.imread(objAnnoCur['img_path']))
            except Exception as e:
                logger.warning('fail to load image pair: %s or %s'%(objAnnoRef['img_path'],objAnnoCur['img_path']))
                return result_dict

            max_scale = self.default_max_scale
            candidate_boxes = get_crop_size(objAnnoRef['box2d'], objAnnoCur['box2d'], max_scale=max_scale,
                                            expand_ratio=self.expand_ratio)
            if candidate_boxes is not None:
                result_dict['refBoxAnnos'].append(candidate_boxes[0])
                result_dict['curBoxAnnos'].append(candidate_boxes[3])
                result_dict['ttc_imu'].append(objAnnoCur['ttc_imu'])
                result_dict['curAnnos'].append(objAnnoCur)
                result_dict['imgPair'], result_dict['ref_padding'], result_dict['cur_padding'] = get_cropped_imgs(imgRef, imgCur, result_dict['refBoxAnnos'][0],
                                                                result_dict['curBoxAnnos'][0], self.receptive_filed,
                                                                max_unit_size=self.box_downsample_thresh,
                                                                )
                result_dict['frame_gap'].append(self.seq_len - ref_idx - 1)

            else:
                logger.warning('box out of img after enlarging: %s' % objAnnoCur['img_path'])
        return result_dict

    def __getitem__(self, index):
        result_dict = self.pull_item(int(index))
        if self.preproc is not None and len(result_dict['refBoxAnnos']) > 0:
            result_dict['imgPair'][0],result_dict['refBoxAnnos'] = self.preproc(result_dict['imgPair'][0],np.array(result_dict['refBoxAnnos']),self.img_size)
            result_dict['imgPair'][1],result_dict['curBoxAnnos'] = self.preproc(result_dict['imgPair'][1],np.array(result_dict['curBoxAnnos']),self.img_size)
        return result_dict
class TTCDataset(torchDataset):
    '''
    TTC sequence
    '''

    def __init__(
            self,
            data_path='',
            img_size=(576, 1024),
            preproc=None,
            seq_len=5,
            first_last=True,
            training=True,
            debug_flag=True,
            add_affine=False,
            use_nerf = False,
            resample = False,
            expand_ratio = 1.1,
            nerf_data_path = '',
            nerf_ratio = 0.1,
            nerf_seed = 0,
            max_scale = 1.5,
            tsttc = None
    ):
        super().__init__()
        self.data_path = data_path
        self.img_size = img_size
        self.preproc = preproc
        self.first_last = first_last
        self.training = training
        self.affine = add_affine
        self.seq_len = seq_len
        self.use_nerf = use_nerf
        self.resample = resample
        self.nerf_data_path = nerf_data_path
        self.nerf_seed = nerf_seed
        self.nerf_ratio = nerf_ratio
        self.expand_ratio = expand_ratio
        self.default_max_scale = max_scale
        #all_anno, seqs = self.reformat_anno(seq_len, self.data_path)

        #load dataset
        if tsttc:
            self.tsttc = tsttc
            condition_ids = self.tsttc.getAnnoIds(cam_ids=[1,3,4,8,9])
            seqs = self.tsttc.loadAnnos(condition_ids)
            seqs = [seq[-seq_len:] for seq in seqs]
        else:
            if type(data_path) is str:
                all_anno, seqs = self.reformat_anno(seq_len,self.data_path)
            else:
                seqs = []
                print(self.data_path)
                for tmp_pth in self.data_path:
                    _anno, _seqs = self.reformat_anno(seq_len,tmp_pth)
                    seqs = seqs + _seqs
        # use nerf data
        if self.use_nerf and self.nerf_ratio > 0 and self.training:
            _,seqs_nerf = self.reformat_anno(seq_len,path=self.nerf_data_path,nerf=True)
            if nerf_ratio<1:
                seqs_nerf = seqs_nerf[:int(nerf_ratio*len(seqs))]
            else:
                seq_03,seq_36 = [],[]
                for seq in seqs_nerf:
                    if 0<seq[-1].ttc_imu<=3 and len(seq_03)<nerf_ratio:seq_03.append(seq)
                    elif 3<seq[-1].ttc_imu<=6 and len(seq_36)<nerf_ratio:seq_36.append(seq)
                seqs_nerf = seq_03+seq_36
            seqs = seqs + seqs_nerf

        if training:
            random.seed(42)
            random.shuffle(seqs)
            self.seqs = seqs
        else:
            self.seqs = seqs
            self.first_last = True

    def __len__(self):
        return len(self.seqs)

    def reformat_anno(self, seq_len=5,path = '',nerf = False):
        if nerf:
            anno_names = get_file_list(path, PKL_EXT)
            random.seed(42-self.nerf_seed)
            random.shuffle(anno_names)
        else:
            anno_names = get_file_list(path, PKL_EXT)
        anno_list = []
        input_list = []

        for anno_name in anno_names:  # bag level
            contents = load_pickle(anno_name)
            if contents == []: continue
            for content in contents:
                bag_stamp = content[-1].bag_stamp.split('/')[-1]
                img_name = str(content[-1].ts) + '.jpg'
                cam = 'cam' + str(content[-1].cam_id)
                img_path = os.path.join(path, bag_stamp, cam, img_name)

                ttc_imu = content[-1]['ttc_imu']
                if not os.path.exists(img_path):
                    print('file not exist:', img_path)
                    continue
                for frame_idx in range(len(content)) :
                    content[frame_idx].bag_stamp =  os.path.join(path, bag_stamp)
                    content[frame_idx].img_path = os.path.join(path, bag_stamp, cam, str(content[frame_idx].ts) + '.jpg')
                input_list.append(content[-seq_len:])

        return anno_list, input_list

    def pull_item(self, seq):
        imgs = []
        all_boxes = []
        ttc_gts_dict = {}
        if self.first_last:
            seq = [seq[0], seq[-1]]
            ttc_gts_dict['gap'] = self.seq_len-1
        else: #random gap
            frame_gap = random.randint(1,self.seq_len-1)
            seq = [seq[-1-frame_gap], seq[-1]]
            ttc_gts_dict['gap'] = frame_gap
        for img_annos in seq:
            cam = 'cam' + str(img_annos.cam_id)
            img_name = str(img_annos.ts) + '.jpg'
            img_path = os.path.join(img_annos.bag_stamp, cam, img_name)
            img = cv2.imread(img_annos.img_path)
            height, width = img.shape[:2]
            img_info = (height, width)
            r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
            img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            boxes = np.array([img_annos.box2d])
            imgs.append(img)
            all_boxes.append(boxes)
        ttc_gt,_gt_annos = [seq[-1].ttc_imu], [seq[-1]]
        enlarge_boxes,ttc_gts,gt_annos = [],[],[]
        for i in range(all_boxes[0].shape[0]):# for each box pair
            #fecth the box pair
            res = get_crop_size(all_boxes[0][i], all_boxes[1][i],expand_ratio=self.expand_ratio,max_scale=self.default_max_scale)
            if res is not None:
                enlarge_bbox, enlarge_cur_bbox, bbox, cur_bbox = res
                enlarge_boxes.append([enlarge_bbox, bbox, cur_bbox])
                ttc_gts.append(ttc_gt[i])
                gt_annos.append(_gt_annos[i])
            else:
                print('expand box fail:!!!!')
        ttc_gts_dict['ttc_gts'] = ttc_gts
        if self.training:
            return imgs, all_boxes, enlarge_boxes, ttc_gts_dict
        else:
            return imgs, all_boxes, enlarge_boxes, ttc_gts_dict, gt_annos

    def __getitem__(self, item):
        seq = self.seqs[item]
        if self.training:
            imgs, targets, enlarge_boxes, ttc_gts_dict = self.pull_item(seq)
        else:
            imgs, targets, enlarge_boxes, ttc_gts_dict, gt_annos = self.pull_item(seq)
        if self.preproc is not None:
            _imgs, _targets = [], []
            for i in range(len(imgs)):
                if self.affine:#extra augmentation
                    if i != len(imgs) - 1:
                        enlarge_boxes = np.array(enlarge_boxes)
                        enlarge_boxes = enlarge_boxes.reshape([-1, 4])
                        img, enlarge_boxes = self.preproc(imgs[i], enlarge_boxes, self.img_size, item)
                        enlarge_boxes = enlarge_boxes.reshape([-1, 3, 4])
                        enlarge_boxes = enlarge_boxes.tolist()
                    else:
                        img, _ = self.preproc(imgs[i], [], self.img_size, item)
                else:
                    img, _ = self.preproc(imgs[i], targets[i], self.img_size)
                _imgs.append(img)
            _targets = targets
        else:
            _imgs, _targets = imgs, targets
        if self.training:
            return _imgs, _targets, enlarge_boxes, ttc_gts_dict
        else:
            return _imgs, _targets, enlarge_boxes, ttc_gts_dict, gt_annos


class TrainSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)


class TestSampler(SequentialSampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class TTCBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size

def ttc_collate_fn(batch):
    cur_valid_idx = 0
    imgs, boxes, ttcs,padding_sizes,dictAnnos = [], [], [],[],{'metaAnnos':[],'dynamicRanges':[],'frame_gap':[],'masks':[]}
    for sample in batch:
        if len(sample['refBoxAnnos']):
            imgs.extend(sample['imgPair'])
            ttcs.extend(sample['ttc_imu'])
            dictAnnos['metaAnnos'].extend(sample['curAnnos'])
            dictAnnos['frame_gap'].extend(sample['frame_gap'])
            if 'dynamicRanges' in sample:
                dictAnnos['dynamicRanges'].extend(sample['dynamicRanges'])
            if 'masks' in sample:
                dictAnnos['masks'].extend(sample['masks'])
            roi_prefix_ref = torch.ones([len(sample['refBoxAnnos']), 1], dtype=torch.float32) * (cur_valid_idx)
            roi_prefix_cur = torch.ones([len(sample['curBoxAnnos']), 1], dtype=torch.float32) * (cur_valid_idx)
            roi_box_ref = torch.cat([roi_prefix_ref, torch.tensor(sample['refBoxAnnos'])], dim=1)
            roi_box_cur = torch.cat([roi_prefix_cur, torch.tensor(sample['curBoxAnnos'])], dim=1)
            roi_boxes = torch.stack([roi_box_ref, roi_box_cur], dim=0).permute(1, 0, 2).flatten(0, 1)
            if 'min_size_after_padding' in sample:
                padding_sizes.append(sample['ref_padding'])
                padding_sizes.append(sample['cur_padding'])
            boxes.append(roi_boxes)
            cur_valid_idx += 1
    if len(boxes)==0:
        return None,None,None,None
    boxes = torch.cat(boxes, dim=0)
    if len(padding_sizes):#only box area
        default_padding_size = [sample['min_size_after_padding'],sample['min_size_after_padding']]
        imgs, orininal_box_list = padding_image_to_same(imgs, padding_sizes, dst_size=default_padding_size)
        normed_orininal_boxes = torch.tensor(orininal_box_list,dtype=torch.float32)
        H,W = imgs[0].shape[1:]
        roi_idx = boxes[:,:1]
        normed_orininal_boxes[:,1::2] = normed_orininal_boxes[:,1::2]/H
        normed_orininal_boxes[:,0::2] = normed_orininal_boxes[:,0::2]/W
        normed_orininal_boxes = torch.cat([roi_idx,normed_orininal_boxes],dim=1)
        boxes = normed_orininal_boxes
    # TODO del None in final version
    if len(imgs) == 0:
        tensor_imgs= None
    else:
        tensor_imgs = [torch.tensor(img) for img in imgs]
        if len(dictAnnos['dynamicRanges']):
            dictAnnos['dynamicRanges'] =[torch.tensor(tmpRange) for tmpRange in dictAnnos['dynamicRanges']]
            dictAnnos['dynamicRanges'] = torch.stack(dictAnnos['dynamicRanges'],dim=0)
        if len(dictAnnos['masks']):
            dictAnnos['masks'] = torch.stack(dictAnnos['masks'],dim=0)
    return torch.stack(tensor_imgs, dim=0),dictAnnos, boxes, torch.tensor(ttcs)



def collate_fn(batch):
    tar = []
    imgs = []
    enlarge_boxes = []
    ttc_gts, ttc_gaps = [], []
    ttc_gts_dict = {}
    for sample in batch:
        tar_ori, ttc_ori, ttc_gap = [], [], []

        for img in sample[0]:
            imgs.append(torch.tensor(img))
        for boxes in sample[1]:
            tar_ori.append(torch.tensor(boxes))

        ttc_ori.append(torch.tensor(sample[3]['ttc_gts']))
        ttc_gap.append(torch.tensor(sample[3]['gap']))
        tar.extend(tar_ori)
        enlarge_boxes.append(sample[2])
        ttc_gts.extend(ttc_ori)
        ttc_gaps.extend(ttc_gap)
    ttc_gts_dict['ttc_gts'], ttc_gts_dict['gap'] = ttc_gts, ttc_gaps
    return torch.stack(imgs), tar, enlarge_boxes, ttc_gts_dict

def collate_fn_eval(batch):
    tar = []
    imgs = []
    enlarge_boxes = []
    ttc_gts,ttc_gaps = [],[]
    annos = []
    ttc_gts_dict = {}
    for sample in batch:
        tar_ori, ttc_ori, ttc_gap = [], [], []

        for img in sample[0]:
            imgs.append(torch.tensor(img))
        for boxes in sample[1]:
            tar_ori.append(torch.tensor(boxes))

        ttc_ori.append(torch.tensor(sample[3]['ttc_gts']))
        ttc_gap.append(torch.tensor(sample[3]['gap']))

        tar.extend(tar_ori)
        enlarge_boxes.append(sample[2])
        ttc_gts.extend(ttc_ori)
        ttc_gaps.extend(ttc_gap)
        annos.extend(sample[4])
    ttc_gts_dict['ttc_gts'], ttc_gts_dict['gap'] = ttc_gts, ttc_gaps
    return torch.stack(imgs), tar, enlarge_boxes, ttc_gts_dict, annos

def get_train_loader(batch_size, data_num_workers, dataset,sequence_flag=False):
    fn = collate_fn
    sampler = TTCBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn': fn
    }
    ttc_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return ttc_loader


def get_eval_loader(batch_size, data_num_workers, dataset,sequence_flag=False):
    fn = collate_fn_eval
    sampler = TTCBatchSampler(TestSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn': fn
    }
    ttc_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    return ttc_loader

def get_ttc_loader(batchSize,data_num_workers,dataset,is_dist = False,seed=0):
    if dataset.training:
        if seed is None: seed = 0
        InfSampler = InfiniteSampler(len(dataset), seed=seed)
        sampler = TTCBatchSampler(sampler=InfSampler,batch_size=batchSize,drop_last=False)
    else:
        if is_dist:
            sampler = TTCBatchSampler(torch.utils.data.distributed.DistributedSampler(dataset,shuffle=False)
                                      , batchSize, drop_last=False)
        else:
            sampler = TTCBatchSampler(TestSampler(dataset),batchSize,drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        "collate_fn": ttc_collate_fn
    }

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    if dataset.training:
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    ttc_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return ttc_loader

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)