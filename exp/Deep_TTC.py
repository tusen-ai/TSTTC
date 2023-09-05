#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import ast
import sys

sys.path.append("..")
sys.path.append("../..")
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp

from data.ttc_dataset import TSTTCDataset, get_ttc_loader
from data.data_aug import TrainTransformSeqLevel,TrainTransform,ValTransform

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.archi_name = 'TTCBase'
        # ---------------- model config ---------------- #
        # random seed
        self.seed = 0
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"
        # scale number
        self.scale_num = 20
        # ttc bin or scale bin
        self.ttc_bin = False
        # min max ratio list for different frame gap (corresponding 1.0 and -1.5)
        self.min_max_scale_list = [[0.90, 1.08], [0.80, 1.20], [0.75, 1.25], [0.70, 1.40], [0.65, 1.5]]
        # sequence len
        self.sequence_len = 6
        # max and min scale factor
        ###Note: change the min and max scale manually when using different sequence len!!!
        self.min_scale = self.min_max_scale_list[self.sequence_len - 2][0]
        self.max_scale = self.min_max_scale_list[self.sequence_len - 2][1]
        # kernel size for conv layer in model
        self.ksize_base = 7
        # n*n center shift
        self.shift_size = 3
        # similarity measurement between two feature maps
        self.distance_type = 'dot'
        # center shift or not
        self.shift = True
        # size for grid sample
        self.grid_size = 50
        # grid sampling padding mode
        self.grid_sample_padding = 'zeros'

        # ---------------- dataloader config ---------------- #
        # set worker to 8 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 8
        # (height, width)
        self.input_size = (576, 1024)
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 0
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # only use two image or the whole seq
        self.use_all = False
        # box level training & evaluation or not, could speed training 3x times but decrease accuracy
        self.box_level = True
        # minimal padding size after crop obj in a batch |box level only|
        self.min_size_after_padding = 300
        # downsample by 2x if the box size is larger than this value |box level only|
        self.box_downsample_thresh = 300  # [300,300]
        # padding img size when cropping
        self.receptive_filed = 16
        # use NeRF data or not
        self.use_nerf = False
        # use resample for ttc 0~6 same lane or not
        self.resample = False
        # train set dir
        self.trainset_dir = None
        self.trainAnnoPath = None
        # val set dir
        self.valset_dir = None
        self.valAnnoPath = None
        # nerf data dir
        self.nerf_data_dir = None
        # total seqs for nerf data
        self.nerf_seqs = 0
        # nerf seed for random nerf data
        self.nerf_seed = 0
        # box is normed or not
        self.normed_box = True
        # trainset ratio of totoal dataset
        self.training_data_ratio = 1.0
        # --------------- transform config ----------------- #

        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # enlarge the cur object image
        self.dynamic_ratio_aug = 1.0
        # prob of applying reverse aug
        self.reverse_aug_prob = 0
        # noisy box ratio to the ref box center(means ratio*H or W)
        self.noisy_box_ratio = 0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 1
        # max training epoch
        self.max_epoch = 36
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.01
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.0001 / 2
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 4
        # apply EMA during training
        self.ema = False

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (576, 1024)

    def get_model(self):
        from model.backbone import TTCBase
        from model.ttc_head import TTCHead
        from model.TTCNet import TTCNet

        backbone = TTCBase(dep_mul=self.depth, wid_mul=self.width, kszie=self.ksize_base, act=self.act)
        head = TTCHead(scale_number=self.scale_num, width=self.width,
                       fps=10 / (self.sequence_len - 1), ttc_bin=self.ttc_bin, min_scale=self.min_scale,
                       max_scale=self.max_scale, distance_type=self.distance_type, shift=self.shift,
                       shift_kernel_size=self.shift_size, grid_size=self.grid_size, normed_box=self.normed_box,
                       sequence_len=self.sequence_len)

        def init_model(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.model = TTCNet(backbone, head)

        self.model.apply(init_model)

        return self.model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from exp.lr_scheduler import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False,
                        data_path=None, anno_path=None):
        assert data_path is not None
        assert anno_path is not None
        #data_path = data_path if data_path != '' else self.valset_dir
        # TODO: fix max_boxSize param
        valArgs = {
            'data_path': data_path, 'anno_path': anno_path, 'img_size': self.test_size, 'preproc': ValTransform(),
            'seq_len': self.sequence_len, 'first_last': not self.use_all, 'training': False,
            'receptive_filed': self.receptive_filed, 'box_downsample_thresh': self.box_downsample_thresh,
            'min_size_after_padding': self.min_size_after_padding, 'whole_img': not self.box_level,
            'default_max_scale':self.max_scale,
            'grid_size':self.grid_size
        }
        if self.box_level:
            valArgs['preproc'] = None
        valdataset = TSTTCDataset(**valArgs)
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        dataset = get_ttc_loader(batch_size, data_num_workers=self.data_num_workers,
                                 dataset=valdataset, is_dist=is_distributed, seed=self.seed)
        return dataset

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False,
                      data_path=None, anno_path=None):
        if data_path is None: data_path = self.valset_dir
        anno_path = anno_path if anno_path != None else self.valAnnoPath
        from core.evaluator import TTCEvaluator
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy, data_path, anno_path)
        evaluator = TTCEvaluator(
            dataloader=val_loader,
            img_size=self.input_size,
            sequence_len=self.sequence_len,
            scale_number=self.scale_num,
            fps=10 / (self.sequence_len - 1),
        )
        return evaluator

    def get_data_loader(self, batch_size, is_distributed):
        from core.auxil import wait_for_the_master
        with wait_for_the_master():
            trainArgs = {
                'data_path': self.trainset_dir, 'anno_path': self.trainAnnoPath, 'img_size': self.test_size,
                'preproc': TrainTransform(hsv_prob=self.hsv_prob),
                'seq_len': self.sequence_len, 'first_last': not self.use_all, 'training': True,
                'receptive_filed': self.receptive_filed, 'box_downsample_thresh': self.box_downsample_thresh,
                'min_size_after_padding': self.min_size_after_padding, 'whole_img': not self.box_level,
                'default_max_scale': self.max_scale,
                'grid_size': self.grid_size
            }
            if self.box_level:
                trainArgs['preproc'] = TrainTransformSeqLevel(hsv_prob=self.hsv_prob)
            if self.use_nerf:
                trainArgs['nerf_path'] = self.nerf_data_dir
                trainArgs['nerf_seqs'] = self.nerf_seqs
                trainArgs['nerf_seed'] = self.nerf_seed
            dataset = TSTTCDataset(**trainArgs)
            if is_distributed:
                batch_size = batch_size // dist.get_world_size()
        ttc_train_loader = get_ttc_loader(batch_size, data_num_workers=self.data_num_workers, dataset=dataset,
                                          is_dist=is_distributed)
        return ttc_train_loader

    def get_trainer(self, args):
        from core.trainer import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half=half)

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)
