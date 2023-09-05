import datetime
import os
import time
from loguru import logger

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import distributed as dist

import sys

sys.path.append("..")
from data.data_prefetcher import DataPrefetcher
from torch.nn.parallel import DistributedDataParallel as DDP

from .auxil import (
    MeterBuffer
, setup_logger
, gpu_mem_usage
, save_checkpoint
, is_parallel
, synchronize, all_reduce_norm, load_ckpt
, get_local_rank
, get_rank
, get_world_size
, adjust_status
)

from exp.Deep_TTC import Exp

class Trainer:
    def __init__(self, exp: Exp, args, eval=False):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_rte = 100

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)
        self.seq_size = [200,200]

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    #
    def train_one_iter(self):
        iter_start_time = time.time()

        inps, dictAnnos, enlarge_boxes, ttc = self.prefetcher.next()
        if inps is None: return
        if ttc.shape[0] == 0:return

        inps = inps.to(self.data_type)
        self.seq_size = inps.shape[-2:]
        data_end_time = time.time()
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, enlarge_boxes, dictAnnos, ttc)
            loss = outputs["total_loss"]
            if loss < 200:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
        )
        logger.info("init prefetcher, this might take minutes or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)


        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )

        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            else:
                raise ValueError("logger must be either 'tensorboard'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best RTE is {:.2f}".format(self.best_rte)
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        #self.prefetcher = DataPrefetcher(self.train_loader)

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            return self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d},{:d}, {}".format(self.seq_size[0],self.seq_size[1], eta_str))
            )

            self.meter.clear_meters()

        # random resizing,
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_rte = ckpt.pop("best_rte", 1)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):

        evalmodel = self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module
        with adjust_status(evalmodel, training=False):
            rte, rse, summary, scale_error_dict = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed
            )

        update_best_ckpt = rte < self.best_rte
        self.best_rte = min(self.best_rte, rte)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/RTE", rte, self.epoch + 1)
                self.tblogger.add_scalar("val/MiD", rse, self.epoch + 1)
                for tmp_key in list(scale_error_dict.keys()):
                    self.tblogger.add_scalar(tmp_key, scale_error_dict[tmp_key], self.epoch + 1)
            logger.info('Average MiD:{}, Average RTE:{}'.format(rse, rte))
            logger.info(summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")
        return rse,summary

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.model.module if is_parallel(self.model) else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_rte": self.best_rte,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
