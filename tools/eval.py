import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
import os
from exp.build import get_exp
from core.auxil import setup_logger
from core.launch import launch
from core.auxil import configure_nccl,configure_omp,configure_module,get_num_devices,get_local_rank
from torch.nn.parallel import DistributedDataParallel as DDP

def make_parser():
    parser = argparse.ArgumentParser("TTC Eval parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default='',
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default='' , type=str, help="checkpoint file")
    parser.add_argument("--dataset", default='val', type=str,
                        help="dataset for evaluation")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--box_level",
        help="use whole imgae for inference or not",
        action="store_true",
        default=True,
    )

    return parser

@logger.catch
def main(exp, args, num_gpu):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )
    is_distributed = num_gpu > 1
    configure_nccl()
    rank = get_local_rank()

    cudnn.benchmark = True
    model = exp.get_model()
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)
    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()
    evaluator = exp.get_evaluator(args.batch_size, 0)
    ckpt_file = args.ckpt
    logger.info("loading checkpoint from {}".format(ckpt_file))
    loc = "cuda:{}".format(rank)
    ckpt = torch.load(ckpt_file, map_location=loc)
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    rte, rse, summary,_ = exp.eval(
        model,evaluator,is_distributed,half=args.fp16
    )
    logger.info('Average RSE:{}, Average RTE:{}'.format(rse, rte))
    logger.info(summary)



if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.box_level = args.box_level
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
