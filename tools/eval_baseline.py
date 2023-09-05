import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
from baseline.ttcmodule import mse_baseline
from data.ttc_dataset import get_eval_loader,TSTTC
import argparse
def make_parser():
    parser = argparse.ArgumentParser('TTC Baseline')
    parser.add_argument('--path', type=str, default='/mnt/weka/scratch/yuheng.shi/dataset/dataset2023/nips_ver_v2/val', help='path to dataset')
    parser.add_argument('--seq_len', type=int, default=6, help='sequence length')
    parser.add_argument('--scales',type=int,default=125)
    parser.add_argument('--winsize',type=int,default=3)
    return parser

def eval_loader(batch_size=1,seq_len=5,tsttc=None,max_scale = 1.5):
    from data.ttc_dataset import TTCDataset
    valdataset = TTCDataset(
                            img_size=(576, 1024),
                            preproc=None,
                            seq_len=seq_len,
                            first_last=True,
                            training=False,
                            tsttc=tsttc,
                            expand_ratio=1.1,
                            max_scale = max_scale
                            )

    dataset = get_eval_loader(batch_size, data_num_workers=8, dataset=valdataset)
    return dataset


if __name__ == "__main__":
    args = make_parser().parse_args()
    tsttc = TSTTC(args.path)
    eval_loader = eval_loader(seq_len=6,tsttc=tsttc)
    rte,rse = mse_baseline(dataloader=eval_loader,scale_range=[0.65,1.5],num_scale=args.scales,seq_len=args.seq_len,win_size=args.winsize,bbox_thrs=128)
    print('RTE: ',rte)
    print('MiD: ',rse)