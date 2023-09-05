import torch
from data.ttc_dataset import ttc_to_scale_ratio,scale_ratio_to_ttc
import numpy as np
from tqdm import tqdm
import math
import time
import itertools
from core.auxil import (
gather,
is_main_process,
synchronize,
time_synchronized,
)

class TTCEvaluator:
    def __init__(
            self,
            dataloader,
            img_size,
            sequence_len,
            scale_number,
            fps,
    ):
        self.dataloader = dataloader
        self.img_size = img_size
        self.sequence_len = sequence_len
        self.scale_number = scale_number
        self.fps = fps
        self.ttc_range = [-20,0,3,6,20]

    def evaluate(self, model, is_distributed=False, half=False,topk=4):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ttc_error_list, scale_error_list = [], []
        analysis_list = []
        result_dict = {}
        progress_bar = tqdm if is_main_process() else iter
        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (imgs, annos, candidate_boxes, ttc_gts_tensor) in enumerate(
                progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs, scale_list, _ = model.forward(imgs, candidate_boxes, annos)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = outputs.reshape(-1, self.scale_number)
                pred_conf, pred_bin = torch.topk(outputs,k=topk,dim=-1)
                pred_conf, pred_bin = pred_conf.cpu(), pred_bin.cpu()
                pred_conf = pred_conf / torch.sum(pred_conf,dim=-1,keepdim=True)
                scale_list = torch.tensor(scale_list)
                pred_scales =  torch.tensor(torch.sum(scale_list[pred_bin] * pred_conf,dim=-1))

                pred_ttcs = scale_ratio_to_ttc(pred_scales,self.fps)
                pred_ttc_list = pred_ttcs.cpu().numpy().tolist()
                ttc_gt = torch.tensor(ttc_gts_tensor).view(-1, 1).flatten()
                scale_gt = torch.tensor([ttc_to_scale_ratio(ttc_gt[i], self.fps) for i in range(ttc_gt.shape[0])])
                ave_ttc_errors, tmp_ttc, ttcs_abs_error = self.compute_error_rate(pred_ttcs, ttc_gt,mid=False)
                ave_scale_errors, tmp_scale, scale_abs_error = self.compute_error_rate(pred_ttcs, ttc_gt,mid=True)

                for box_id, contents in enumerate(zip(annos['metaAnnos'], tmp_ttc, tmp_scale, ttcs_abs_error, scale_abs_error,pred_ttc_list)):
                    raw_obj, ttc_rel, scale_rel, ttc_abs, scale_abs,pred_ttc = contents
                    result_dict[raw_obj.anno_id] = pred_ttc
                    analysis_list.append(np.array(
                        [ttc_rel, ttc_abs, scale_rel, scale_abs, float(ttc_gt[box_id]), float(scale_gt[box_id])]
                    ))
                ttc_error_list.extend(tmp_ttc)
                scale_error_list.extend(tmp_scale)
                
        if is_distributed:
            ttc_error_list = gather(ttc_error_list, dst=0)
            scale_error_list = gather(scale_error_list, dst=0)
            analysis_list = gather(analysis_list, dst=0)
            ttc_error_list = list(itertools.chain(*ttc_error_list))
            scale_error_list = list(itertools.chain(*scale_error_list))
            analysis_list = list(itertools.chain(*analysis_list))


        summary,scale_error_dict = self.log_error(self.cal_error(analysis_list))
        summary += '-----------------------------\n'
        summary += 'Ave inference time: %.4f' % float(inference_time / n_samples) + '\n'
        synchronize()
        return sum(ttc_error_list) / max(len(ttc_error_list),1), sum(scale_error_list) / max(len(scale_error_list),1),summary,scale_error_dict


    def compute_error_rate(self, pred, gt, mid = False):
        assert pred.shape[0] == gt.shape[0]
        relative_errors, abs_error = [], []

        for i in range(pred.shape[0]):
            if mid:
                pred_ttc = max(min(pred[i], 20), -20)
                pred_tmp = ttc_to_scale_ratio(pred_ttc)
                gt_tmp = ttc_to_scale_ratio(gt[i])
                error = float(abs(math.log(pred_tmp) - math.log(gt_tmp)) * 10**4)
            else:
                pred_tmp = max(min(pred[i], 20), -20)
                gt_tmp = gt[i]
                error = float(abs((pred_tmp - gt_tmp) / abs(gt_tmp)) * 100)
            abs_error.append(abs(pred_tmp - gt_tmp))
            relative_errors.append(error)
        return sum(relative_errors) / len(relative_errors), relative_errors, abs_error



    def cal_error(self, data):
        '''

        Args:
            data: N * M dimension, N is number of samples and may be duplicated due to multi camera

        Returns:

        '''
        if not is_main_process():
            return []
        data = np.stack(data)
        ttc_dict = {}

        # show error in different ttc range
        for i in range(len(self.ttc_range[:-1])):
            lower, upper = self.ttc_range[i], self.ttc_range[i + 1]
            # select errors in range [lowe,upper]
            mask_l = np.tile(np.expand_dims(data[:, -2] > lower, axis=1), (1, 4))
            mask_u = np.tile(np.expand_dims(upper > data[:, -2], axis=1), (1, 4))
            errors = np.reshape(data[:, :4][mask_l * mask_u], [-1, 4])
            ttc_dict['ttc ' + str(lower) + '~' + str(upper)] = errors

        return [ttc_dict]
    def log_error(self, result_list: list):
        if not is_main_process():
            return '',{}

        name_list = ['ttc']
        summary = ''
        scale_error_dict = {}
        for result_dic, name in zip(result_list, name_list):
            key_list, value_list = [], []
            for key in list(result_dic.keys()):
                summary += '-----------------------------\n'
                summary += key + ":\n"
                if 'ttc' in key:
                    scale_error_dict[key] = 0
                data = result_dic[key]
                ttc_error_relative = np.abs(data[:, 0])
                ttc_error_abs = data[:, 1]
                scale_error_relative = data[:, 2]
                scale_error_abs = data[:, 3]
                counts = int(ttc_error_abs.shape[0])
                key_list.append(key)
                value_list.append(0)
                if counts != 0:
                    summary += 'total sample number: %i' % (counts) + '\n'
                    summary +='Ave relative ttc error: %.2f %%' % float(np.average(ttc_error_relative))+ '\n'
                    summary +='Ave abs ttc error: %.2f' % float(np.average(ttc_error_abs))+ '\n'
                    summary +='Ave motion in depth: %.4f %%' % float(np.average(scale_error_relative))+ '\n'
                    summary +='Ave abs scale error: %.4f' % float(np.average(scale_error_abs))+ '\n'

                    if 'ttc' in key:
                        scale_error_dict[key] = float(np.average(scale_error_relative))
                    value_list[-1] = counts
                else:
                    summary +='No samples in this key'+ '\n'

        return summary,scale_error_dict
