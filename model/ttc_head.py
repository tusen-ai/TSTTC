from torch import nn
from .network_blocks import FocalLoss
from torchvision.ops import roi_align
import torch
from data.ttc_dataset import ttc_to_scale_ratio, scale_ratio_to_ttc


class TTCHead(nn.Module):
    def __init__(
            self,
            scale_number=10,
            in_channel=24,
            fps=None,
            ttc_bin=False,
            min_scale=0.8,
            max_scale=1.2,
            shift = False,
            shift_kernel_size = 3,
            grid_size = -1,
            use_focal_loss = False,
            normed_box = False,
            sequence_len = 6,
            smoother_factor = 1.0,
            **kwargs
    ):
        super().__init__()
        #scale_number = scale_bin + 1
        if use_focal_loss:
            self.loss = FocalLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.scale_number = scale_number
        self.in_channel = in_channel
        self.ttc_bin = ttc_bin
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.total_stride = 1
        self.shift = shift
        self.shift_kernel_size = shift_kernel_size
        self.grid_size = grid_size
        self.fps = fps
        self.sequence_len = sequence_len
        self.normed_box = normed_box
        self.scale_preds = nn.Linear(scale_number, scale_number)
        self.smoother_factor = smoother_factor # training only

    def forward(self, xin, tar_boxes,ref_boxes,ttc_imu =None,**kwargs):
        '''

        :param xin: input features from backbone
        :param tar_boxes: input boxes in roi_align format #[K,5]
        :param ref_boxes:
        :return: predicted scale confidences
        '''

        C,H,W = xin.shape[-3:]
        G,S,Box = self.grid_size,self.scale_number, tar_boxes.shape[0]
        scale_list = self.get_scale_list(S)
        ref_boxes,tar_boxes = self.boxes_sample(ref_boxes,tar_boxes,scale_list,H,W)
        ref_boxes,tar_boxes = ref_boxes.type_as(xin),tar_boxes.type_as(xin)
        tar_features = roi_align(xin[1::2, ], tar_boxes, (G, G))
        tar_tensor = tar_features.view([-1, 1, C, G, G])  # boxes,1,C,H,W
        tar_tensor = torch.flatten(tar_tensor, start_dim=3).permute([0,1,3,2]).unsqueeze(3)#boxes,1,H*W,1,C
        if self.shift:
            ref_features = roi_align(xin[::2, ], ref_boxes, (G+self.shift_kernel_size-1, G+self.shift_kernel_size-1))
            ref_features = self.shift_split(ref_features).view([Box,S,-1,C,G,G])
            ref_features = ref_features.reshape([Box,-1,C,G,G])
            ref_features = ref_features.flatten(start_dim=-2).permute([0,1,3,2]).unsqueeze(-1) #boxes,scale_number*shift_kernel_size**2,H*W,C,1
        else:
            ref_features = roi_align(xin[::2, ], ref_boxes, (G, G)) #boxes*scale_number,C,H,W
            ref_features = ref_features.view([Box,-1,xin.shape[1], G, G])#boxes,scale_number,C,H,W
            ref_features = ref_features.flatten(start_dim=-2).permute([0,1,3,2]).unsqueeze(-1)#boxes,scale_number,H*W,C,1
        #TODO add codes for other distance type here
        simlarities_map = torch.matmul(tar_tensor, ref_features).squeeze(-1).squeeze(-1) #boxes,scale_number,H*W

        simlarities_scale = torch.mean(simlarities_map,dim=-1) #boxes,scale_number
        if self.shift:
            simlarities_scale = torch.max(simlarities_scale.view([-1,self.scale_number, self.shift_kernel_size ** 2]), dim=-1).values

        predictions = self.scale_preds(simlarities_scale)
        if self.training:
            predictions = predictions.view(-1,1)
            if 'dictAnnos' in kwargs:
                dictAnnos = kwargs['dictAnnos']
                frame_gap = dictAnnos['frame_gap']
            else:
                frame_gap = self.sequence_len-1
            gt_one_hot = self.gt_to_one_hot(ttc_imu, gap=frame_gap,scale_list=scale_list)
            scale_loss = self.get_loss(predictions, gt_one_hot)
            return scale_loss
        return predictions.sigmoid(), scale_list, None

    def shift_split(self,ref_features):
        '''
        split the reference features into shift_kernel_size*shift_kernel_size parts
        :param ref_features: [N,C,self.grid_size+self.shift_kernel_size-1, self.grid_size+self.shift_kernel_size-1]
        :return: [N,self.shift_kernel_size**2,C,H,W]
        '''
        tmp_list = []
        for i in range(self.shift_kernel_size):
            for j in range(self.shift_kernel_size):
                tmp_list.append(ref_features[:,:,i:i+self.grid_size,j:j+self.grid_size])
        ref_features = torch.stack(tmp_list,dim=1) #N,shift_kernel_size*shift_kernel_size,C,H,W
        return ref_features

    def get_loss(self, predictions, gts):
        gts = gts.type_as(predictions)
        # print(predictions)
        sample_pairs = gts.shape[0] / self.scale_number
        scale_loss = (
                         self.loss(predictions, gts)
                     ).sum() / sample_pairs

        outputs = predictions.reshape(-1, self.scale_number)
        gts = gts.reshape(-1, self.scale_number)
        _, gt_bin = torch.max(gts, -1, keepdim=False)
        return scale_loss

    def gt_to_one_hot(self, gts, gap, scale_list):
        if type(gap) is list:
            scale_gt = [ttc_to_scale_ratio(ttc, fps=10 / tmp_gap) for ttc,tmp_gap in zip(gts,gap)]
        else:
            scale_gt = [ttc_to_scale_ratio(ttc, fps=10 / gap) for ttc in gts]
        scale_number,gt_number,range_list = self.scale_number,len(gts),scale_list
        range_tensor = torch.tensor(range_list).view([-1, scale_number])
        range_tensor = range_tensor.repeat(gt_number, 1)
        gts_tensor = torch.tensor(scale_gt).view(gt_number, -1).repeat([1, scale_number]).type_as(range_tensor)

        ones_mat = torch.ones_like(gts_tensor)
        zero_mat = torch.zeros_like(ones_mat)
        dist_mat = torch.abs(range_tensor - gts_tensor)
        min_bin = self.smoother_factor*(self.max_scale - self.min_scale) / (self.scale_number - 1)
        dist_tensor = torch.where(dist_mat > min_bin, zero_mat, min_bin - dist_mat) * (1 / min_bin)
        gt_one_hot = dist_tensor.view(-1, 1)
        return gt_one_hot

    def get_scale_list(self, S):
        return torch.linspace(self.min_scale, self.max_scale, S)

    def boxes_sample(self,ref_boxes,tar_boxes,scale_list,H=576,W=1024):
        '''
        convert the boxes to multi-scale format
        :param tar_boxes: [K,5]
        :param ref_boxes: [K,5]
        :return: [K,scale_number,5]
        '''
        def to_algin_space(center_boxes,hw_boxes,scale_list):
            box_num = center_boxes.shape[0]
            box_xx, box_yy = torch.tensor([-0.5 * W, 0.5 * W]).type_as(ref_boxes), torch.tensor([-0.5 * H, 0.5 * H]).type_as(ref_boxes)
            box_xx = box_xx.unsqueeze(0).unsqueeze(0).repeat(box_num, scale_list.shape[-1],1)
            box_yy = box_yy.unsqueeze(0).unsqueeze(0).repeat(box_num, scale_list.shape[-1],1)  # [box_num,scale_num,2]
            if len(scale_list.shape) == 1:
                scaled_w = scale_list.unsqueeze(1).unsqueeze(0).repeat([box_num, 1, 2]).type_as(ref_boxes)
                scale_h = scale_list.unsqueeze(1).unsqueeze(0).repeat([box_num, 1, 2]).type_as(ref_boxes)  # [box_num,scale_num,2]
            else:#dynamic mode
                scaled_w = scale_list.unsqueeze(-1).repeat([1, 1, 2]).type_as(ref_boxes)
                scale_h = scale_list.unsqueeze(-1).repeat([1, 1, 2]).type_as(ref_boxes)  # [box_num,scale_num,2]
            box_xx, box_yy = torch.mul(scaled_w, box_xx), torch.mul(scale_h, box_yy)

            tar_boxes_wh = hw_boxes[:, 2:] - hw_boxes[:, :2]  # box_num,2
            if self.shift and scale_list.shape[-1] > 1:
                tar_boxes_wh = tar_boxes_wh*(self.grid_size + self.shift_kernel_size - 1) /self.grid_size#
            tar_boxes_wh = tar_boxes_wh.unsqueeze(1).repeat([1, scale_list.shape[-1], 1])  # box_num,scale_num,2
            box_xx = box_xx * tar_boxes_wh[:, :, 0].unsqueeze(-1) / W  # box_num,scale_num,2
            box_yy = box_yy * tar_boxes_wh[:, :, 1].unsqueeze(-1) / H  # box_num,scale_num,2
            box_cx, box_cy = (center_boxes[:, 2] + center_boxes[:, 0]) / 2, (center_boxes[:, 3] + center_boxes[:, 1]) / 2
            box_cx = box_cx.unsqueeze(1).unsqueeze(-1).repeat([1, scale_list.shape[-1], 2])
            box_cy = box_cy.unsqueeze(1).unsqueeze(-1).repeat([1, scale_list.shape[-1], 2])
            box_xx, box_yy = box_xx + box_cx, box_yy + box_cy  # box_num,scale_num,2
            align_boxes = torch.stack([box_xx[:, :, 0], box_yy[:, :, 0], box_xx[:, :, 1], box_yy[:, :, 1]],dim=-1)  # box_num,scale_num,4
            return align_boxes

        batch_index = tar_boxes[:,:1]
        tar_boxes = tar_boxes[:,1:]
        ref_boxes = ref_boxes[:,1:]
        if self.normed_box:
            tar_boxes[:,::2] = tar_boxes[:,::2]*W
            tar_boxes[:,1::2] = tar_boxes[:,1::2]*H
            ref_boxes[:,::2] = ref_boxes[:,::2]*W
            ref_boxes[:,1::2] = ref_boxes[:,1::2]*H
        scaled_ref_boxes = to_algin_space(ref_boxes,tar_boxes,scale_list).view(-1,4)
        scaled_tar_boxes = to_algin_space(tar_boxes,tar_boxes,torch.tensor([1.0])).view(-1,4)
        ref_align_boxes = torch.cat([batch_index.repeat([1,self.scale_number]).view([-1,1]),scaled_ref_boxes],dim=-1)
        tar_align_boxes = torch.cat([batch_index.repeat([1,1]),scaled_tar_boxes],dim=-1)
        return ref_align_boxes,tar_align_boxes

