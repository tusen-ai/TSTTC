import torch.nn as nn
import torch
import sys

sys.path.append("..")

class TTCNet(nn.Module):

    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, input_box_list, dictAnnos=None, ttc_gts=None,):

        backbone_outs = self.backbone(x)
        ref_boxes, tar_boxes = input_box_list[::2, :], input_box_list[1::2, :]
        if self.training:
            assert ttc_gts is not None
            ttc_loss = self.head(backbone_outs, tar_boxes, ref_boxes, ttc_gts,dictAnnos=dictAnnos)
            outputs = {
                "total_loss": ttc_loss,
                "ttc_loss": ttc_loss
            }
        else:
            outputs = self.head(backbone_outs, tar_boxes,
                                ref_boxes, ttc_gts,dictAnnos=dictAnnos)

        return outputs

