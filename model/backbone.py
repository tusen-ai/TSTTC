import torch
from torch import nn
from .network_blocks import BaseConv, CSPLayer, DWConv

class TTCBase(nn.Module):
    def __init__(
            self,
            dep_mul=1,
            wid_mul=1,
            depthwise=False,
            act="relu",
            kszie = 7
    ):
        super().__init__()

        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 12)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = BaseConv(3,base_channels,ksize=kszie,act=act,stride=2)

        self.stage2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, kszie, 1, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 3, 2, 1,)
        self.stage3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 2, kszie, 1, act=act),
            Conv(base_channels * 2, base_channels * 2, kszie, 1, act=act)
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.stage2(x)
        outputs["stage2"] = x
        upsample_dst = torch.tensor(x.size())
        upsample_dst[-2:] = upsample_dst[-2:]*2
        x = self.upsample(x,output_size=upsample_dst)
        x = self.stage3(x)
        return x#{k: v for k, v in outputs.items() if k in self.out_features}


