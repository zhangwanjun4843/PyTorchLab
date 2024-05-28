import pytorch_lightning as pl
from .CSPBackbone import CSPBackbone
from src.pytorchlab.models.yoloVID.yolov_msa_online import YOLOXHead
from src.pytorchlab.models.yoloVID.post_trans import MSA_yolov_online
import torch.nn as nn

class MemMatYOLOX_Lit(pl.LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.backbone_config = cfgs['backbone']

        self.depth = self.backbone_config['depth']
        self.width = self.backbone_config['width']



        self.MemBlock = []
        self.backbone = CSPBackbone(self.depth, self.width)
        self.trans = MSA_yolov_online(dim=int(256 * self.width), out_dim=4 * int(256 * self.width), num_heads=4,
                                      attn_drop=self.drop_rate)
        self.linear_pred = nn.Linear(int(4 * int(256 * self.width)),
                                     self.num_classes + 1)  # Mlp(in_features=512,hidden_features=self.num_classes+1)
        self.head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, heads=4, drop=self.drop_rate,
                              trans=self.trans, linear_pred=self.linear_pred)

