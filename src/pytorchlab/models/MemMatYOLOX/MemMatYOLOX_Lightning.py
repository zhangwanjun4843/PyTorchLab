import pytorch_lightning as pl

class MemMatYOLOX_Lit(pl.LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
