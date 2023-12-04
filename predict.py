from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from src.pytorchlab.datamodules.yolox.yolox_dataset import COCODataModule
from src.pytorchlab.models.yolox.pl_yolox import LitYOLOX
import torch
from src.pytorchlab.utils.defaults import argument_parser, load_config
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.pytorchlab.datamodules.yolox.cocoDataset import COCODataset
from src.pytorchlab.datamodules.yolox.mosaic_detection import MosaicDetection
from src.pytorchlab.datamodules.yolox.augmentation.data_augments import TrainTransform, ValTransform
from torch.utils.data.sampler import BatchSampler, RandomSampler

    
class COCOEvalModule(pl.LightningDataModule):
    def __init__(self, cfgs):
        super().__init__()
        self.dataset_train = None
        self.dataset_val = None
        self.cd = cfgs['dataset']
        self.ct = cfgs['transform']
        # dataloader parameters
        self.data_dir = self.cd['dir']
        self.train_dir = self.cd['train']
        self.val_dir = self.cd['val']
        self.img_size_train = tuple(self.cd['train_size'])
        self.img_size_val = tuple(self.cd['val_size'])
        self.train_batch_size = self.cd['train_batch_size']
        self.val_batch_size = self.cd['val_batch_size']
        # transform parameters
        self.hsv_prob = self.ct['hsv_prob']
        self.flip_prob = self.ct['flip_prob']
        # mosaic
        self.mosaic_prob = self.ct['mosaic_prob']
        self.mosaic_scale = self.ct['mosaic_scale']
        self.degrees = self.ct['degrees']
        self.translate = self.ct['translate']
        self.shear = self.ct['shear']
        self.perspective = self.ct['perspective']
        # copypaste
        self.copypaste_prob = self.ct['copypaste_prob']
        self.copypaste_scale = self.ct['copypaste_scale']
        # cutpaste
        self.cutpaste_prob = self.ct['cutpaste_prob']
    
    def test_dataloader(self):
        self.dataset_val = COCODataset(
            self.data_dir,
            name=self.val_dir,
            img_size=self.img_size_val,
            preprocess=ValTransform(legacy=False),
            cache=True,
        )
        sampler = torch.utils.data.SequentialSampler(self.dataset_val)
        val_loader = DataLoader(self.dataset_val, batch_size=self.val_batch_size, sampler=sampler,
                                num_workers=0, pin_memory=True, shuffle=False)
        return val_loader
    
# 清空viz文件夹
import os
import shutil
viz_dir = r"viz"
if os.path.exists(viz_dir):
    shutil.rmtree(viz_dir)
os.mkdir(viz_dir)

configs_path = r"example\yolox\yolox_nano.yaml"
configs = load_config(configs_path)

model = LitYOLOX.load_from_checkpoint(r"lightning_logs\version_28\checkpoints\epoch=299-step=185400.ckpt",cfgs=configs)


dataloader = COCOEvalModule(configs).test_dataloader()
trainer = pl.Trainer(devices=1,precision='16-mixed')
predictions= trainer.predict(model,dataloader)
