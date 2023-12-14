import pytorch_lightning as pl
from src.pytorchlab.datamodules.yoloVID.VIDDataset import VIDDataset, get_trans_loader, get_vid_loader


class YOLOVDataModule(pl.LightningDataModule):
    def __init__(self, cfgs):
        super().__init__()
        self.dataset_val = None
        self.VID_config = cfgs['dataset']['VID']
        self.trainsforms_config = cfgs['transform']

        self.train_file_path = self.VID_config['train_file_path']
        self.val_file_path = self.VID_config['val_file_path']
        self.img_train_size = self.VID_config['img_train_size']
        self.img_val_size = self.VID_config['img_test_size']
        self.lframe = self.VID_config['lframe']
        self.gframe = self.VID_config['gframe']
        self.hsv_prob = self.trainsforms_config['hsv_prob']
        self.flip_prob = self.trainsforms_config['flip_prob']
        self.data_dir = self.VID_config['data_path']
        self.degrees = self.trainsforms_config['degrees']
        self.translate = self.trainsforms_config['translate']
        self.mosaic_scale = self.trainsforms_config['mosaic_scale']
        self.mixup_scale = self.trainsforms_config['mixup_scale']
        self.shear = self.trainsforms_config['shear']
        self.perspective = self.trainsforms_config['perspective']
        self.enable_mixup = self.trainsforms_config['enable_mixup']
        self.mosaic_prob = self.trainsforms_config['mosaic_prob']
        self.mixup_prob = self.trainsforms_config['mixup_prob']
        self.train_batch_size = self.VID_config['train_batch_size']
        self.val_batch_size = self.VID_config['val_batch_size']

    def train_dataloader(self):
        from src.pytorchlab.datamodules.yolox.augmentation.data_augments import TrainTransform
        from src.pytorchlab.datamodules.yoloVID.mosaicDetectionVID import MosaicDetection_VID
        dataset = VIDDataset(file_path=self.train_file_path,
                             img_size=(self.img_train_size[0], self.img_train_size[1]),
                             preproc=TrainTransform(
                                 max_labels=50,
                                 flip_prob=self.flip_prob,
                                 hsv_prob=self.hsv_prob),
                             lframe=self.lframe,  # batch_size,
                             gframe=self.gframe,
                             dataset_path=self.data_dir)
        dataset = MosaicDetection_VID(
            dataset,
            mosaic=True,
            img_size=self.img_train_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            dataset_path=self.data_dir
        )

        dataloader = get_trans_loader(batch_size=self.train_batch_size, data_num_workers=0, dataset=dataset)
        return dataloader

    def val_dataloader(self):
        from src.pytorchlab.utils.yoloVID import Vid_Val_Transform
        self.dataset_val = VIDDataset(file_path=self.val_file_path,
                                      img_size=(self.img_val_size[0], self.img_val_size[1]),
                                      preproc=Vid_Val_Transform(),
                                      lframe=self.lframe,
                                      gframe=self.gframe, val=True, dataset_path=self.data_dir)
        val_loader = get_vid_loader(batch_size=self.val_batch_size, data_num_workers=0, dataset=self.dataset_val, )
        return val_loader
