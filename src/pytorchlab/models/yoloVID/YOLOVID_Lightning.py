import pytorch_lightning as pl
from src.pytorchlab.models.yoloVID.yolo_pafpn import YOLOPAFPN
from src.pytorchlab.models.yoloVID.yolov_msa_online import YOLOXHead
from src.pytorchlab.models.yoloVID.YOLOV import YOLOV
import torch.nn as nn
from torch.optim import SGD
from src.pytorchlab.models.yolox.lr_scheduler import CosineWarmupScheduler
from src.pytorchlab.utils.ema import ModelEMA
from src.pytorchlab.utils.flops import model_summary
from src.pytorchlab.models.yoloVID.vid_evaluator_v2 import VIDEvaluator


def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


class YOLOV_Lit(pl.LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.best_ap50_95 = 0
        self.best_ap50 = 0
        self.evaluator: VIDEvaluator = None
        self.backbone_config = cfgs['backbone']
        self.neck_config = cfgs['neck']
        self.head_config = cfgs['head']
        self.VIDdata_config = cfgs['dataset']['VID']
        self.optimizer_config = cfgs['optimizer']

        self.depth = self.backbone_config['depth']
        self.width = self.backbone_config['width']
        self.num_classes = self.VIDdata_config['num_classes']
        self.drop_rate = self.backbone_config['drop_rate']
        self.learning_rate = self.optimizer_config['learning_rate']
        self.momentum = self.optimizer_config['momentum']
        self.warmup = self.optimizer_config['warmup']
        self.ema = self.optimizer_config['ema']
        self.ema_model = None

        in_channels = [256, 512, 1024]

        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
        for layer in backbone.parameters():
            layer.requires_grad = False  # fix the backbone

        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, heads=4, drop=self.drop_rate)
        for layer in head.stems.parameters():
            layer.requires_grad = False  # set stem fixed

        for layer in head.reg_convs.parameters():
            layer.requires_grad = False
        for layer in head.reg_preds.parameters():
            layer.requires_grad = False
        for layer in head.obj_preds.parameters():
            layer.requires_grad = False
        for layer in head.cls_convs.parameters():
            layer.requires_grad = False
        self.model = YOLOV(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

    def on_train_start(self) -> None:
        if self.ema is True:
            self.ema_model = ModelEMA(self.model, 0.9998)
        model_summary(self.model, self.img_size_train)

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        total_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup * total_steps, max_iters=total_steps
        )
        return [optimizer], [lr_scheduler]

    def on_validation_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        if self.ema_model is not None:
            model = self.ema_model.ema
        else:
            model = self.model

        summary = self.evaluator.evaluate(
            model=model,
            half=False,
        )
        ap50_95 = summary[0]
        ap50 = summary[1]
        self.log("val/mAP", ap50_95, prog_bar=False)
        self.log("val/mAP50", ap50, prog_bar=False)
        self.best_ap50 = max(self.best_ap50, ap50)
        self.best_ap50_95 = max(self.best_ap50_95, ap50_95)
        print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(
            self.current_epoch, ap50_95, ap50))


    def on_validation_start(self) -> None:
        self.evaluator = VIDEvaluator(
            dataloader=self.trainer.datamodule.val_dataloader(),
            img_size=self.img_size_val,
            confthre=self.cfgs['confthre'],
            num_classes=self.num_classes
        )

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        output = self.model(imgs, labels)
        loss = output["total_loss"]
        self.log("total_loss", loss, prog_bar=True)
        # Backward
        self.log(
            "lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        if self.ema is True:
            self.ema_model.update(self.model)
        self.lr_schedulers().step()
        print("Batch {:d}, loss = {:.3f}".format(self.current_epoch, loss))
