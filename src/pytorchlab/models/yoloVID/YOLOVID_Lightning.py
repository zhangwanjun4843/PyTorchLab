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
from .post_trans import MSA_yolov_online


def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


class YOLOV_Lit(pl.LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.best_ap50_95 = 0
        self.best_ap50 = 0
        self.evaluator: VIDEvaluator = None
        self.backbone_config = cfgs['backbone']
        self.neck_config = cfgs['neck']
        self.head_config = cfgs['head']
        self.VIDdata_config = cfgs['dataset']['VID']
        self.optimizer_config = cfgs['optimizer']

        self.img_size_val = cfgs['dataset']['VID']['img_train_size']
        self.img_size_train = cfgs['dataset']['VID']['img_train_size']
        self.img_size_test = cfgs['dataset']['VID']['img_test_size']
        self.depth = self.backbone_config['depth']
        self.width = self.backbone_config['width']
        self.num_classes = self.VIDdata_config['num_classes']
        self.drop_rate = self.backbone_config['drop_rate']
        self.learning_rate = self.optimizer_config['learning_rate']
        self.momentum = self.optimizer_config['momentum']
        self.warmup = self.optimizer_config['warmup']
        self.ema = self.optimizer_config['ema']
        self.ema_model = None
        # 手动backward
        self.automatic_optimization = False

        in_channels = [256, 512, 1024]

        self.backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
        for layer in self.backbone.parameters():
            layer.requires_grad = False  # fix the backbone
        self.trans =MSA_yolov_online(dim=int(256 * self.width), out_dim=4 * int(256 * self.width), num_heads=4, attn_drop=self.drop_rate)
        self.linear_pred = nn.Linear(int(4 * int(256 * self.width)), self.num_classes+1)  # Mlp(in_features=512,hidden_features=self.num_classes+1)
        self.head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, heads=4, drop=self.drop_rate,trans=self.trans, linear_pred=self.linear_pred)
        # for layer in self.head.stems.parameters():
        #     layer.requires_grad = False  # set stem fixed
        #
        # for layer in self.head.reg_convs.parameters():
        #     layer.requires_grad = False
        # for layer in self.head.reg_preds.parameters():
        #     layer.requires_grad = False
        # for layer in self.head.obj_preds.parameters():
        #     layer.requires_grad = False
        # for layer in self.head.cls_convs.parameters():
        #     layer.requires_grad = False
        self.model = YOLOV(self.backbone, self.head)
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
        self.log("best_mAP", self.best_ap50_95, prog_bar=True)
        self.log("best_mAP50", self.best_ap50, prog_bar=True)


    def on_validation_end(self) -> None:
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
        print(summary[2])
        self.best_ap50 = max(self.best_ap50, ap50)
        self.best_ap50_95 = max(self.best_ap50_95, ap50_95)

        print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(
            self.current_epoch, ap50_95, ap50))

    def validation_step(self, batch, batch_idx):
        # 获取batch的数量
        imgs, labels, info_imgs, label, path, _ = batch

    def on_validation_start(self) -> None:
        self.evaluator = VIDEvaluator(
            dataloader=self.trainer.datamodule.val_dataloader(),
            img_size=self.img_size_val,
            confthre=self.cfgs['confthre'],
            nmsthre=self.cfgs['nmsthre'],
            num_classes=self.num_classes
        )

    def training_step(self, batch, batch_idx):
        imgs, labels, info_imgs, label_list, path, _ = batch
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

    def on_train_epoch_end(self) -> None:
        print("Best mAP = {:.3f}, best mAP50 = {:.3f}".format(
            self.best_ap50_95, self.best_ap50))
