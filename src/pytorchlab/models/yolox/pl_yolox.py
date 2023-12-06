import time
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
# Model
from src.pytorchlab.models.yolox.OneStage import OneStageD
from src.pytorchlab.models.yolox.darknet_csp import CSPDarkNet
from src.pytorchlab.models.yolox.pafpn import PAFPN
from src.pytorchlab.models.yolox.decoupled_head import DecoupledHead
from src.pytorchlab.models.yolox.yolox_loss import YOLOXLoss
from src.pytorchlab.models.yolox.yolox_decoder import YOLOXDecoder

from src.pytorchlab.models.yolox.evaluators.coco import COCOEvaluator, convert_to_coco_format
# Data
from src.pytorchlab.utils.ema import ModelEMA
from torch.optim import SGD, AdamW, Adam

from src.pytorchlab.models.yolox.lr_scheduler import CosineWarmupScheduler
from src.pytorchlab.utils.flops import model_summary


class LitYOLOX(LightningModule):

    def __init__(self, cfgs):
        super().__init__()
        # {'yaobian': 1, 'shaochuanduanhu': 2, 'qikong': 3, 'jiazha': 4, 'weihantou': 5, 'weironghe': 6, 'liewen': 7}
        self.classids = ["yaobian", "shaochuanduanhu", "qikong",
                         "jiazha", "weihantou", "weironghe", "liewen"]
        self.val_step_outputs = []
        self.cb = cfgs['backbone']
        self.cn = cfgs['neck']
        self.ch = cfgs['head']
        self.cd = cfgs['dataset']
        self.co = cfgs['optimizer']
        # backbone parameters
        b_depth = self.cb['depth']
        b_norm = self.cb['normalization']
        b_act = self.cb['activation']
        b_channels = self.cb['input_channels']
        out_features = self.cb['output_features']
        # neck parameters
        n_depth = self.cn['depth']
        n_channels = self.cn['input_channels']
        n_norm = self.cn['normalization']
        n_act = self.cn['activation']
        # head parameters
        n_anchors = 1
        strides = [8, 16, 32]
        # loss parameters
        self.use_l1 = False
        # evaluate parameters
        self.nms_threshold = 0.65
        self.confidence_threshold = 0.5
        # data
        self.num_classes = self.cd['num_classes']
        self.train_batch_size = self.cd['train_batch_size']
        self.val_batch_size = self.cd['val_batch_size']
        self.img_size_train = tuple(self.cd['train_size'])
        self.img_size_val = tuple(self.cd['val_size'])
        # Training
        self.ema = self.co['ema']
        self.warmup = self.co['warmup']
        self.iter_times = []
        # Network
        self.backbone = CSPDarkNet(
            b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = PAFPN(n_depth, n_channels, n_norm, n_act)
        self.head = DecoupledHead(
            self.num_classes, n_anchors, n_channels, n_norm, n_act)
        self.loss = YOLOXLoss(self.num_classes, strides)
        self.decoder = YOLOXDecoder(self.num_classes, strides)
        self.model = OneStageD(self.backbone, self.neck, self.head)
        self.ema_model = None

        self.head.initialize_biases(1e-2)
        self.model.apply(initializer)
        self.automatic_optimization = False

        self.ap50_95 = 0
        self.ap50 = 0

    def on_train_start(self) -> None:
        if self.ema is True:
            self.ema_model = ModelEMA(self.model, 0.9998)
        model_summary(self.model, self.img_size_train)

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        output = self.model(imgs)
        loss, loss_iou, loss_obj, loss_cls, loss_l1, proportion = self.loss(
            output, labels)
        self.log("loss/loss", loss, prog_bar=True)
        self.log("loss/iou", loss_iou, prog_bar=False)
        self.log("loss/obj", loss_obj, prog_bar=False)
        self.log("loss/cls", loss_cls, prog_bar=False)
        self.log("loss/l1", loss_l1, prog_bar=False)
        self.log("loss/proportion", proportion, prog_bar=False)
        self.log(
            "lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        # Backward
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        if self.ema is True:
            self.ema_model.update(self.model)
        self.lr_schedulers().step()

    def forward(self, batch):
        imgs, labels, img_hw, image_id, img_name = batch
        path = r"H:\001-datasets\yolox\datasets\coco\test2017"
        if self.ema_model is not None:
            model = self.ema_model.ema
        else:
            model = self.model
        output = model(imgs)
        detections = self.decoder(
            output, self.confidence_threshold, self.nms_threshold)

        self.visualize_dets(imgs, detections, self.classids, path, img_name)
        return detections

    def visualize_dets(self, imgs, detections, class_ids, path, img_name):
        import cv2
        import numpy as np
        import os
        for i in range(len(detections)):
            raw_img = cv2.imread(os.path.join(path, img_name[i]))
            img = np.zeros_like(raw_img)
            img[:, :, :] = raw_img[:, :, :]
            height, width, _ = raw_img.shape
            if detections[i] is None:
                continue
            for det in detections[i]:
                x1, y1, x2, y2, conf, _, cls = det
                x1, y1, x2, y2 = int(x1 / self.img_size_train[0] * width), int(
                    y1 / self.img_size_train[0] * height), int(
                    x2 / self.img_size_train[0] * width), int(y2 / self.img_size_train[0] * height)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, class_ids[int(cls)], (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join("viz", img_name[i]), img)

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        if self.ema_model is not None:
            model = self.ema_model.ema
        else:
            model = self.model
        start_time = time.time()
        output = model(imgs)
        detections = self.decoder(
            output, self.confidence_threshold, self.nms_threshold)
        self.iter_times.append(time.time() - start_time)
        detections = convert_to_coco_format(detections, image_id, img_hw, self.img_size_val,
                                            self.trainer.datamodule.dataset_val.class_ids)
        self.val_step_outputs.append(detections)
        return detections

    def on_validation_epoch_end(self):
        detect_list = []
        for i in range(len(self.val_step_outputs)):
            detect_list += self.val_step_outputs[i]
        ap50_95, ap50, summary = COCOEvaluator(
            detect_list, self.trainer.datamodule.dataset_val)
        print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(
            self.current_epoch, ap50_95, ap50))
        print(summary)
        self.val_step_outputs = []
        self.log("val/mAP", ap50_95, prog_bar=False)
        self.log("val/mAP50", ap50, prog_bar=False)
        if ap50_95 > self.ap50_95:
            self.ap50_95 = ap50_95
        if ap50 > self.ap50:
            self.ap50 = ap50

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(), lr=self.co["learning_rate"], momentum=self.co["momentum"])
        total_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup * total_steps, max_iters=total_steps
        )
        return [optimizer], [lr_scheduler]

    def on_train_end(self) -> None:
        average_ifer_time = torch.tensor(
            self.iter_times, dtype=torch.float32).mean()
        print("The average iference time is ", average_ifer_time, " ms")
        print("Best mAP = {:.3f}, best mAP50 = {:.3f}".format(
            self.ap50_95, self.ap50))


def initializer(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
