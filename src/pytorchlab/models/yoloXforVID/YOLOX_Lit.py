import time

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from pytorchlab.models.yolox.yolox_decoder import YOLOXDecoder
# Data
from src.pytorchlab.utils.ema import ModelEMA
from torch.optim import SGD, AdamW, Adam
from src.pytorchlab.models.yolox.yolox_loss import YOLOXLoss
from src.pytorchlab.models.yolox.lr_scheduler import CosineWarmupScheduler
from src.pytorchlab.utils.flops import model_summary
from src.pytorchlab.models.yoloVID.yolo_pafpn import YOLOPAFPN
from src.pytorchlab.models.yolox.evaluators.coco import COCOEvaluator, convert_to_coco_format


class YOLOXFORVID(LightningModule):
    def __init__(self, cfgs):
        super().__init__()

        self.val_step_outputs = None
        self.data_list = None
        self.n_samples = None
        self.nms_time = None
        self.inference_time = None
        self.cb = cfgs['backbone']
        self.cn = cfgs['neck']
        self.ch = cfgs['head']
        self.cd = cfgs['dataset']['DET']
        self.co = cfgs['optimizer']

        self.b_width = self.cb['width']
        self.b_depth = self.cb['depth']
        self.b_norm = self.cb['normalization']
        self.b_act = self.cb['activation']
        self.b_channels = self.cb['input_channels']
        self.out_features = self.cb['output_features']
        self.n_depth = self.cn['depth']
        self.n_channels = self.cn['input_channels']
        self.n_norm = self.cn['normalization']
        self.n_act = self.cn['activation']

        self.n_anchors = 1
        self.strides = [8, 16, 32]

        self.use_l1 = False

        self.nms_threshold = cfgs['nmsthre']
        self.confidence_threshold = cfgs['confthre']

        self.num_classes = self.cd['num_classes']
        self.train_batch_size = self.cd['train_batch_size']
        self.val_batch_size = self.cd['val_batch_size']
        self.img_size_train = tuple(self.cd['train_size'])
        self.img_size_val = tuple(self.cd['val_size'])

        self.learning_rate = self.co['learning_rate']
        self.momentum = self.co['momentum']
        self.ema = self.co['ema']
        self.ema_model = None
        self.warmup = self.co['warmup']

        self.in_channels = [256, 512, 1024]
        self.backbone = YOLOPAFPN(
            depth=self.b_depth,
            width=self.b_width,
            in_channels=self.in_channels,
            act=self.b_act,
        )
        from src.pytorchlab.models.yolox.decoupled_head import DecoupledHead
        self.head = DecoupledHead(
            self.num_classes, 1, [int(self.n_depth * i) for i in self.in_channels], self.n_norm, self.n_act)
        from src.pytorchlab.models.yolox.OneStage import OneStageD
        self.model = OneStageD(self.backbone, None, self.head)
        self.loss = YOLOXLoss(self.num_classes, self.strides)
        self.decoder = YOLOXDecoder(self.num_classes, self.strides)

        self.head.initialize_biases(1e-2)
        self.model.apply(initializer)
        self.automatic_optimization = False
        self.iter_times = []
        self.val_step_outputs  = []
        self.ap50_95 = 0
        self.ap50 = 0

    # def on_validation_start(self) -> None:
    #     self.evaluator = COCOEvaluator(
    #         dataloader=self.trainer.datamodule.val_dataloader(),
    #         img_size=self.img_size_val,
    #         confthre=self.confidence_threshold,
    #         nmsthre=self.nms_threshold,
    #         num_classes=self.num_classes
    #     )
    #     self.inference_time = 0
    #     self.nms_time = 0
    #     self.n_samples = max(len(self.evaluator.dataloader) - 1, 1)
    #
    # def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
    #     global infer_end
    #     imgs, labels, img_hw, ids, _ = batch
    #     # img, target, img_hw, np.array([id_]), img_name
    #     is_time_record = batch_idx < len(self.evaluator.dataloader) - 1
    #     if is_time_record:
    #         start = time.time()
    #     if self.ema_model is not None:
    #         model = self.ema_model.ema
    #     else:
    #         model = self.model
    #     self.data_list = []
    #     output = model(imgs)
    #     if is_time_record:
    #         infer_end = time_synchronized()
    #         self.inference_time += infer_end - start
    #
    #     output = postprocess(
    #         output, num_classes=self.num_classes,
    #         conf_thre=self.confidence_threshold,
    #         nms_thre=self.nms_threshold
    #     )
    #     if is_time_record:
    #         nms_end = time_synchronized()
    #         self.nms_time += nms_end - infer_end
    #     print(output)
    #     self.data_list.extend(self.evaluator.convert_to_coco_format(output, img_hw, ids))
    #
    # def on_validation_epoch_end(self) -> None:
    #     statistics = torch.tensor([self.inference_time, self.nms_time, self.n_samples], dtype=torch.float32,
    #                               device='cuda:0')
    #     eval_results = self.evaluator.evaluate_prediction(self.data_list, statistics)
    #     print(eval_results)

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

    def on_train_end(self) -> None:
        average_ifer_time = torch.tensor(
            self.iter_times, dtype=torch.float32).mean()
        print("The average iference time is ", average_ifer_time, " ms")
        print("Best mAP = {:.3f}, best mAP50 = {:.3f}".format(
            self.ap50_95, self.ap50))

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
        print("detect_list",detect_list)
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
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        total_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup * total_steps, max_iters=total_steps
        )
        return [optimizer], [lr_scheduler]


def initializer(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
