import sys

from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

from src.pytorchlab.datamodules.yoloVID.yoloVDataModule import YOLOVDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from src.pytorchlab.utils.defaults import argument_parser, load_config
from src.pytorchlab.utils.build_logger import build_logger
from src.pytorchlab.models.yoloVID.YOLOVID_Lightning import YOLOV_Lit
from pytorch_lightning.callbacks import TQDMProgressBar


def main():
    args = argument_parser().parse_args()
    configs = load_config(args.cfg)

    model = YOLOV_Lit.load_from_checkpoint("lightning_logs/version_55/checkpoints/epoch=299-step=185400.ckpt",
                                           cfgs=configs, strict=False)

    data = YOLOVDataModule(configs)
    logger = build_logger(args.logger, model, configs)
    seed_everything(96, workers=True)
    device_stats = DeviceStatsMonitor()
    # ckpt_callback = ModelCheckpoint(
    #     monitor='best_mAP50',
    #     save_top_k=1,
    #     mode='min',
    #     # filename='body_pixel-epoch={epoch}-val_body_pix={val/body_pix:.4f}',
    #     filename='best-{}'.format("map50"),
    #     auto_insert_metric_name=False
    # )
    # ckpt_callback5095 = ModelCheckpoint(
    #     monitor='best_mAP',
    #     save_top_k=1,
    #     mode='min',
    #     # filename='body_pixel-epoch={epoch}-val_body_pix={val/body_pix:.4f}',
    #     filename='best-{}'.format("map50:95"),
    #     auto_insert_metric_name=False
    # )
    #
    # last_callback = ModelCheckpoint(
    #     every_n_epochs=300,
    #     save_top_k=1,
    #     # filename='body_pixel-epoch={epoch}-val_body_pix={val/body_pix:.4f}',
    #     filename='last-{}'.format("yolov"),
    #     auto_insert_metric_name=False
    # )


    trainer = Trainer(
        devices=1,
        max_epochs=100,
        check_val_every_n_epoch=10,
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=logger,
        # precision=16,
        # amp_backend="apex",
        # amp_level=01,
        # auto_lr_find=True,
        # benchmark=False,
        callbacks=[device_stats, MyProgressBar()],
        # default_root_dir="lightning_logs",
        # detect_anomaly=True,
        # limit_train_batches=3,
        # limit_val_batches=2,
        # reload_dataloaders_every_n_epochs=10,
    )

    trainer.fit(model, datamodule=data)


class MyProgressBar(TQDMProgressBar):

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


if __name__ == "__main__":
    main()
