from src.pytorchlab.datamodules.yoloVID.yoloVDataModule import YOLOVDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import DeviceStatsMonitor
from src.pytorchlab.utils.defaults import argument_parser, load_config
from src.pytorchlab.utils.build_logger import build_logger
from src.pytorchlab.models.yoloVID.YOLOVID_Lightning import YOLOV_Lit

def main():
    args = argument_parser().parse_args()
    configs = load_config(args.cfg)
    model = YOLOV_Lit.load_from_checkpoint("lightning_logs/version_28/checkpoints/epoch=299-step=185400.ckpt",cfgs=configs)

    data= YOLOVDataModule(configs)
    logger = build_logger(args.logger, model, configs)
    seed_everything(96, workers=True)
    trainer = Trainer(
        devices=1,
        max_epochs=300,
        check_val_every_n_epoch=10,
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=logger,
        # precision=16,
        # amp_backend="apex",
        # amp_level=01,
        # auto_lr_find=True,
        # benchmark=False,
        # callbacks=[device_stats],
        # default_root_dir="lightning_logs",
        # detect_anomaly=True,
        # limit_train_batches=3,
        # limit_val_batches=2,
        # reload_dataloaders_every_n_epochs=10,
    )

    trainer.fit(model, datamodule=data)









if __name__ == "__main__":
    main()