from typing import Optional
import pyrootutils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import sys

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
# from pytorch_lightning.trainingtype import DDPPlugin
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm
from prima.configs import dataset_config
from prima.datasets import DataModule
from prima.models.prima import PRIMA
from prima.utils.pylogger import get_pylogger
from prima.utils.misc import task_wrapper, log_hyperparameters
import signal

signal.signal(signal.SIGUSR1, signal.SIG_DFL)


class MyTQDMProgressBar(TQDMProgressBar):

    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()
        
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ncols = 150
        bar.dynamic_ncols=False
        return bar

    def init_validation_tqdm(self):
        bar = tqdm(
            desc=self.validation_description,
            position=0,
            disable=self.is_disabled,
            leave=True,
            # dynamic_ncols=True,
            file=sys.stdout,
            dynamic_ncols= False,
            ncols = 150,
        )
        return bar


@hydra.main(version_base="1.2", config_path= "./prima/configs_hydra", config_name="train.yaml")

def main(cfg: DictConfig) -> Optional[float]:
    datamodule = DataModule(cfg)
    model = PRIMA(cfg)

    # Setup Tensorboard logger
    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='',
                               default_hp_metric=False)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'),
        # every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        every_n_epochs=cfg.GENERAL.CHECKPOINT_EPOCHS,
        save_last=True,
        # Monitor a metric so `save_top_k` keeps the best checkpoint instead of the last one.
    # We monitor the validation loss logged as 'val/loss' (lower is better).
        monitor='val/loss',
        mode='min',
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        filename="best-{epoch:03d}-{val_loss:.4f}",  # 清晰标记最好的checkpoint
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback,
        lr_monitor,
        # rich_callback
        MyTQDMProgressBar()
    ]

    log = get_pylogger(__name__)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher', None) is not None) else None),
        sync_batchnorm=True,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Train the model
    # Determine checkpoint path
    ckpt_path = None
    last_v1_ckpt = os.path.join(cfg.paths.output_dir, 'checkpoints', 'last-v1.ckpt')
    last_ckpt = os.path.join(cfg.paths.output_dir, 'checkpoints', 'last.ckpt')
    
    if os.path.exists(last_v1_ckpt):
        ckpt_path = last_v1_ckpt
        log.info(f"Resuming from checkpoint: {ckpt_path}")
    elif os.path.exists(last_ckpt):
        ckpt_path = last_ckpt
        log.info(f"Resuming from checkpoint: {ckpt_path}")
    else:
        log.info("No checkpoint found, starting from scratch")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info("Fitting done")


if __name__ == "__main__":
    import torch
    import gc
    
    gc.collect()
    
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated()/1024**2:.2f} MiB allocated, "
            f"{torch.cuda.memory_reserved()/1024**2:.2f} MiB reserved")
    main()
