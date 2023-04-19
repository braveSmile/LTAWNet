import os
import sys
import time
import math
from collections import namedtuple

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
import megengine.autodiff as autodiff
import megengine.optimizer as optim
import yaml
from tensorboardX import SummaryWriter

import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_ltaw import PL_LTAW

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LTAW')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    
    
    parser.add_argument(
        '--mixed_precision', type=bool, default=False, help='if adoptmixed_precision when training')
    parser.add_argument(
        '--n_total_epoch', type=int, default=600, help='max epoch')
    parser.add_argument(
        '--minibatch_per_epoch', type=int, default=500, help='mini epoch size')
    parser.add_argument(
        'log_dir', type=str,  default="./train_log", help='train_log config path')
    parser.add_argument(
        '--model_save_freq_epoch', type=int, default=1, help='model save frequency epoch')
    
    
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8):

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = F.abs(flow_preds[i] - flow_gt)
        flow_loss += i_weight * (F.expand_dims(valid, axis=1) * i_loss).mean()

    return flow_loss

def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # initial info
    rank, world_size = dist.get_rank(), dist.get_world_size()
    mge.dtr.enable()  # Dynamic tensor rematerialization for memory optimization
    
    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation
    
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_LTAW(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"LTAW LightningModule initialized!")
    model.load_state_dict(pretrained_dict["state_dict"], strict=True)
    optimizer.load_state_dict(pretrained_dict["optim_state_dict"])
    start_epoch_idx = resume_epoch_idx + 1
    start_iters = resume_iters

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"LTAW DataModule initialized!")
    
    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='train_log/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    
    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=5, mode='max',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)


    # auxiliary
    if world_size > 1:
        dist.bcast_list_(model.tensors())

    # counter
    cur_iters = start_iters
    for epoch_idx in range(start_epoch_idx, args.n_total_epoch + 1):
        # adjust learning rate
        epoch_total_train_loss = 0
        # Lightning Trainer
        trainer = pl.Trainer.from_argparse_args(
            args,
            plugins=DDPPlugin(find_unused_parameters=False,
                            num_nodes=args.num_nodes,
                            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
            gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
            callbacks=callbacks,
            logger=logger,
            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
            replace_sampler_ddp=False,  # use custom sampler
            reload_dataloaders_every_epoch=False,  # avoid repeated samples!
            weights_summary='full',
            profiler=profiler)
        loguru_logger.info(f"Trainer initialized!")
        loguru_logger.info(f"Start training!")
        trainer.fit(model, datamodule=data_module) 
        
        t1 = time.perf_counter()
        batch_idx = 0       
        
            
 


if __name__ == '__main__':
    main()
