import argparse
from bunch import Bunch
from loguru import logger
from ruamel import yaml
from torch.utils.data import DataLoader, ConcatDataset
import models
from dataset import vessel_dataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import os
from torch import nn

def setup():
    init_process_group("nccl")

def cleanup():
    """Distributed Data Parallel를 위한 종료 함수"""
    destroy_process_group()

def main(CFG, data_paths, batch_size, with_val=False):
    setup()
    seed_torch()

    train_dataset = []
    val_dataset = []
    for data_path in data_paths:
        if with_val:
            train_dataset.append(vessel_dataset(data_path, mode="training", split=0.9))
            val_dataset.append(vessel_dataset(
                data_path, mode="training", split=0.9, is_val=True))
        else:
            train_dataset.append(vessel_dataset(data_path, mode="training"))

    if with_val:
        val_dataset = ConcatDataset(val_dataset)
        val_sampler = DistributedSampler(val_dataset)
    train_dataset = ConcatDataset(train_dataset)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)


    if with_val:
        val_loader = DataLoader(
            val_dataset, batch_size, num_workers=4, pin_memory=True, sampler=val_sampler)
    train_loader = DataLoader(
        train_dataset, batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)
    

    logger.info('The patch number of train is %d' % len(train_dataset))
    model = models.FR_UNet(num_classes=1, num_channels=3)
    logger.info(f'\n{model}\n')
    loss = nn.BCEWithLogitsLoss()
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader if with_val else None
    )

    trainer.train()

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_paths', default="{'datasets/DRIVE','datasets/STARE','datasets/CHASEDB1'}", type=str,
                        help='the path of dataset')
    parser.add_argument('-bs', '--batch_size', default=512,
                        help='batch_size for trianing and validation')
    parser.add_argument("--val", help="split training data for validation",
                        required=False, default=False, action="store_true")
    args = parser.parse_args()

    data_paths = eval(args.dataset_paths)

    
    with open("config.yaml", encoding="utf-8") as file:
        yaml = yaml.YAML(typ='safe', pure=True)
        CFG = Bunch(yaml.load(file))

    main(CFG, data_paths, args.batch_size, args.val)
