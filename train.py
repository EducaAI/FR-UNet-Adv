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

def main(CFG, data_paths, batch_size, with_val=False):
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
        val_loader = DataLoader(
            val_dataset, batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
        
    train_dataset = ConcatDataset(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    logger.info('The patch number of train is %d' % len(train_dataset))
    model = get_instance(models, 'model', CFG)
    logger.info(f'\n{model}\n')
    loss = get_instance(losses, 'loss', CFG)
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader if with_val else None
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_paths', default="{'datasets/DRIVE'}", type=str,
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
