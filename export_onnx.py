import argparse
import torch
from bunch import Bunch
from ruamel import yaml
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from tester import Tester
from utils import losses
from utils.helpers import get_instance
from torch import nn
from einops import rearrange
import torchvision
import numpy as np
import cv2

def main(data_path, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')    
    model = models.FR_UNet(num_classes=1, num_channels=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_dataset = vessel_dataset(data_path, mode="test")
    test_dataloader = DataLoader(test_dataset)

    for idx,(x,y) in enumerate(test_dataloader):
        pred = model(x)
        pred = torch.clip(pred, 0, 1)
        pred = pred.detach().clone().numpy()
        # pred = pred > 0.5
        pred = rearrange(pred, 'n c h w -> n h w c')
        pred = pred[0,...] * 255.0
        pred = pred.astype(np.uint8)
        print(pred.shape)
        cv2.imwrite(f'out{idx}.png', pred)
        # img = Image.fromarray(pred)
        # img.save(f'out{idx}.png')

    x = torch.randn(1, 3, 512, 512)
    print(x.shape)
    torch.onnx.export(model, x, 'vesselseg.onnx', input_names=['input'], output_names=['output'], opset_version=7)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default="/home/lwt/data_pro/vessel/DRIVE", type=str,
                        help="the path of dataset")
    parser.add_argument("-wp", "--weight_path", default="pretrained_weights/DRIVE/checkpoint-epoch40.pth", type=str,
                        help='the path of wetght.pt')
    args = parser.parse_args()
    
    main(args.dataset_path, args.weight_path)