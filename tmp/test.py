import argparse
import os
from math import log10
from datetime import datetime
import time
import random
from PIL import Image

import torch
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch import nn

#import pytorch_ssim
from loss import GeneratorLoss
from model import ResNet50
import utils
from data_loader import *

import torch_xla                      # turing
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
import turing
from turing.pytorch import ArnoldClusterResolver


parser = argparse.ArgumentParser(description='Train Super Resolution Models')

parser.add_argument('--data_dir', default = '/home/tiger/dataset/test', type = str, help = ' test image path')
parser.add_argument('--checkpoint', default = './checkpoint/model_best.pth.tar', type = str, help = 'model save path')
parser.add_argument('--save_path', default = './benckmark')
parser.add_argument('--resize_size', default=(512,512), type=int, help='training images size')
parser.add_argument('--batch_size', default = 1, type = int, help = 'training batch size')
parser.add_argument('--nums_gpu', default = 1, type = int, help = 'use gpu nums')
parser.add_argument('--print_freq', default = 10, type = int, help = 'the frequency with which messages are printed')
parser.add_argument('--num_workers', default = 4, type = int, help = ' data loader num workers')
parser.add_argument('--threshold', default = 0.5, type = int)
parser.add_argument('--delete_origin_img', default = False, type = bool)

args = parser.parse_args()


def main(index):
    global args
    time.sleep(xm.get_ordinal())
    device = xm.xla_device()
    print("Process: {}, xla device: {}, real device: {}, rank: {}, world size: {}".format(index, device, 
          xm.xla_real_devices([str(device)])[0], xm.get_ordinal(), xm.xrt_world_size()))
    args.rank = xm.xla_real_devices([str(device)])[0]

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
 
    transform = transforms.Compose([transforms.Resize(args.resize_size),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])

    test_data, _ = split_dataset(args.data_dir, shuffle=False, ratio=1)

    test_set = TestDataset(test_data, transforms = transform)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    resnet50 = ResNet50()
    print('# model parameters:', sum(param.numel() for param in resnet50.parameters()))

    resnet50.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    resnet50.to(device)


    resnet50.eval()
    with torch.no_grad():
        for i, (imgs, img_path) in enumerate(test_loader):
            imgs = imgs.to(device)
            pred = resnet50(imgs)
            print('step: {}, image name : {}'.format(i, img_path))
            # pred = pred.detach().cpu().squeeze().numpy()
            # imgs = imgs.detach().cpu().squeeze().numpy().transpose(1,2,0)
            # imgs = (imgs + 1.0) * 127.5
            # imgs = Image.fromarray(imgs.astype(np.uint8))
            label = 1 if pred > args.threshold else 0
            path = os.path.join(args.save_path, str(label))
            if not os.path.exists(path):
                os.mkdir(path)
            # imgs.save(os.path.join(path, name[0]))
            name = os.path.basename(img_path[0])
            save_path = os.path.join(path, name)
            shutil.copy(img_path[0], save_path)
            if args.delete_origin_img:
                os.remove(img_path[0])
    print('Done.')


if __name__ == '__main__':
    print('turing version:', turing.__version__)   # turing
    token = '09efffbfb7ce22febd0d3c1db998e44c6490252b'
    cluster = ArnoldClusterResolver(token=token, num_gpus=args.nums_gpu)
    print('{} gpu to use'.format(args.nums_gpu))
    torch.set_default_tensor_type('torch.FloatTensor')
    main(0)
    
