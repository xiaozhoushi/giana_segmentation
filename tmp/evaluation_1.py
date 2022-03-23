import argparse
import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms

from data.GianaDataset import TestDataset, ValDataset
import utils.utils as utils
from model.segformer import Segformer
from utils.patch import ImageSegMerge, ImagePadToSize



def parse_args():
    parser = argparse.ArgumentParser(description='test Super Resolution Models')

    # data
    parser.add_argument('--data_path', default = '/mnt/bd/aurora-mtrc-data/datas/GIANA_challenge/segmentation/val', type = str, help = ' val image path')
    parser.add_argument('--crop_size', default=(512,512), type=int, help='val images size')
    #test 
    parser.add_argument('--batch_size', default = 1, type = int, help = 'val batch size')
    parser.add_argument('--num_workers', default = 4, type = int, help = ' data loader num workers')
    parser.add_argument('--threshold', default = 0.5, type = int)
    parser.add_argument('--device', default = 'cuda:0', type = str, help = 'device')
    parser.add_argument('--result_save_path', default = './result', type = str)
    parser.add_argument('--model_path', default = './checkpoint/20210715_213009/model_best.pth.tar', type = str)
    #parser.add_argument('--model_path', default = './checkpoint/20210716_122812/checkpoint.pth.tar', type = str)
    parser.add_argument('--if_crop', default = True, type = bool)
    # model
    parser.add_argument('--num_classes', default =2, type = int, help = 'number of classes')
    

    args = parser.parse_args()
    return args


def main():
    args = parse_args() 
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.result_save_path):
        os.mkdir(args.result_save_path)
    

    test_set = TestDataset(args)
    
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, \
        num_workers = args.num_workers)

    model = Segformer()
    print('# model parameters:', sum(param.numel() for param in model.parameters()))

    checkpoint = torch.load(args.model_path)['state_dict']
    print('epoch:', torch.load(args.model_path)['epoch'])
    print('best eval:', torch.load(args.model_path)['best_eval'])
    model.load_state_dict(checkpoint)
    
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        for i, (imgs, labels, _) in enumerate(test_loader):
            metrics = utils.IOUMetric(args.num_classes, args.threshold)
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            #path_seg = ImageSegMerge(args.crop_size, model, 1)
            #pred = path_seg(imgs)
            img_forward = ImagePadToSize(args.crop_size, model)
            pred = img_forward(imgs)
            metrics.add_batch(pred, labels)
            acc, acc_cls, iu, mean_iu, fwavacc = metrics.evaluate()
            print(mean_iu)
        print('test acc: {}, iu: {}, mean_iu: {}'.format(acc, iu, mean_iu))
    print('Done.')


if __name__ == '__main__':
    main() 
