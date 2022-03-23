import argparse
import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms

from data.GianaDataset import TestDataset
import utils.utils as utils
from utils.patch import ImageSegMerge, ImagePadToSize, CropBlack 
import copy
import skimage.measure as measure



def parse_args():
    parser = argparse.ArgumentParser(description='test Super Resolution Models')

    # data
    parser.add_argument('--data_path', default = '/mnt/bd/aurora-mtrc-data/datas/GIANA_challenge/segmentation/val', type = str, help = ' val image path')
    parser.add_argument('--model_upsample_num', default=5, type=int, help='val images size')
    parser.add_argument('--crop_black', default = True, type=bool)
    #test 
    parser.add_argument('--batch_size', default = 1, type = int, help = 'val batch size')
    parser.add_argument('--num_workers', default = 4, type = int, help = ' data loader num workers')
    parser.add_argument('--device', default = 'cuda:0', type = str, help = 'device')
    parser.add_argument('--result_save_path', default = './result', type = str)
    parser.add_argument('--model_path', default = './checkpoint/20210823_214049/model_best.pth.tar', type = str)
    parser.add_argument('--use_connect_domain', default = True, type = bool)
    # model
    parser.add_argument('--num_classes', default =2, type = int, help = 'number of classes')
    parser.add_argument('--model', default = None, type = str)
    

    args = parser.parse_args()
    return args


def main():
    args = parse_args() 
    print(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.result_save_path):
        os.mkdir(args.result_save_path)
    

    test_set = TestDataset(args)
    
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, \
        num_workers = args.num_workers)

    if args.model == 'segformer':
        from model.segformer.segformer import Segformer
        model = Segformer()
    elif args.model == 'deeplabv3plus':
        from model.deeplabv3Plus.deeplab import DeepLab
        model = DeepLab()
    elif args.model == 'unet':
        from model.unet.unet_model import UNet
        model = UNet()
    elif args.model == 'swin_unet':
        from model.swin_unet.vision_transformer import SwinUnet
        model = SwinUnet(args)
    else:
        raise('model error')
    print('# model parameters:', sum(param.numel() for param in model.parameters()))

    checkpoint = torch.load(args.model_path)['state_dict']
    print('epoch:', torch.load(args.model_path)['epoch'])
    print('best eval:', torch.load(args.model_path)['best_eval'])
    model.load_state_dict(checkpoint)
    
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        iou_list, dice_list = [[] for _ in range(9)], [[] for _ in range(9)] 
        for i, (imgs, labels, name_list) in enumerate(test_loader):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            #path_seg = ImageSegMerge(args.model_upsample_num, model, 1)
            #pred = path_seg(imgs)
            if args.crop_black:
                crop_black = CropBlack(imgs, args)
                crop_img = crop_black.crop()
                img_forward = ImagePadToSize(args.model_upsample_num, model)
                pred = img_forward(crop_img)
                pred = crop_black.merge(pred)
            else:
                img_forward = ImagePadToSize(args.model_upsample_num, model)
                pred = img_forward(imgs)
            pred = pred.detach().permute(0,2,3,1).squeeze().cpu().numpy()
            for j in range(1, 10):
                pred_tmp = pred.copy()
                threshold = j * 0.1
                if args.use_connect_domain:
                    #pred = utils.max_connected_domain(pred, args.threshold)
                    pred_tmp = max_prob_domain(pred_tmp, threshold)
                metrics = utils.IOUMetric(args.num_classes, threshold)
                metrics.add_batch(pred_tmp, labels)
                iou, dice = metrics.evaluate_front()
                iou_list[j-1].append(iou)
                dice_list[j-1].append(dice)
        best_th, best_iou, best_dice = 0, 0, 0
        for i in range(9):
            avg_iou, avg_dice = sum(iou_list[i]) / len(iou_list[i]), sum(dice_list[i]) / len(dice_list[i])
            if best_iou + best_dice < avg_iou + avg_dice:
                best_th, best_iou, best_dice = i+1, avg_iou, avg_dice
            print('mean iou: {}, mean dice: {}'.format(avg_iou, avg_dice))
        print('best threshold: {}, best iou: {}, best dice: {}'.format(best_th, best_iou, best_dice))
    print('Done.')


def max_prob_domain(pred, threshold):
    orig_pred = copy.deepcopy(pred) 
    pred[pred >= threshold] = 255
    pred[pred < threshold] = 0
    pred = pred.astype(np.uint8)

    #label_img, num = measure.label(img, neighbors=8,background=0,return_num = True)
    label_img = measure.label(pred, connectivity=2)
    max_label, max_avg_prob = 0, 0
    properties = measure.regionprops(label_img)
    for prop in properties:
        coords = prop.coords
        sum_num = sum([orig_pred[i,j] for i, j in coords])
        avg_prob = sum_num / prop.area
        if avg_prob > max_avg_prob:
            max_label = prop.label
            max_avg_prob = avg_prob 
    if max_label != 0:
        pred[label_img == max_label] = 1 
        pred[label_img != max_label] = 0
    return pred

if __name__ == '__main__':
    main() 
