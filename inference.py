import argparse
import os
from PIL import Image
import numpy as np
import cv2
import glob

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import utils.utils as utils
from utils.patch import ImageSegMerge, ImagePadToSize, CropBlack 



def parse_args():
    parser = argparse.ArgumentParser(description='test Super Resolution Models')

    # data
    parser.add_argument('--data_path', default = '/mnt/bd/aurora-mtrc-data/datas/GIANA_challenge/segmentation/val', type = str, help = ' val image path')
    parser.add_argument('--model_upsample_num', default=5, type=int, help='val images size')
    parser.add_argument('--crop_black', default = True, type=bool)
    #test 
    parser.add_argument('--batch_size', default = 1, type = int, help = 'val batch size')
    parser.add_argument('--num_workers', default = 4, type = int, help = ' data loader num workers')
    parser.add_argument('--threshold', default = 0.5, type = int)
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
        for i, (imgs, name_list) in enumerate(test_loader):
            imgs = imgs.to(args.device)
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
            if args.use_connect_domain:
                #pred = utils.max_connected_domain(pred, args.threshold)
                pred = utils.max_prob_domain(pred, args.threshold)
            else:
                pred = pred.detach().permute(0,2,3,1).squeeze().cpu().numpy()
            name = name_list[0].split('.')[0] +  '_mask.png'
            #pred *= 255
            result = Image.fromarray(pred.astype(np.uint8))
            save_path = os.path.join(args.result_save_path, name)
            result.save(save_path)
            print('save ', name)
    print('Done.')


class TestDataset(Dataset):
    def __init__(self, args): 
        super().__init__()
        image_path = args.data_path
        self.imgfiles = glob.glob(image_path + '/*.png')
        self.trans_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 

    def __getitem__(self, index):
        img_path = self.imgfiles[index]
        image = Image.open(img_path)
        name = os.path.basename(img_path)
        if self.trans_img:
            image = self.trans_img(image)
        return image, name 

    def __len__(self):
        return len(self.imgfiles)

if __name__ == '__main__':
    main() 
