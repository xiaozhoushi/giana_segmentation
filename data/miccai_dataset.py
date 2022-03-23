""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.utils.data as data
import os
import torch
import logging

from PIL import Image
import glob

_logger = logging.getLogger(__name__)
import torchvision.transforms as transforms
from .data_utils.data_transforms import * 



class TrainDataset(data.Dataset):
    def __init__(
            self,
            root,
            args,
            transform=None,
            img_transform=None,
            target_transform=None,
    ):
        self.root = root
        image_list = glob.glob(self.root+'/*/images/*.png')
        image_list += glob.glob(self.root+'/*/images/*.jpg')
        self.transform = transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.dataset = []
        for item in image_list:
            mask = item.replace('images', 'masks')
            if os.path.exists(mask):
                self.dataset.append([item, mask])
        self._consecutive_errors = 0
        print('------', len(self.dataset))

        self.trans_img = RandomChoiceOrder([
            RandomDownResizeUpResize((0.1, 1.0)),
            transforms.ColorJitter(0.5,0.5,0.5,0.2),
            #dt.AddPepperNoise(0.999, 0.2),
            AddGaussianNoise(0, 0.05),
            transforms.GaussianBlur(kernel_size = 5, sigma=(0.05, 5.0)),
            transforms.RandomPosterize(bits=2, p = 0.1),
            transforms.RandomSolarize(threshold = 245, p = 0.5),
            transforms.RandomAdjustSharpness(0.1, p=0.5)
            ], p = 0.5)
        self.trans_img2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 
        self. trans_all = Compose([
            #dt.RandomCrop(args.crop_size)
            RandomResizedCrop(args.crop_size, scale = (0.2, 1))
            ])
        self.trans_all_random = RandomChoiceOrderImgMask([
            RandomPerspective(0.1, 0.1),
            RandomAffine(60, translate = (0.05, 0.05), scale = (0.8, 1.0)),
            RandomHorizontalFlip(),
            RandomVerticalFlip()
            ], p = 0.5)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        try:
            img =  Image.open(img).convert('RGB')
            target = Image.open(target).convert('L')
        except Exception as e:
            raise e
        image, mask = self.trans_all(img, target)
        image, mask = self.trans_all_random(image, mask)
        image = self.trans_img(image)
        target = self.trans_mask(mask)
        img = self.trans_img2tensor(image)
        return img, target

    def __len__(self):
        return len(self.dataset)


class ValDataset(data.Dataset):
    def __init__(
            self,
            root,
            args,
            transform=None,
            img_transform=None,
            target_transform=None,
    ):
        self.root = root
        image_list = glob.glob(self.root+'/*/images/*.png')
        image_list += glob.glob(self.root+'/*/images/*.jpg')
        self.transform = transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.dataset = []
        for item in image_list:
            mask = item.replace('images', 'masks')
            if os.path.exists(mask):
                self.dataset.append([item, mask])
        self._consecutive_errors = 0
        print('------', len(self.dataset))

        self.trans_img2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 

    def __getitem__(self, index):
        img, target = self.dataset[index]
        try:
            img =  Image.open(img).convert('RGB')
            target = Image.open(target).convert('L')
        except Exception as e:
            raise e
        target = self.trans_mask(target)
        img = self.trans_img2tensor(img)
        return img, target

    def __len__(self):
        return len(self.dataset)


class TestDataset(data.Dataset):
    def __init__(
            self,
            root,
            args,
            transform=None,
            img_transform=None,
            target_transform=None,
    ):
        self.root = root
        image_list = glob.glob(self.root+'/images/*.png')
        image_list += glob.glob(self.root+'/images/*.jpg')
        self.transform = transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.dataset = []
        for item in image_list:
            mask = item.replace('images', 'masks')
            if os.path.exists(mask):
                self.dataset.append([item, mask])
        self._consecutive_errors = 0
        print('------', len(self.dataset))

        self.trans_img2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 

    def __getitem__(self, index):
        img, target = self.dataset[index]
        name = os.path.basename(img)
        try:
            img =  Image.open(img).convert('RGB')
            target = Image.open(target).convert('L')
        except Exception as e:
            raise e
        target = self.trans_mask(target)
        img = self.trans_img2tensor(img)
        return img, target, name

    def __len__(self):
        return len(self.dataset)
