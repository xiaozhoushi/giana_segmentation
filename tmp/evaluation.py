import argparse
import os
from math import log10
from datetime import datetime
import time
import random
from PIL import Image
import matplotlib.pyplot as plt

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

parser.add_argument('--data_dir', default = '/home/tiger/dataset/class_test', type = str, help = ' test image path')
parser.add_argument('--checkpoint', default = './checkpoint/model_best.pth.tar', type = str, help = 'model save path')
parser.add_argument('--resize_size', default=(256,256), type=int, help='training images size')
parser.add_argument('--batch_size', default = 1, type = int, help = 'training batch size')
parser.add_argument('--nums_gpu', default = 1, type = int, help = 'use gpu nums')
parser.add_argument('--print_freq', default = 50, type = int, help = 'the frequency with which messages are printed')
parser.add_argument('--num_workers', default = 4, type = int, help = ' data loader num workers')
parser.add_argument('--threshold', default = 0.5, type = int)
parser.add_argument('--save_path', default = './checkpoint', type = str)
parser.add_argument('--rows', default = 4, type = int)
parser.add_argument('--cloms', default = 5, type = int)

parser.add_argument('--seed', default=None)

args = parser.parse_args()


def main(index):
    global args
    time.sleep(xm.get_ordinal())
    device = xm.xla_device()
    print("Process: {}, xla device: {}, real device: {}, rank: {}, world size: {}".format(index, device, 
          xm.xla_real_devices([str(device)])[0], xm.get_ordinal(), xm.xrt_world_size()))
    args.rank = xm.xla_real_devices([str(device)])[0]
 
    transform = transforms.Compose([transforms.Resize(args.resize_size),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])

    test_data, _ = split_dataset(args.data_dir, shuffle=False, ratio=1)

    test_set = EvalDataset(test_data, transforms = transform)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    resnet50 = ResNet50()
    print('# model parameters:', sum(param.numel() for param in resnet50.parameters()))

    resnet50.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    resnet50.to(device)


    resnet50.eval()
    with torch.no_grad():
        labels = []
        preds = []
        paths = []
        for i, (imgs, label, path) in enumerate(test_loader):
            imgs = imgs.to(device)
            label = label.to(device)
            pred = resnet50(imgs)
            print('step: {}'.format(i))
            pred, label = pred.squeeze().detach().cpu().numpy(), label.squeeze().detach().cpu().numpy()
            labels.append(label)
            preds.append(pred)
            paths.append(path[0])
        labels, preds = np.asarray(labels), np.asarray(preds)
        positive, negative = [], []
        positive_negative_path, negative_positive_path = [], []
        positive_negative, negative_positive = [], []
        for i in range(len(labels)):
            if labels[i] == 1:
                positive.append(preds[i])
            else:
                negative.append(preds[i])
            if labels[i] == 1 and preds[i] < args.threshold:
                positive_negative_path.append(paths[i])
                positive_negative.append(preds[i])
            elif labels[i] == 0 and preds[i] >= args.threshold:
                negative_positive_path.append(paths[i])
                negative_positive.append(preds[i])

        plt.figure()
        plt.title('data distribute')
        kwargs = dict(histtype = 'stepfilled', alpha = 0.7, bins = 10)
        n1, bins1, _ = plt.hist(positive, label = 'postive example', **kwargs)
        n2, bins2, _ = plt.hist(negative, label = 'negative example', **kwargs)
        add_labels(n1, bins1, 'left', 'blue')
        add_labels(n2, bins2, 'right', 'red')
        plt.legend(loc = 'upper right')
        plt.ylabel('num')
        plt.xlabel('predict')
        # plt.xticks(rotation = 30)
        plt.tick_params(labelsize = 20)
        plt.savefig(os.path.join(args.save_path, 'pred_distribute.png'))

        auc, acc, precision, f1, recall, matrix = utils.evaluation(preds, labels, args.threshold)
        print('evaluation: auc: {}, acc: {}, precision: {}, f1: {}, recall: {}, matrix: {}.'.format(auc, acc, precision, f1, recall, matrix))

        plt.figure()
        plt.title('confusion matrix')
        plt.matshow(matrix)
        plt.colorbar()
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                plt.annotate(matrix[i, j], xy = (j,i))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(args.save_path, 'confusion_matrix.png'))
        
        plt.figure()
        #plt.suptitle('posit predict nega')
        for index in range(args.rows * args.cloms):
            img = Image.open(positive_negative_path[index])
            name = 'score:' + str(round(positive_negative[index], 2))
            plt.subplot(args.rows, args.cloms, index + 1)
            plt.title(name, fontsize = 10)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_path, 'post_predict_nega.png'))

        plt.figure()
        #plt.suptitle('nega predict posit')
        for index in range(args.rows * args.cloms):
            img = Image.open(negative_positive_path[index])
            name = 'score:' + str(round(negative_positive[index], 2))
            plt.subplot(args.rows, args.cloms, index + 1)
            plt.title(name, fontsize = 10)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_path, 'nage_predict_posit.png'))
        plt.show()

    print('Done.')

def add_labels(n, bins, local, color):
    if local == 'left':
        width = 0
    else:
        width = bins[1] - bins[0]
    for num, x in zip(n, bins):
        plt.text(x + width / 2, num + 5, int(num), rotation = 'vertical', ha = 'center', va = 'bottom', fontsize = 10, color = color)

if __name__ == '__main__':
    print('turing version:', turing.__version__)   # turing
    token = '09efffbfb7ce22febd0d3c1db998e44c6490252b'
    cluster = ArnoldClusterResolver(token=token, num_gpus=args.nums_gpu)
    print('{} gpu to use'.format(args.nums_gpu))
    torch.set_default_tensor_type('torch.FloatTensor')
    main(0)
    