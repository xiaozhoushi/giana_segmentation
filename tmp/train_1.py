import argparse
import os
from datetime import datetime
import time
import random
import warnings

import torch
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.utils.data import sampler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from tensorboardX import SummaryWriter 
import torchvision.utils as vutils
from torch import nn

import torch.distributed as dist

from data.GianaDataset import TrainDataset, ValDataset
import utils.utils as utils
from utils.loss import focal_loss, dice_loss
from model.segformer import Segformer
from utils.patch import ImagePadToSize



def parse_args():
    parser = argparse.ArgumentParser(description='Train Models')

    # data
    parser.add_argument('--data_path', default ='/mnt/bd/aurora-mtrc-data/datas/GIANA_challenge/segmentation/train', type = str, help = ' training image path')
    parser.add_argument('--val_path', default ='/mnt/bd/aurora-mtrc-data/datas/GIANA_challenge/segmentation/val', type = str, help = ' training image path')
    parser.add_argument('--save_path', default = './checkpoint', type = str, help = 'model save path')
    parser.add_argument('--crop_size', default=(512,512), type=int, help='training images size')
    # train
    parser.add_argument('--num_epochs', default=2000, type=int, help='train epoch number')
    parser.add_argument('--batch_size', default = 4, type = int, help = 'training batch size')
    parser.add_argument('--lr', default = 0.00006, type = float, help='learning rate')
    parser.add_argument('--print_freq', default = 10, type = int, help = 'the frequency with which messages are printed')
    parser.add_argument('--tensorboard_freq', default = 200, type = int, help = 'the frequency with tensorboard')
    parser.add_argument('--save_freq', default = 10, type = int, help = 'save model according to epoch')
    parser.add_argument('--best_eval', default = 0, type = float, help = 'best auc of model')
    parser.add_argument('--num_workers', default = 4, type = int, help = ' data loader num workers')
    parser.add_argument('--decay_gamma', default = 1, type = int)
    parser.add_argument('--threshold', default = 0.5, type = int)
    parser.add_argument('--benchmark', default = True, type = bool)
    # distributed
    parser.add_argument('--gpus', default = 2, type = int, help = 'number of gpus of per node')
    parser.add_argument('--local_rank', type = int, help = 'rank of current process')
    parser.add_argument('--init_method', default = 'env://')
    parser.add_argument('--device', type = str)
    # model
    parser.add_argument('--num_classes', default =2, type = int, help = 'number of classes')
    # other
    parser.add_argument('--seed', default=10, type = int, help = 'seed for initializing training. ')
    parser.add_argument('--reuse', default = './checkpoint/20210715_213009/model_best.pth.tar')
    

    args = parser.parse_args()
    return args


def main():
    args = parse_args() 
    dist.init_process_group(backend='nccl', init_method = args.init_method, rank = 0, world_size = 1)
    args.device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.save_path = os.path.join(args.save_path,timestamp)

    print('print device:', args.local_rank)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    writer = SummaryWriter(args.save_path)
    

    train_set = TrainDataset(args)
    val_set = ValDataset(args)
    print('train set lenght:', train_set.__len__(), 'val set lenght:', val_set.__len__())
    
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, \
        shuffle = True, num_workers = args.num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=1, \
        num_workers = args.num_workers)

    model = Segformer()
    print('# model parameters:', sum(param.numel() for param in model.parameters()))

    if args.reuse is not None:
        model.load_state_dict(torch.load(args.reuse)['state_dict'])

    
    model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    loss_func = nn.BCELoss()
    # expLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.decay_gamma)

    start = datetime.now()
    steps = train_loader.__len__()
    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        metrics = utils.IOUMetric(args.num_classes, args.threshold)
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.cuda()
            labels = labels.cuda()
            pred = model(imgs)
            # loss = loss_func(pred, labels)
            #loss = focal_loss(pred, labels, alpha = 0.7, reduction = 'mean')
            loss = dice_loss(pred, labels) + loss_func(pred, labels)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

            metrics.add_batch(pred, labels)
            acc, acc_cls, iu, mean_iu, fwavacc = metrics.evaluate()
            if step % args.print_freq == 0:
                print("epoch: {}|{}, step: {}|{}, loss: {:.5f}, acc: {:.2f}, iu: {}, mean_iu: {}, time: {}".format(epoch, args.num_epochs, i, steps, \
                    loss.item(), acc, iu, mean_iu, datetime.now() - start))
                writer.add_scalar('train/BCE loss', loss.item(), step)
                writer.add_scalar('train/acc', acc, step)
                writer.add_scalar('train/iu', iu[1], step)
                writer.add_scalar('train/mean_iu', mean_iu, step)
                start = datetime.now()
            
            if step % args.tensorboard_freq == 0:
                imgs = vutils.make_grid(imgs, normalize=True, scale_each=True)
                writer.add_image('trian/inputs', imgs, step)
                gts = vutils.make_grid(labels, normalize=True, scale_each=True)
                writer.add_image('train/gts', gts, step)
                preds= vutils.make_grid(pred, normalize=True, scale_each=True)
                writer.add_image('train/preds', preds, step)
            step += 1
        # expLR.step()
        model.eval()
        is_best = False
        with torch.no_grad():
            metrics = utils.IOUMetric(args.num_classes, args.threshold)
            for i, (imgs, labels) in enumerate(val_loader):
                metrics = utils.IOUMetric(args.num_classes, args.threshold)
                imgs = imgs.cuda()
                labels = labels.cuda()
                img_forward = ImagePadToSize(args.crop_size, model)
                pred = img_forward(imgs)
                loss = loss_func(pred, labels)
                metrics.add_batch(pred, labels)
                acc, acc_cls, iu, mean_iu, fwavacc = metrics.evaluate()
                print(mean_iu)

            acc, acc_cls, iu, mean_iu, fwavacc = metrics.evaluate()
            writer.add_scalar('val/acc', acc, epoch)
            writer.add_scalar('val/iu', iu[1], epoch)
            writer.add_scalar('val/mean_iu', mean_iu, epoch)
            imgs = vutils.make_grid(imgs, normalize=True, scale_each=True)
            writer.add_image('val/inputs', imgs, epoch)
            gts = vutils.make_grid(labels, normalize=True, scale_each=True)
            writer.add_image('val/gts', gts, epoch)
            preds= vutils.make_grid(pred, normalize=True, scale_each=True)
            writer.add_image('val/preds', preds, epoch)
            print('test acc: {}, iu: {}, mean_iu: {}'.format(acc, iu, mean_iu))
            if mean_iu > args.best_eval:
                args.best_eval = mean_iu 
                is_best = True
            # save model parameters
            if epoch % args.save_freq == 0 or is_best:
                print('save checkpoint.')
                #os.system("hdfs dfs -put -f ./run $ARNOLD_OUTPUT")
                utils.save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_eval': args.best_eval, 'optimizer': optimizer.state_dict()}, \
                                    is_best, args.save_path)

    print('Done.')


if __name__ == '__main__':
    main() 
