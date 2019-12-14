#coding=utf-8
# Train network to judge whether there is fire in image
"""
date: 2019/11/28
Author: Yang Qian
"""
# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import argparse
import os
import numpy as np
import time
from datetime import datetime

from dataset import *
from network import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "resnet50", help='network type, must be one of (resnet18,resnet34, resnet50)')
    parser.add_argument("--mode", default = "train")
    parser.add_argument("--root_dir", default = "../dataset")
    parser.add_argument("--use_weight_sample", type=bool, default=False, help='whether use use_weight_sample')
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--n_class", type=int, default=3, help='number of classes')

    parser.add_argument('--tensorboard_dir', type=str, default='../tensorboard', help='directory to save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='../model', help='directory to save checkpoint infos')
    parser.add_argument("--save_model_name", default = "resnet50.pth")
    parser.add_argument("--save_count", type=int, default = 2000)
    parser.add_argument('--checkpoint', type=str, default='')

    parser.add_argument("--total_step", type=int, default = 20000)
    parser.add_argument("--keep_step", type=int, default = 10000)
    parser.add_argument("--decay_step", type=int, default = 10000)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--use_gpu', type=bool, default=True)

    opt = parser.parse_args()
    return opt

def train_smoke(opt, train_loader, model, board):
    '''
    训练分类网络
    '''
    model.cuda()
    model.train()

    # loss
    criterionClf = nn.CrossEntropyLoss()
    avr_acc = AverageMeter('average acc')

    # optimizer
    params = [
        {'params': model.parameters(), 'initial_lr': opt.lr, 'lr': opt.lr},
    ]
    optimizer = torch.optim.Adam(params, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.total_step):
        inputs = train_loader.next_batch()
        im = inputs['im'].cuda()
        label = inputs['label']
        label = label.clone().detach().long().cuda()

        # forward
        pred = model(im)

        # update accuracy
        n_batch = pred.shape[0]
        acc = torch.sum(torch.argmax(pred, 1) == label).double().cpu().numpy() / n_batch
        avr_acc.update(acc, n_batch)

        # calculate loss
        loss = criterionClf(pred, label)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step+1) % opt.display_count == 0:
            # board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('acc', acc, step+1)
            board.add_scalar('loss', loss.item(), step+1)
            board.add_scalar('lr', optimizer.param_groups[0]['lr'], step+1)
            print('step: %8d, loss: %4f' % (step+1, loss.item()))

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.model_name, opt.mode, 'step_%06d.pth' % (step+1)))

if __name__ == "__main__":
    opt = get_opt()

    # get the number of available free gpu
    gpu_ids = []
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    for i in range(len(memory_gpu)):
        if memory_gpu[i] > 8000:
            gpu_ids.append(i)
    os.system('rm tmp')

    while True:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
        memory_max = max(memory_gpu)
        if memory_max>8000:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
            os.system('rm tmp')
            print('Find vacant GPU!')
            break

    # visualization
    if not os.path.exists(os.path.join(opt.tensorboard_dir, opt.model_name, opt.mode)):
        os.makedirs(os.path.join(opt.tensorboard_dir, opt.model_name, opt.mode))
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.model_name, opt.mode, time_stamp))

    # load dataset
    train_dataset = SmokeDataset(opt)
    train_dataloader = SmokeDataLoader(opt, train_dataset)

    # build network
    model = FireSmokeClf(opt)
    # if opt.use_gpu and len(gpu_ids) > 1:
    #     model = nn.DataParallel(model, device_ids=gpu_ids)
    #     model.to(torch.device("cuda:%d" % gpu_ids[0]))
    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    train_smoke(opt, train_dataloader, model, board)
    save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.model_name, opt.mode, opt.save_model_name))