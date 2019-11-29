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
    parser.add_argument("--model_name", default = "resnet50")
    parser.add_argument("--mode", default = "test")
    parser.add_argument("--root_dir", default = "../dataset")
    parser.add_argument("--use_weight_sample", type=bool, default=False, help='whether use use_weight_sample')
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--input_h", type=int, default=224)

    parser.add_argument('--tensorboard_dir', type=str, default='../tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='../model', help='save checkpoint infos')
    parser.add_argument("--save_model_name", default = "resnet50.pth")
    parser.add_argument("--save_count", type=int, default = 2000)
    parser.add_argument('--checkpoint', type=str, default='resnet50.pth')

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

def test_smoke(opt, test_loader, model, board):
    '''
    训练分类网络用于判断是否有火情发生
    '''
    model.cuda()
    model.eval()

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    n_fp, n_fn = 0, 0
    bad_case = []

    for step, inputs in enumerate(test_loader.data_loader):
        inputs = test_loader.next_batch()
        im = inputs['im'].cuda()
        im_name = inputs['im_name']
        label = inputs['label']
        label = label.clone().detach().long().cuda()

        # forward
        pred = model(im)
        pred = torch.argmax(pred, 1)
        correct += (pred == label).sum().float()
        total += len(label)

        wrong_pred = np.where((pred==label).cpu().numpy()==0)[0]
        for i in range(len(wrong_pred)):
            bad_case.append(im_name[wrong_pred[i]])
            if label[wrong_pred[i]].item()==0:
                n_fp += 1
            else:
                n_fn += 1
    np.savetxt('bad_case.txt', bad_case, fmt='%s')
    print('total: %d, acc: %4f, FN: %d, FP: %d' % (total.cpu().detach().data.numpy(), (correct/total).cpu().detach().data.numpy(), n_fn, n_fp))

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
    test_dataset = SmokeDataset(opt)
    test_dataloader = SmokeDataLoader(opt, test_dataset)

    # build network
    model = FireSmokeClf(opt)
    # if opt.use_gpu and len(gpu_ids) > 1:
    #     model = nn.DataParallel(model, device_ids=gpu_ids)
    #     model.to(torch.device("cuda:%d" % gpu_ids[0]))
    load_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.model_name, 'train', opt.save_model_name))
    with torch.no_grad():
        test_smoke(opt, test_dataloader, model, board)