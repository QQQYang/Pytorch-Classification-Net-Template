#coding=utf-8
# inference for tree classification
"""
date: 2019/12/17
Author: Yang Qian, Yinlin Li
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

import argparse
import os
import numpy as np
import cv2

class TreeClf(nn.Module):
    def __init__(self, opt):
        super(TreeClf, self).__init__()
        self.opt = opt
        self.model = models.resnet18(pretrained=True)
        self.n_feat = self.model.fc.in_features
        self.model.fc = nn.Linear(self.n_feat, opt.n_class).cuda()
        self.model.cuda()

    def forward(self, x):
        return self.model(x)

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_weight_sample", type=bool, default=False, help='whether use use_weight_sample')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument("--mode", default="test")

    # the following two directories should be reconfigured
    parser.add_argument("--img_dirname",
                        default="/media/liyinlin/lyl_work/datasets/languang-dataset-1/Gingko/银杏23cm,h 9-10m,w5-6m (6).jpg",
                        help='read image from the img_dirname directory')
    parser.add_argument("--save_model_dirname", default="../model/resnet18/train/resnet18.pth",
                        help='diretory that save the pre-trained model')
    #***********************


    #the following args may be updated in the future
    parser.add_argument("--input_w", type=int, default=196)  # 224
    parser.add_argument("--input_h", type=int, default=264)  # 224
    parser.add_argument("--n_class", type=int, default=3, help='number of classes')
    parser.add_argument("--label2class", type=dict, default={0: 'hackberry', 1: 'fragran', 2: 'gingko'})
    parser.add_argument("--model_name", default="resnet18",
                        help='network type, must be one of (resnet18,resnet34, resnet50)')

    #configure the wokers accordign to your computer
    parser.add_argument('-j', '--workers', type=int, default=1)

    opt = parser.parse_args()
    return opt

class ResizePad(object):
    """
    convert to original bag image to target size
    """
    def __call__(self, img, target_w, target_h):
        h, w, _ = img.shape
        scale_h = target_h / h
        scale_w = target_w / w
        if scale_h*w <= target_w:
            scale = scale_h
        else:
            scale = scale_w
        img = cv2.resize(img, (int(round(w*scale)), int(round(h*scale))))
        h, w, c = img.shape
        img_pad = np.zeros((target_h, target_w, c), dtype=np.float32)
        img_pad[:h, :w, :] = img.copy()
        return img_pad

def img_pre_process(opt):
    transform = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }
    im = cv2.imread(opt.img_dirname)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    resize_pad = ResizePad()
    im = resize_pad(im, opt.input_w, opt.input_h)
    im = Image.fromarray(im.astype(np.uint8))
    im = transform[opt.mode](im)
    return im

def test_tree(opt, model):
    model.cuda()
    model.eval()

    im = img_pre_process(opt).unsqueeze(0)
    im=im.cuda()
    # forward
    pred = model(im)
    ## add softmax to output probability
    prob = F.softmax(pred, dim=-1).cpu().numpy()
    pred_classid = torch.argmax(pred, 1).cpu()
    pred_classid=pred_classid.numpy()
    pred_classname = opt.label2class[pred_classid[0]]

    ##output of the classification network
    print('class_id = %d, class_name = %s, probability = %f' % (pred_classid, pred_classname, prob[0][pred_classid[0]]))

if __name__ == "__main__":
    opt = get_opt()

    # build network
    model = TreeClf(opt)
    model.load_state_dict(torch.load(opt.save_model_dirname))
    with torch.no_grad():
        test_tree(opt, model)