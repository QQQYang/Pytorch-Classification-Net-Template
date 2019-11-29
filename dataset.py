#coding=utf-8
# Function for smoke dataset
"""
date: 2019/11/28
Author: Yang Qian
"""
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import numpy as np
import os
from PIL import Image

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

class SmokeDataset(data.Dataset):
    '''
    define dataset for smoke and fire recognition
    '''
    def __init__(self, opt):
        super(SmokeDataset, self).__init__()
        self.opt = opt
        
        # load data
        self.data_dir = os.path.join(opt.root_dir, opt.mode)
        im_names_fire = os.listdir(os.path.join(self.data_dir, 'fire'))
        im_names_nonfire = os.listdir(os.path.join(self.data_dir, 'nonfire'))
        self.im_names = im_names_fire + im_names_nonfire
        self.labels = np.concatenate((np.ones(len(im_names_fire), dtype=np.int), np.zeros(len(im_names_nonfire), dtype=np.int)))

        # transform
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ColorJitter(1),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        }
        
        self.resize_pad = ResizePad()

    def __getitem__(self, index):
        label = self.labels[index]
        if label == 1:
            im_name = os.path.join(self.data_dir, 'fire', self.im_names[index])
        else:
            im_name = os.path.join(self.data_dir, 'nonfire', self.im_names[index])
        
        im = cv2.imread(im_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = self.resize_pad(im, self.opt.input_w, self.opt.input_h)
        im = Image.fromarray(im.astype(np.uint8))
        im = self.transform[self.opt.mode](im)

        result = {
            'im': im,
            'im_name': im_name,
            'label': label,
        }
        return result

    def __len__(self):
        return len(self.im_names)

class SmokeDataLoader(object):
    def __init__(self, opt, dataset):
        super(SmokeDataLoader, self).__init__()

        if opt.mode :
            if opt.use_weight_sample is True:
                train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=dataset.weight, num_samples=len(dataset), replacement=True)
            else:
                train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch