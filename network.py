import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import numpy as np

os.environ['TORCH_HOME'] = '/ssd/yqian/smoke/model/pretrained'

BaseModel = {
    'resnet18': models.resnet18(pretrained=True),
    'resnet34': models.resnet34(pretrained=True),
    'resnet50': models.resnet50(pretrained=True),
}

class FireSmokeClf(nn.Module):
    def __init__(self, opt):
        super(FireSmokeClf, self).__init__()
        self.opt = opt
        self.model = BaseModel[opt.model_name]
        self.n_feat = self.model.fc.in_features
        self.model.fc = nn.Linear(self.n_feat, 2).cuda()
        self.model.cuda()

    def forward(self, x):
        return self.model(x)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path, param_filter=''):
    if not os.path.exists(checkpoint_path):
        print('wrong model path')
        return
    if param_filter == '':
        model.load_state_dict(torch.load(checkpoint_path))
        model.cuda()
    else:
        trained_dict = torch.load(checkpoint_path)
        model.model_part.load_state_dict(trained_dict)
        model.cuda()