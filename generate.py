from model import GNet, DNet

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.utils as vutils
from torchvision import models, datasets, transforms

from collections import OrderedDict
#import os
import random
import numpy as np



def generate(opt, device):

    criterion = nn.BCELoss()
    
    dnet = DNet(opt).to(device)  # 可能需要从其他GPU移动到0号，若满足条件则不作为
    gnet = GNet(opt).to(device)

    state_dict = torch.load('dnet.pth')
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    dnet.load_state_dict(new_state_dict)
    
    state_dict = torch.load('gnet.pth')
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    gnet.load_state_dict(new_state_dict)

    dnet.eval()
    gnet.eval()

    noise = torch.randn(opt.batch_size, opt.nd, 1, 1, device=device)
    with torch.no_grad():
        fake = gnet(noise)
        output = dnet(fake)
    label = torch.full((opt.batch_size, ), opt.real_label, device=device)
    d_err_fake = criterion(output, label)  # 生成图像的损失；还是tensor
    mean_score = output.mean()  #生成图像的平均得分；还是tensor
    fake_img = vutils.make_grid(fake, normalize=True)

    writer = SummaryWriter(log_dir='generate_result')
    writer.add_image('fake_img', fake_img)
    writer.close()
    print('生成图像的平均损失值：%.4f' % d_err_fake.item())
    print('生成图像的平均得分：%.4f' % mean_score.item())