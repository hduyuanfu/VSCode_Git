from __future__ import print_function  # 这行必须是第一行

from model import GNet, DNet

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict

import torchvision
import torchvision.utils as vutils
from torchvision import models, datasets, transforms

#import os
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"  # 选择显卡
class Config(object):
    data_path = '/data/yuanfu/GAN/Data'  # 数据集存放路径
    num_workers = 0  # 多进程加载数据所用的进程数,0为主进程
    image_size = 96  # 图片尺寸
    batch_size = 64
    max_epoch = 5
    real_label = 1
    fake_label = 0
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    nd = 100  # 噪声维度
    gfm = 64  # 生成器feature map数
    dfm = 64  # 判别器feature map数
    #noise_mean = 0  # 噪声的均值
    #noise_std = 1  # 噪声的方差
    '''官网有多卡教程'''

def weight_init(model):
    classname = model.__class__.__name__
    print('Initializing Parameters of %s !' % classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    print('finshed parameters initalizations!')

def data_loader(opt):
    transform = transforms.Compose([    # 评估不需要加载真实图片
                                        transforms.Resize(opt.image_size), #重新设置图片大小，opt.image_size默认值为96
                                        transforms.CenterCrop(opt.image_size), #从中心截取大小为opt.image_size的图片
                                        transforms.ToTensor(), #转为Tensor格式，并将值取在[0,1]中
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #标准化，得到在[-1,1]的值
                                    ])
    dataset = datasets.ImageFolder(opt.data_path, transform=transform) #从data中读取图片，图片类别会设置为文件夹名faces
    dataloader = torch.utils.data.DataLoader(dataset, #然后对得到的图片进行批处理，默认一批为256张图，使用4个进程读取数据
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                num_workers=opt.num_workers,
                                                drop_last=True
                                                )
    print('%d images were found there!' % len(dataloader))
    return dataloader

def train():
    opt = Config()  # 配置的实例
    dataloader = data_loader(opt)
    criterion = nn.BCELoss()

    device = torch.device("cuda: 0, 1, 2" if torch.cuda.is_available() else "cpu")  # 有cuda这个架构就使用0卡，无则cpu;device(可以指定任何设备，cpu,哪块或哪几块显卡)
    gnet = GNet(opt).to(device)
    dnet = DNet(opt).to(device)
    #writer.add_graph(gnet)'''做实验试验下第二个参数'''
    #writer.add_graph(dnet)
    if device.type == 'cuda':
        gnet = nn.DataParallel(gnet, [0, 1, 2])
        dnet = nn.DataParallel(dnet, [0, 1, 2])
    gnet.apply(weight_init)
    dnet.apply(weight_init)
    print('Generative NetWork:')
    print(gnet)
    print('')
    print('Discriminative NetWork:')
    print(dnet)

    g_optimizer = optim.Adam(gnet.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
    d_optimizer = optim.Adam(dnet.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))
    print('g_optimizer:')
    print(g_optimizer)
    print('d_optimizer:')
    print(d_optimizer)

    writer = SummaryWriter(log_dir='result_shiyan')
    #dummy1_input = torch.rand(opt.batch_size, 3, 96,96)
    #dummy2_input = torch.rand(opt.batch_size, opt.nd,1,1)
    #writer.add_graph(dnet, dummy1_input.detach())
    #writer.add_graph(gnet, dumm2_input.detach())
    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    fixed_noise = torch.randn(opt.batch_size, opt.nd, 1, 1, device=device)
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(1, opt.max_epoch + 1):
        # For each batch in the dataloader
        print(len(dataloader))
        print(type(dataloader))
        for i, (imgs, _) in enumerate(dataloader, 1):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            ## Train with all-real batch
            dnet.zero_grad()
            '''先训练判别器，再训练生成器'''
            # Format batch
            real_img = imgs.to(device)
            label = torch.full((opt.batch_size, ), opt.real_label, device=device)
            # Forward pass real batch through D
            output = dnet(real_img)  # 在model模块中已经被展成一维的啦
            # Calculate loss on all-real batch
            d_err_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            d_err_real.backward()
            D_x = output.mean().item()  # 真实图片的平均得分，当然是接近1越好

            ## Train with all-fake batch
            # Generate batch of latent vectors  latent:隐藏的，潜伏的
            noise = torch.randn(opt.batch_size, opt.nd, 1, 1, device=device)
            # Generate fake image batch with G
            fake = gnet(noise)
            label.fill_(opt.fake_label)
            # Classify all fake batch with D
            output = dnet(fake.detach())
            # Calculate D's loss on the all-fake batch
            d_err_fake = criterion(output, label)
            # Calculate the gradients for this batch
            d_err_fake.backward()
            D_G_z1 = output.mean().item()  # 假图像的分数，自然是越接近0越好
            # Add the gradients from the all-real and all-fake batches
            d_err = d_err_real + d_err_fake  #tensor(1.272)+tensor(0.183)可以直接相加，不需要先取出数值。tensor
            # 自成体系，tensor和tensor的加减乘除和标量一模一样；只是tensor和标量之间不能直接算
            # Update D
            d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            gnet.zero_grad()
            label.fill_(opt.real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = dnet(fake)  # 更新了一步D网络后，同样一批假图片，自然是希望判别得分output比更新前的假图片得分要小,也就是使下面的g_err扩大
            # Calculate G's loss based on this output
            g_err = criterion(output, label)
            '''生成器就是要把假图片往真标签身上凑；所以假图片+真标签，进行比较后，损失越小越好'''
            # Calculate gradients for G
            g_err.backward()
            D_G_z2 = output.mean().item()  # 因为更新过一次判别器，所以这个假图片的output均值应该比上面的假图片的output均值更接近0才健康
            # Update G
            g_optimizer.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\treal_img_mean_score: %.4f\tfake_img_mean_score_1/2: %.4f / %.4f'
                    % (epoch, opt.max_epoch, i, len(dataloader),d_err.item(), g_err.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(g_err.item())
            D_losses.append(d_err.item())

            writer.add_scalars('dnet_gnet_loss', {'G_losses': G_losses[iters], 'D_losses': D_losses[iters]}, iters)
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == opt.max_epoch) and (i == len(dataloader))):
                with torch.no_grad():
                    fake = gnet(fixed_noise)#.detach().cpu()
                img_list.append(vutils.make_grid(fake, normalize=True))
                '''还不知道合成的图有多少个小图呢'''
                writer.add_image('fake%d'%(iters/500), img_list[int(iters/500)], int(iters/500))

            iters += 1
    
    #torch.save(dnet, 'dnet1.pth')
    #torch.save(gnet, 'gnet1.pth')
    torch.save(dnet.state_dict(), 'dd.pth')
    torch.save(gnet.state_dict(), 'gg.pth')

    #writer = SummaryWriter(log_dir='result_')
    print('最后的iters为: %d'%iters)
    print('G_losses长度为: %d'%len(G_losses))
    print('D_losses长度为: %d'%len(D_losses))
    #for i in range(iters):
        #writer.add_scalars('dnet_gnet_loss', {'G_losses': G_losses[i], 'D_losses': D_losses[i]}, i)
    print('img_list的长度为: %d'%len(img_list))
    #for i in range(len(img_list)):
        #writer.add_image('fake%d'%i, img_list[i], i)
    #writer.add_graph(dnet, input_to_model=(torch.rand(opt.batch_size, 3, 96, 96), ), verbose=False)
    #writer.add_graph(gnet, input_to_model=(torch.rand(opt.batch_size, op.nd, 1, 1), ), verbose=False)
    writer.close()

def generate():
    opt = Config()
    criterion = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# 训练可能多卡，预测一张就够了，所以有点小不同
    #dnet = torch.load('dnet1.pth').to(device)#可能需要从其他GPU移动到0号，若满足条件则不作为
    #gnet = torch.load('gnet1.pth').to(device)
    dnet = DNet(opt).to(device)
    gnet = GNet(opt).to(device)
    
    state_dict = torch.load('dd.pth')
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    dnet.load_state_dict(new_state_dict)
    
    state_dict = torch.load('gg.pth')
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    gnet.load_state_dict(new_state_dict)

    
    dnet.eval()
    gnet.eval()
    noise = torch.randn(opt.batch_size, opt.nd, 1, 1, device=device)
    #with torch.no_grad():
    fake = gnet(noise)
    output = dnet(fake)
    label = torch.full((opt.batch_size, ), opt.real_label, device=device)
    d_err_fake = criterion(output, label)  # 生成图像的损失；还是tensor
    mean_score = output.mean()  #生成图像的平均得分；还是tensor
    fake_img = vutils.make_grid(fake, normalize=True)

    writer = SummaryWriter(log_dir='generate_rusult')
    writer.add_image('fake_img', fake_img)
    writer.close()
    print('生成图像的平均损失值：%.4f'%d_err_fake.item())
    print('生成图像的平均得分：%.4f'%mean_score.item())

if __name__ == "__main__":
    you = 'train'
    if you == 'train':
        train()
    else:
        generate()