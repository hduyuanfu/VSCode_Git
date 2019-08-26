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

from model import DNet, GNet, weight_init
from preparedata import data_loader


def train(opt ,device):

    dataloader = data_loader(opt)

    gnet = GNet(opt).to(device)
    dnet = DNet(opt).to(device)
    #writer.add_graph(gnet)'''做实验试验下第二个参数'''
    #writer.add_graph(dnet)
    if device.type == 'cuda':  # 就算device里有多个GPU可见，但是若不用分发功能，仍然只有第0块在跑
        gnet = nn.DataParallel(gnet, [0, 1, 2])  # list(range(ngpu))不好使，只能用前几个
        dnet = nn.DataParallel(dnet, [0, 1, 2])
    gnet.apply(weight_init)  # 也就是初始化了下面的d/gnet.parameters()；不进行初始化则会系统给你进行一次随机初始
    dnet.apply(weight_init)
    print('Generative NetWork:')
    print(gnet)
    print('')
    print('Discriminative NetWork:')
    print(dnet)

    criterion = nn.BCELoss()

    '''
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    除了下面的整体赋值，还可以通过迭代给优化器赋值，把模型中所有需要参数的过程都分别设置值;如学长代码：
    optimizer = optim.SGD([
                            {'params': model.features.parameters(), 'lr': 0.1 * lr},
                            {'params': model.sample_128.parameters(), 'lr': lr},
                            {'params': model.sample_256.parameters(), 'lr': lr},
                            {'params': model.fc_concat.parameters(), 'lr': lr}
                        ], lr=1e-1, momentum=0.9, weight_decay=1e-5)
    '''
    g_optimizer = optim.Adam(gnet.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
    d_optimizer = optim.Adam(dnet.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))
    # 优化器只会进行一次初始赋值，其他都是反向调整
    print('g_optimizer:')
    print(g_optimizer)
    print('d_optimizer:')
    print(d_optimizer)

    writer = SummaryWriter(log_dir='train_result')
    # 定义writer时候就会生成events文件，而tensorboard执行时会搜索大文件下的所有路径，找出所有需要的文件
    #dummy1_input = torch.rand(opt.batch_size, 3, 96,96)
    #dummy2_input = torch.rand(opt.batch_size, opt.nd,1,1)
    #writer.add_graph(dnet, dummy1_input)
    #writer.add_graph(gnet, dumm2_input

    # Training Loop
    # Lists to keep track of progress
    '''完全可以不用列表，但是为了以后可能有其他用，就保留了'''
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    fixed_noise = torch.randn(opt.batch_size, opt.nd, 1, 1, device=device)
    print("Starting Training Loop...")
    dnet.train()  
    gnet.train()
    # 不写也默认为train模式；当有BN层和dropout层时，肯定得考虑模式切换，因为训练时这两个层有变化，验证时不能让它变，而eval()模式就不变，train()会变
    # For each epoch
    for epoch in range(1, opt.max_epoch + 1):
        # For each batch in the dataloader
        print(len(dataloader))
        print(type(dataloader))
        for i, (imgs, _) in enumerate(dataloader, 1):
            # torch.utils.data.DataLoader()返回的就是二元组组成的一个特殊的对象(不是列表等，也不能切片);
            # 在MNIST数据集中img, label = data；这些动漫头像没有标签，打印出来后发现是tensor([0, 0,...0, 0 0])
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            ## Train with all-real batch
            dnet.zero_grad()
            '''先训练判别器，再训练生成器'''
            # Format batch
            real_img = imgs.to(device)  # 每个batch.to(device)
            # torch.full((2,3), 1.2),第一个参数必须是元组，可以是任意维数，但想要一维填充时也得为元组，而元组只有一个元素时后面必须有个,
            label = torch.full((opt.batch_size, ), opt.real_label, device=device)
            # Forward pass real batch through D
            output = dnet(real_img)  # 在model模块中已经被展成一维的啦
            # Calculate loss on all-real batch
            d_err_real = criterion(output, label)  # 平均损失
            # Calculate gradients for D in backward pass
            d_err_real.backward()
            D_x = output.mean().item()  # 真实图片的平均得分，当然是接近1越好

            ## Train with all-fake batch
            # Generate batch of latent vectors  latent:隐藏的，潜伏的
            noise = torch.randn(opt.batch_size, opt.nd, 1, 1, device=device)
            # gnet会生成opt.batch_size个图像，因为一个(opt.nd,1,1)可以生成一个图像；在gnet中，每张图有otp.nd个feature maps
            # ,每个feature map大小为1 x 1,所以每个opt.nd(也就是一个值)，控制着生成图像中的一个特征
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
            d_err = d_err_real + d_err_fake  #tensor(1.272)+tensor(0.183)可以直接相加，不需要先取出数值。
            # tensor自成体系，tensor和tensor的加减乘除和标量一模一样；只是tensor和标量之间不能直接算
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
                with torch.no_grad():  # 上下文管理，处于with范围内的tensor待会不反向，所以前向时不用求局部梯度了，节省计算。因为forward时就会把每层对应局部梯度公式求出来
                    fake = gnet(fixed_noise)#.detach().cpu() 截断再放CPU里没什么特殊用啊，有没有效果一样，只是拷贝一份假图片存放cpu里
                img_list.append(vutils.make_grid(fake, normalize=True))
                '''还不知道合成的图有多少个小图呢'''
                writer.add_image('fake%d'%(iters/500), img_list[int(iters/500)], int(iters/500))

            iters += 1
    
    torch.save(dnet.state_dict(), 'dnet.pth')
    torch.save(gnet.state_dict(), 'gnet.pth')

    writer.close()
    '''
    for i in range(iters):
        writer.add_scalars('dnet_gnet_loss', {'G_losses': G_losses[i], 'D_losses': D_losses[i]}, i)
    print('img_list的长度为: %d'%len(img_list))
    for i in range(len(img_list)):
        writer.add_image('fake%d'%i, img_list[i], i)  # 如果名字一样，新图会覆盖旧图
    '''
    #writer.add_graph(dnet, input_to_model=(torch.rand(opt.batch_size, 3, 96, 96), ), verbose=False)
    #writer.add_graph(gnet, input_to_model=(torch.rand(opt.batch_size, op.nd, 1, 1), ), verbose=False)
    # 第二个参数是这个网络的输入数据的shape,也就是(torch.rand(B,C,H,W),torch.rand(B,C,H,W)),第二个rand可以改成,
