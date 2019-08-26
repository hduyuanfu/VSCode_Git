import os
import torch
import torch.nn as nn
from torch import optim
import torchvision as tv
from torchvision import models,datasets,transforms
from model import GNet,DNet
from torchnet.mater import AverageValueMeter

import tqdm
import ipdb
#import fire
#import visdom

class Config(object):

    data_path = 'data/'  # 数据集存放路径
    num_workers = 1  # 多进程加载数据所用的进程数
    image_size = 96  # 图片尺寸
    batch_size = 128
    max_epoch = 200
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    gpu = True  # 是否使用GPU
    nd = 100  # 噪声维度
    gfm = 64  # 生成器feature map数
    dfm = 64  # 判别器feature map数

    save_path = 'imgs/'  # 生成图片保存路径

    vis = True  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 20  # 每间隔20 batch，visdom画图一次

    debug_file = '/tmp/debuggan'  # 存在该文件则进入debug模式
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 10  # 每10个epoch保存一次模型
    dnet_path = None  # 'checkpoints/netd_.pth' #预训练模型
    gnet_path = None  # 'checkpoints/netg_211.pth'

    # 只测试不训练
    get_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    get_num = 64
    get_search_num = 512
    noise_mean = 0  # 噪声的均值
    noise_std = 1  # 噪声的方差

opt = Config()  # 配置的实例


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)

    # 数据处理
    transforms = transforms.Compose([
                                    transforms.Resize(opt.image_size), #重新设置图片大小，opt.image_size默认值为96
                                    transforms.CenterCrop(opt.image_size), #从中心截取大小为opt.image_size的图片
                                    transforms.ToTensor(), #转为Tensor格式，并将值取在[0,1]中
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #标准化，得到在[-1,1]的值
                                    ])
    dataset = datasets.ImageFolder(opt.data_path, transform=transforms) #从data中读取图片，图片类别会设置为文件夹名faces
    dataloader = torch.utils.data.DataLoader(dataset, #然后对得到的图片进行批处理，默认一批为256张图，使用4个进程读取数据
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers,
                                            drop_last=True  # 什么鬼
                                            )


    # 网络，gnet为生成器，dnet为判别器
    gnet, dnet = GNet(opt), DNet(opt)
    map_location = lambda storage, loc: storage
        if opt.dnet_path:
            dnet.load_state_dict(torch.load(opt.dnet_path, map_location=map_location))
        if opt.gnet_path:
            gnet.load_state_dict(torch.load(opt.gnet_path, map_location=map_location))
    gnet.cuda()
    dnet.cuda()
    '''
    # 把所有的张量加载到CPU中
    map_location = lambda storage, loc: storage
    # 把所有的张量加载到GPU 1中
    #torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
    #也可以写成：
    #device = torch.device('cpu')
    #netd.load_state_dict(t.load(opt.netd_path, map_location=device))
    #或：
    #netd.load_state_dict(t.load(opt.netd_path))
    #dnet.to(device)

    if opt.dnet_path: #是否指定训练好的预训练模型，加载模型参数
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)
    '''
    # 定义优化器和损失，学习率都默认为2e-4，beta1默认为0.5
    optimizer_g = optim.Adam(gnet.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(dnet.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    '''除了模型和变量，优化器和损失函数也要放GPU里吗,先不用试试'''
    criterion = nn.BCELoss()
    criterion = criterion.cuda()


    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    '''true_labels = torch.ones(opt.batch_size).cuda()  # 这样直接在后面不太好'''
    true_labels = torch.ones(opt.batch_size)
    fake_labels = torch.zeros(opt.batch_size)
    fix_noises = torch.randn(opt.batch_size, opt.nd, 1, 1)#opt.nd为噪声维度，默认为100
    noises = torch.randn(opt.batch_size, opt.nd, 1, 1)

    true_labels = true_labels.cuda()
    fake_labels = fake_labels.cuda()
    fix_noises = fix_noises.cuda()
    noises = noises.cuda()

    #AverageValueMeter测量并返回添加到其中的任何数字集合的平均值和标准差,
    #对度量一组示例的平均损失是有用的。
    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    '''
    1）训练判别器
    先固定生成器
    对于真图片，判别器的输出概率值尽可能接近1
    对于生成器生成的假图片，判别器尽可能输出0
    2）训练生成器
    固定判别器
    生成器生成图片，尽可能使生成的图片让判别器输出为1
    3）返回第一步，循环交替进行
    '''
    '''
    epochs = range(opt.max_epoch)
        for epoch in iter(epochs):
    '''
    for epoch in range(opt.max_epoch)
        for i, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.cuda()
            if i % opt.d_every == 0:
                # 训练判别器
                # 每d_every=1(默认)个batch训练一次判别器
                optimizer_d.zero_grad()
                # 尽可能的把真图片判别为正确
                output = dnet(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                # 尽可能把假图片判别为错误
                # 更新noises中的data值
                noises.data.copy_(torch.randn(opt.batch_size, opt.nd, 1, 1))  # 这是什么意思
                fake_img = gnet(noises).detach()  # 根据噪声生成假图

                output = dnet(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()

                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            if i % opt.g_every == 0:
                # 训练生成器
                # 每g_every=5个batch训练一次生成器
                optimizer_g.zero_grad()
                #更新noises中的data值
                noises.data.copy_(torch.randn(opt.batch_size, opt.nd, 1, 1))
                fake_img = gnet(noises)
                # 生成器要生成以假乱真的假图。所以下面的error_g越小越好
                output = dnet(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()

                errorg_meter.add(error_g.item())
        '''
        注意：
        训练生成器时，无须调整判别器的参数；训练判别器时，无须调整生成器的参数
        在训练判别器时，需要对生成器生成的图片用detach()操作进行计算图截断，避免反向传播将梯度传到生成器中。因为在训练判别器时，我们不需要训练生成器，也就不需要生成器的梯度
        在训练判别器时，需要反向传播两次，一次是希望把真图片判定为1，一次是希望把假图片判定为0.也可以将这两者的数据放到一个batch中，进行一次前向传播和反向传播即可。但是人们发现，分两次的方法更好
        对于假图片，在训练判别器时，我们希望判别器输出为0；而在训练生成器时，我们希望判别器输出为1，这样实现判别器和生成器互相对抗提升
        '''

        '''
        可视化：
        接下来就是一些可视化代码的实现。每次可视化时使用的噪音都是固定的fix_noises，因为这样便于我们比较对于相同的输入，可见生成器生成的图片是如何一步步提升的
        因为对输出的图片进行了归一化处理,值在(-1,1)，所以在输出时需要将其还原会原来的scale,值在(0,1),方法就是图片的值*mean + std
        '''
        # 每间隔20 batch，visdom画图一次
        if opt.vis and i % opt.plot_every == opt.plot_every - 1:
            # 可视化
            # 存在该文件则进入debug模式
            if os.path.exists(opt.debug_file):
                ipdb.set_trace()
            fix_fake_imgs = gnet(fix_noises)
            vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
            vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
            vis.plot('errord', errord_meter.value()[0])
            vis.plot('errorg', errorg_meter.value()[0])

        '''保存模型'''
        # 每10个epoch保存一次模型
        if (epoch+1) % opt.save_every == 0:
            # 保存模型、图片
            tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
                                range=(-1, 1))
            torch.save(dnet.state_dict(), 'checkpoints/dnet_%s.pth' % epoch)
            torch.save(gnet.state_dict(), 'checkpoints/gnet_%s.pth' % epoch)

            errord_meter.reset()#重置，清空里面的值
            errorg_meter.reset()

'''
验证：
使用训练好的模型进行验证
'''
@torch.no_grad()
def generate(**kwargs):#进行验证
    """
    随机生成动漫头像，并根据dnet的分数选择较好的
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    #device=torch.device('cuda') if opt.gpu else torch.device('cpu')

    gnet, dnet = GNet(opt).eval(), DNet(opt).eval()

    noises = torch.randn(opt.get_search_num, opt.nd, 1, 1).normal_(opt.noise_mean, opt.noise_std)
    #noises = noises.to(device)
    noises = noises.cuda()
    
    map_location = lambda storage, loc: storage
    dnet.load_state_dict(torch.load(opt.dnet_path, map_location=map_location))
    gnet.load_state_dict(torch.load(opt.gnet_path, map_location=map_location))
    dnet.cuda()
    gnet.cuda()

    # 生成图片，并计算图片在判别器的分数
    fake_img = gnet(noises)
    scores = dnet(fake_img).detach()

    # 挑选最好的某几张，默认opt.get_num=64张，并得到其索引
    indexs = scores.topk(opt.get_num)[1]  # tokp()返回元组，一个为分数，一个为索引
    result = []
    for i in indexs:
        result.append(fake_img.data[i])
    # 保存图片
    tv.utils.save_image(torch.stack(result), opt.get_img, normalize=True, range=(-1, 1))
    # save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

if __name__=='__main__':
    train()

        