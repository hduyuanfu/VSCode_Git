#from __future__ import print_function  # 这行必须是第一行
#Python提供的__future__模块，把下一个新版本特性导入到当前版本，于是就可以在当前版本测试新版本的一些特性。
#比如上面的使用print_function后，相当于即使是在2.x中运行python，但是也需要维持3.x的print特性，即添加括号
'''
先本地跑排除字符语法错误，两分钟不报错即可，对不太清楚确定的再写一个小实验文件，进行多种实验；修改大文件后先跑5-10epoch,
tensorboard查看结果排除逻辑无错后再多次迭代跑。

.cuda()是老用法，不要用了
# 开始脚本，创建一个张量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 但是无论你获得一个新的Tensor或者Module
# 如果他们已经在目标设备上则不会执行复制操作
input = data.to(device)
model = MyModule(...).to(device)
to(device)代表移动(针对模型)或拷贝(针对变量)，从cpu到gpu,从gpu到cpu，从这个gpu到另一个几个gpu，都可以
用法(1):
可以先import os,再os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2" ,
然后device = torch.device("cuda: 0, 1, 2" if torch.cuda.is_available() else "cpu")
用法(2):直接抛弃os那两行，只用device也行；
用法(3):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")这句命令只说明用gpu但不指定卡，用os那两行
指定卡；或者不用os那两行，直接不在程序中指定卡号，而是在终端执行程序时指定GPU，CUDA_VISIBLE_DEVICES=0 python  your_file.py
'0';'0,1,3';0,1,3总之带不带引号都一样
'''

'''
(1)模型，外来以及自己生成的参与运算比较的变量(数据)(e.g. batch_data,label,troch.randn(),torch.ones(),torch.zeros(),torch.full())
等都要放gpu里，而由模型和数据等已经在GPU里的东西计算生成的各种对象以及这些对象再次运算求得的东西都已经gpu里了，所以不需
要放进去；由于参与损失函数的两个对象都是位于GPU中的tensor，计算损失值时，数据会把这个函数带到GPU中计算；同理优化器中因
为优化的是GPU中模型的参数，所以bp求解时数据会把算法带进来，在GPU完成计算过程，所以损失函数和优化器都不需要放进去，会被自动带进去。
(2)模型实例化，实例.to(device),模型分发。其他数据、损失函数、优化器都不动就可以进行多卡计算
(3)模型实例化，实例.to(device)[,模型分发]。最好放在损失函数、优化器之前，这样肯定不出错
(4)top,nvidia-smi查看在跑的进程的ID，之后可以kill
(5)torch.backends.cudnn.benchmark = True;在模型实例化前写上这句话就成。设置这个 flag 可以让内置的 cuDNN 的
auto-tuner 自动寻找最适合当前配置的高效算法,来达到优化运行效率的问题。因为这句话没额外开销，有益无害，一般都会加。
准则：如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
(6)label = torch.full((d1,d2,···dn),填充值1，device=device),label.fill_(填充值2)。torch.randn和torch.rand与torch.ones/zeros，
他们加减乘除配合产生我们想要的多种结果。
torch.randn(*sizes, out=None) → Tensor
返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义。
torch.rand(*sizes, out=None) → Tensor
返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。
(7.1)训练时不论是一个卡或者多个卡跑，生成时不论一个或多个卡，用这句话：
torch.save(dnet, 'dnet.pth')
dnet = torch.load('dnet.pth').to(device)
(7.2)训练是单卡时，测试时单卡时：
torch.save(model.state_dict(), 'firststep_aircarft.pth')
pre_dic = torch.load('firststep_aircarft.pth')
model.load_state_dict(pre_dic)
单卡-单卡时有一种王涛的方法：
def save_model_dict(model, optimizer, path):
    checkpoint = {'model':model.state_dict(), 'optimizer':optimizer.state_dict()}
    torch.save(checkpoint, path)
def load_model_dict(model, path):  # model是刚实例化的空壳网络
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoit['model'])
    model = model.to(device)  # 或者直接model.to(device)也行
    optimizer = optim.Adam(model.parameters(), lr = 2e-4, betas = (0.5, 0.999))
    optimizer.load_state_dict(checkpoint['optimizer'])  # 优化器也有参数的吗？？？
    return model, optimizer
(7.3)训练是多卡，测试是一卡时：
# original saved file with DataParallel
state_dict = torch.load('myfile.pth')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.'
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
(7.4)训练是多卡，测试是多卡是(训练与测试最好是同样的卡组合；数量一样卡号组合不一样不知道行不行)，还用训练时的分卡语句：
model = torch.nn.DataParallel(model,[])
# cudnn.benchmark = True
pre_dic = torch.load('firststep_aircarft.pth')
model.load_state_dict(pre_dic)
(8)return '%s (- %s)' % (asMinutes(s), asMinutes(rs));return '%dm %ds' % (m, s)
writer.add_image('fake%d'%(iters/500), img_list[int(iters/500)], int(iters/500));也就是说'%d'%()这个结构哪里都可以用，
不一定在print()里
(9)pip install screen
screen -S name  创建
screen -r name  进入
screen -ls  查看
ctrl+a+d  返回主页
screen -X -S name(or ID) quit  杀死某屏幕及其任务
某个screen运行完只剩print没有输出但生成的文件都有了的时候卡住啦，ctrl+c同时使劲多次，总能打断重来的
(10)先建立监听隧道，文件-当前会话属性-连接-隧道，自己是监听端口，服务器是目标端口，双方都是localhost不用变，目标端口得是默认的6006端口，
本地随意，但为了好记，也6006；在screen中跑程序，然后在主页中tensorboard --logdir=name(端口默认6006)，在本地浏览器就可以实时监控了
在Xftp上把服务器文件删了，很难恢复，所以悠着点。
运行新的events文件时可以google页面大刷新；边跑边观察，用页面内右上角小刷新。
Xshell,Xftp上有长得一样的重连接按钮；并且可以相互打开，Xftp上点一下就打开了对应Xshell,反之亦然。
pip install tb-nightly
结果文件在服务器上时：
先cd进入所在目录，在tensorboard --logdir=name(端口默认6006)，然后在本地Google浏览器localhost:6006
结果文件拿回本地时：
在cmd中或者anaconda prompt中先'F:'进入文件所在盘符，然后tensorboard --logdir=name(端口默认6006)，端口被占用或有它用时，
tensorboard --logdir=name --port 1234(举例而已，可以随意)，然后本地浏览器localhost:1234。
注：逻辑意义上的端口，一般是指TCP/IP协议中的端口，端口号的范围从0到65535，比如用于浏览网页服务。系统很多网络功能、还有许多软件
都需要在网络上联系，为了不冲突混淆，就分别使用不同端口。默认下不是全部打开的一般有相关软件运行才会打开相应的端口。
(11)top 或nvidia-smi查看进程PID或名称；之后kill PID或killall name(不用-PID，也就是不用杠)；而且自己也没权限kill掉别人进程
(12)step1:判断真图片，dnet反向传播(先让判别器学下真图片应该什么样，然后才能去下一行判别假不假，假的程度)
          生成假图片，(生假和判假截断梯度)dnet判别假图片，dnet反向传播(上行已经学了真图片应该什么样，这行学习假图片一般是什么样子，真假图片的样子都学到了的判别器才能去判别真伪啊)
    step2:用两次反向过的dnet再次判断step1中的(假图片,1)，求得gnet损失，gnet反向传播
step1训练判别器；step2训练生成器
在step1中：是训练判别器，此时，真图片应该给他高分，假图片应该给他低分；所以loss_real=criterion(真图片，1),反向传播loss_real使它变小，越小dnet越能识别真图片；
          loss_fake=(假图片，0)，反向传播loss_fake使它变小，越小dnet越能识别假图片；这样两次反向传播后，判别器对真假图片分辨能力都会有次提升。
在step2中：是训练生成器，此时使用同样假图片，loss=criterion(假图片，1),反向使loss变小，那么新参数也会使假图片得分更接近1(只有得分更接近1时loss才会小啊)，往后再生成的假图也会更逼真
(13)一维数组(or tensor)中每个值代表一个特征，具体代表那哪个特征，每个值与其对应特征的特性是什么关系，可以由auto_encoder实现
(14)卷积一次，池化一次，等等操作；虽然可能提起特征信息，高度概括化信息等，但随着每一步操作都可能会丢失一些信息的；怎么才能少丢失或者找回？比如残差相加是一种方法
'''
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

from train import train
from generate import generate

device_train = torch.device("cuda: 0, 1, 2" if torch.cuda.is_available() else "cpu")  # 有cuda这个架构就使用0卡，无则cpu;device(可以指定任何设备，cpu,哪块或哪几块显卡)
device_generate = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# 训练可能多卡，预测一张就够了，所以有点小不同
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"  # 选择显卡

class Config(object):
    how = 'train'  # 'train'决定训练，'get'决定验证
    data_path = '/data/yuanfu/GAN/Data'  # 数据集存放路径,这三种都不能写/Data/,/Data/faces,/Data/faces/,
    num_workers = 0  # 多进程加载数据所用的进程数,0为主进程
    image_size = 96  # 图片尺寸
    batch_size = 64
    max_epoch = 100
    real_label = 1
    fake_label = 0
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    nd = 100  # 噪声维度
    gfm = 64  # 生成器feature map数
    dfm = 64  # 判别器feature map数
    

if __name__ == "__main__":
    opt = Config()
    if opt.how == 'train':
        train(opt, device_train)
    elif opt.how == 'get':
        generate(opt, device_generate)
        

    