import torch.nn as nn
from torchvision import models


def weight_init(model):
    classname = model.__class__.__name__
    print('Initializing Parameters of %s !' % classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    print('finshed parameters initalizations!')

    
class DNet(nn.Module):
    """
    判别器定义
    """
    def __init__(self, opt):
        super(DNet, self).__init__()
        dfm = opt.dfm  #判别器channel值
        self.main = nn.Sequential(
            # 输入 3 x 96 x 96
            # kernel_size = 5,stride = 3, padding =1
            # 按式子计算 (96 + 2*1 - 5)/3 + 1 = 32
            # 是same卷积，96/32 = stride = 3
            # Conv2d(in_channels, out_channels, kernel_size, 
            # stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d(in_channels=3, out_channels=dfm, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # 控制负坡度的角度什么鬼？
            # 输出 (dfm) x 32 x 32

            #kernel_size = 4,stride = 2, padding =1
            #按式子计算 (32 + 2*1 - 4)/2 + 1 = 16
            #是same卷积，32/16 = stride = 2
            nn.Conv2d(dfm, dfm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (dfm*2) x 16 x 16

            #kernel_size = 4,stride = 2, padding =1
            #按式子计算 (16 + 2*1 - 4)/2 + 1 = 8
            #是same卷积，16/8 = stride = 2           
            nn.Conv2d(dfm * 2, dfm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (dfm*4) x 8 x 8

            #kernel_size = 4,stride = 2, padding =1
            #按式子计算 (8 + 2*1 - 4)/2 + 1 = 4
            #是same卷积，8/4 = stride = 2
            nn.Conv2d(dfm * 4, dfm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (dfm*8) x 4 x 4

            #kernel_size = 4,stride = 1, padding =0
            #按式子计算 (4 + 2*0 - 4)/1 + 1 = 1
            nn.Conv2d(dfm * 8, 1, 4, 1, 0, bias=False),
            #输出为1*1*1
            nn.Sigmoid()  # 返回[0,1]的值，输出一个数(作为概率值)
        )

    def forward(self, input):
        return self.main(input).view(-1) #输出从1*1*1变为1，得到生成器生成假图片的分数，分数高则像真图片


class GNet(nn.Module):
    """
    生成器定义
    """
    
    def __init__(self, opt):
        super(GNet, self).__init__()
        gfm = opt.gfm  # 生成器feature map数channnel，默认为64

        self.main = nn.Sequential(
            # 输入 opt.batch_size x opt.nd x 1 x 1 
            # 输入是一个nd维度(默认为100)的噪声，我们可以认为它是一个1*1*nd的feature map(因为是torch.randn()所有值可正可负)
            # kernel_size = 4,stride = 1, padding =0, outputpadding默认为0, in/output为输入/出的尺寸
            # 上采样输入输出尺寸公式：output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
            # 根据计算式子 (1-1)*1 + 0 - 2*0 + 4 = 4
            # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, 
            # padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            nn.ConvTranspose2d(in_channels=opt.nd, out_channels=gfm * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(gfm * 8),
            nn.ReLU(True),
            # 上一步的输出形状：(gfm*8) x 4 x 4
    
            #kernel_size = 4,stride = 2, padding =1
            #根据计算式子 (4-1)*2 - 2*1 + 4 = 8
            nn.ConvTranspose2d(gfm * 8, gfm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfm * 4),
            nn.ReLU(True),
            # 上一步的输出形状： (gfm*4) x 8 x 8
    
            #kernel_size = 4,stride = 2, padding =1
            #根据计算式子 (8-1)*2 - 2*1 + 4 = 16
            nn.ConvTranspose2d(gfm * 4, gfm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfm * 2),
            nn.ReLU(True),
            # 上一步的输出形状： (gfm*2) x 16 x 16
    
            #kernel_size = 4,stride = 2, padding =1
            #根据计算式子 (16-1)*2 - 2*1 + 4 = 32
            nn.ConvTranspose2d(gfm * 2, gfm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfm),
            nn.ReLU(True),
            # 上一步的输出形状：(gfm) x 32 x 32
    
            # kernel_size = 5,stride = 3, padding =1
            #根据计算式子 (32-1)*3 - 2*1 + 5 = 96
            nn.ConvTranspose2d(gfm, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )
    
    def forward(self, input):
         return self.main(input)