import torch.nn as nn
import torch
import math
from collections import OrderedDict


model_urls = {
                'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            }

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

'''这个东西看的有毛病，说明自己对残差网络还没理解到位'''
# get BasicBlock which layers < 50(18, 34)
class BasicBlock(nn.Module):
    expansion = 1
     # 实验证明通过 实例名.exoansion也能调用这个变量；所以如果变量和具体使用地方(实例怎么用)无关的话，可以把变量定义在函数外
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        # 就这一行和下下下行用到了conv3x3，还定义了个函数，这给矫情的
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.downsample = downsample
        self.relu= nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4  
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m  = OrderedDict()
        # 师兄在源码基础上增加了'relu1'和'relu2'，并且把三个卷积层的False改成了True
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.downsample = downsample
        self.relu= nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out

'''
import torch.nn as nn
class Bottleneck(nn.Module):
    expansion = 4.
    def __init__(self):
        self.z = 2.
        super(Bottleneck, self).__init__()
        self.x = 2.
    k = 3
y = Bottleneck()
print(y.expansion, y.z, y.x, y.k)
这种情况完全ok。变量位置在函数外面时候不用弄成self.变量名,就可以 实例名.变量名 引用；位置在函数里面(init或者其他函数都一样)时，必须是
self.变量名才可以引用。init函数接受到的传递过来的参数，必须在init函数体里转化为self.才能在其他函数调用；其他函数使用的参数必须是自己
所处位置上面的self.或者自己接受的参数。
'''

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        '''和nn.Module有关的语句在super(ResNet,self).__init__()下面就行，至于上面有是什么语句不用管，都可以的'''
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)
        '''
        self.group1 = nn.Sequential()；然后self.group1.add_module(name, layer)也可以，或者ModuleList(),ModuleDict()都行
        '''
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        '''
        完整的resnet模型应该还有这两层，但是我们只需要利用提取特征的部分
        self.average_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        '''
        # 对卷积层与BN层初始化，
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 2.就是2.0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  #这一层的输出层数*扩张系数=下一层的输入层数
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
        '''比较上面的self.group1 = nn.Sequential(m)会发现，如果是列表就需要加这个*号，字典不用'''

    def forward(self, x):
        x = self.group1(x)

        fea1 = self.layer1(x)
        fea2 = self.layer2(fea1)
        fea3 = self.layer3(fea2)
        fea4 = self.layer4(fea3)

        return fea2, fea3, fea4


def load_state_dict(model, model_root):
    from torch.utils import model_zoo
    from torch import nn
    import re  # 正则表达式
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = torch.load(model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            if 'fc' in name:
                continue
            print(own_state.keys())  # 返回字典中的键
            raise KeyError('unexpected key "%s" in state_dict'
                           %(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

'''上面所有东西都是被它调用的，并向HQPMR_model返回model'''
def resnet34(pretrained=False, model_root=None):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        load_state_dict(model, model_root=model_root)
    return model
