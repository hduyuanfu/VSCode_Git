import torch
import torch.nn as nn
import torch.nn.functional as F

import model.resnet_model as resnet_model


class RNet(nn.Module):
    def __init__(self, in_dim, out_dim, pool_size):
        super(RNet, self).__init__()

        self.PathA = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=pool_size, padding=0, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.PathB = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, stride=pool_size, padding=0, bias=False),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        x = self.PathA(x) + self.PathB(x)
        return x


def mask_Generation(feature, device, alpha):
    batch_size = feature.size(0)
    kernel = feature.size(2)
    sum = torch.sum(feature.detach(), dim=1)
    # 不管怎么样，把计算过程中的features map拿出来做其他事了，还是.detach()安全啊
    # features是B * C * H * W，dim=1指的是C，沿着C求和，就把C维度搞没了，所以sum是 B * H * W
    avg = torch.sum(torch.sum(sum, dim=1), dim=1) / kernel ** 2
    # dim=1代表第二个维度，也就是把H搞没了，变成C * W；此时W就成了第二个维度，所以又是一个dim=1，变成了B
    mask = torch.where(sum > alpha * avg.view(batch_size, 1, 1), torch.ones(sum.size()).to(device),
                       (torch.zeros(sum.size()) + 0.1).to(device))
                       # 三个输入参数，第一个是判断条件，第二个是符合条件的设置值，第三个是不满足条件的设置值
    
    mask = mask.unsqueeze(1)
    # mask一会儿还要和其他features map交互，所以它不能比别人少个维度啊，得加一个虚拟的通道维度
    return mask

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sample_128 = RNet(in_dim=128, out_dim=512, pool_size=4)
        self.sample_256 = RNet(in_dim=256, out_dim=512, pool_size=2)

        self.fc_concat = torch.nn.Linear(512 ** 2 * 3, 200)
        self.pool_kernel4 = nn.MaxPool2d(kernel_size=4)
        self.pool_kernel2 = nn.MaxPool2d(kernel_size=2)

        self.softmax = nn.LogSoftmax(dim=1)
        self.features = resnet_model.resnet34(pretrained=True,
                                              model_root='/data/guijun/HBP_finegrained/pth/resnet34.pth')


    def extract_feature(self, feature1, feature2, feature3, batch_size):

        feature1 = self.sample_128(feature1)
        feature2 = self.sample_256(feature2)

        inter1 = (feature1 * feature2).view(batch_size, 512, 14 ** 2)
        inter2 = (feature3 * feature1).view(batch_size, 512, 14 ** 2)
        inter3 = (feature3 * feature2).view(batch_size, 512, 14 ** 2)

        inter1 = (torch.bmm(inter1, torch.transpose(inter1, 1, 2) / 14 ** 2)).view(batch_size, -1)
        inter2 = (torch.bmm(inter2, torch.transpose(inter2, 1, 2) / 14 ** 2)).view(batch_size, -1)
        inter3 = (torch.bmm(inter3, torch.transpose(inter3, 1, 2) / 14 ** 2)).view(batch_size, -1)

        result1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        result2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        result3 = torch.nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))

        
        return result1, result2, result3

    def forward(self, x, device):

        batch_size = x.size(0)
        feature1, feature2, feature3 = self.features(x)  
        # 每张图求一个聚合掩码，而不是一个batch，每个batch里很多张不同图片，怎么可能生成一张共同的聚合掩码嘛

        slack_mask1 = mask_Generation(feature1, device, alpha=0.8)
        slack_mask2 = mask_Generation(feature2, device, alpha=0.7)
        slack_mask3 = mask_Generation(feature3, device, alpha=0.6)

        Aggregated_mask = self.pool_kernel4(slack_mask1) * self.pool_kernel2(slack_mask2) * slack_mask3
        Aggregated_mask_feature1 = F.interpolate(Aggregated_mask, scale_factor=4, mode='bilinear', align_corners=True)
        Aggregated_mask_feature2 = F.interpolate(Aggregated_mask, scale_factor=2, mode='bilinear', align_corners=True)


        feature1 = feature1 * Aggregated_mask_feature1
        feature2 = feature2 * Aggregated_mask_feature2
        feature3 = feature3 * Aggregated_mask

        result1, result2, result3 = self.extract_feature(feature1, feature2, feature3, batch_size)

        result = torch.cat((result1, result2, result3), 1)
        result = self.fc_concat(result)

        return self.softmax(result)  # 返回结果为对数打分的二维张量(N, num_classes)，元素为得分
