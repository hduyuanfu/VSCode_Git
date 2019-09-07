# encoding:utf-8
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import model.HQPMR_model as HQPMR  # 导入自定义模型
import data.data as data
import os
import torch.backends.cudnn as cudnn  # 听说可以加快运算
from PIL import Image
from utils.utils import progress_bar  # 从实用工具中导入进度条
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择显卡

trainset = data.MyDataset('/data/guijun/aircraft/trainlist_shuffle.txt', transform=transforms.Compose([
                            transforms.Resize((500, 480), Image.BILINEAR),  # resize时需要的变换方法为双线性
                            transforms.RandomHorizontalFlip(),  # 水平/垂直翻转(上下/左右折叠会重合) 

                            transforms.RandomCrop(448),  # 随机裁剪(h,w)或(size,size)
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                            ]))  # 为什么我记得mean和std必须为[]呢？
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=4)  # 四个线程

testset = data.MyDataset('/data/guijun/aircraft/testlist.txt', transform=transforms.Compose([
                            transforms.Resize((500, 480), Image.BILINEAR),
                            transforms.CenterCrop(448),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                            ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=False, num_workers=4)
cudnn.benchmark = True



model = HQPMR.Net()

model.cuda()


pretrained = True
if pretrained:

    # pre_dic = torch.load('aircraft92.64.pth')

    pre_dic = torch.load('firststep_aircarft.pth')

    model.load_state_dict(pre_dic)

'''
加载模型：
(1)加载完整的模型结构和参数信息,使用load_model  = torch.load('model.pth'), 在网络比较大的时候加载时间比较长,
同样存储空间也比较大;
(2)加载模型的参数信息,需要先导入模型的结构,然后通过 model.load_state_dic(torch.load('model_state.pth')) 来导入。
保存模型：
torch.save(net1, "net.pkl")  # entire net  整个神经网络保存
torch.save(net1.state_dict(), 'net_params.pkl')  # parameters 保存神经网络中的参数

'''
criterion = nn.NLLLoss()
'''
损失函数NLLLoss() 的 输入 是一个对数概率向量和一个目标标签. 
它不会为我们计算对数概率，适合最后一层是log_softmax()的网络. 
损失函数 CrossEntropyLoss() 与 NLLLoss() 类似, 唯一的不同是它为我们去做 softmax.
可以理解为：CrossEntropyLoss()=log_softmax() + NLLLoss()
'''
lr = 1e-1



# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
optimizer = optim.SGD([
                        {'params': model.features.parameters(), 'lr': 0.1 * lr},
                        {'params': model.sample_128.parameters(), 'lr': lr},
                        {'params': model.sample_256.parameters(), 'lr': lr},
                        {'params': model.fc_concat.parameters(), 'lr': lr},
], lr=1e-1, momentum=0.9, weight_decay=1e-5)


def train(epoch):
    model.train()
    print('----------------------------------------Epoch: {}----------------------------------------'.format(epoch))
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), target.cuda()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader), 'loss:'+ str('{:.4f}'.format(loss.data.item())) + ' | train')



def test():
    model.eval()
    print('----------------------------------------Test---------------------------------------------')
    test_loss = 0
    correct = 0
    for batch_idx,(data, target) in enumerate(testloader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        progress_bar(batch_idx, len(testloader), 'test')
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 8., correct, len(testloader.dataset),
        100.0 * float(correct) / len(testloader.dataset)))

def adjust_learning_rate(optimizer, epoch):
    if epoch % 40 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

for epoch in range(1, 161):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    if epoch % 5 == 0:
        test()
    elif epoch > 60:
        test()

torch.save(model.state_dict(), 'aircraft_result.pth')