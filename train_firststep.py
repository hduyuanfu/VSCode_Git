# encoding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
#from PIL import Image
#import model.HQPMR_model as HQPMR
from data.prepareddata import transform_train, transform_test, MyDataset
from model.HQPMR_model import Net
from utils.train import train, adjust_learning_rate
from utils.test import test

import torch.nn.parallel 
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
#w * w后还是个矩阵，只是对应位置元素相乘
class Config(object):
    train_path = '/data/yuanfu/MyHQPMR/data/trainlist_shuffle.txt'
    test_path = '/data/yuanfu/MyHQPMR/data/testlist.txt'
    batch_size_train = 8
    shuffle_train = True
    num_workers_train = 4
    batch_size_test = 8
    shuffle_test = False
    num_workers_test = 4
    lr = 1.0

opt = Config()

trainset = MyDataset(opt.train_path, transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size_train,
                                          shuffle=opt.shuffle_train, num_workers=opt.num_workers_train)

testset = MyDataset(opt.test_path, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size_test,
                                         shuffle=opt.shuffle_test, num_workers=opt.num_workers_test)


model = Net().to(device)
model.features.requires_grad = False
cudnn.benchmark = True
#model = model.to(device)
optimizer = optim.SGD([
                        {'params': model.sample_128.parameters(), 'lr': opt.lr},
                        {'params': model.sample_256.parameters(), 'lr': opt.lr},
                        #  这样可以在不同的层使用不同的学习率，实现差异化
                        {'params': model.fc_concat.parameters(), 'lr': opt.lr},
                        ], lr=1, momentum=0.9, weight_decay=1e-5)
#model = nn.DataParallel(model, device_ids=[0,1,2])

#model.features.requires_grad = False
'''待会打印出来模型看看有没有这个features'''
criterion = nn.NLLLoss()
'''
optimizer = optim.SGD([
                        {'params': model.sample_128.parameters(), 'lr': opt.lr},
                        {'params': model.sample_256.parameters(), 'lr': opt.lr},
                        #  这样可以在不同的层使用不同的学习率，实现差异化
                        {'params': model.fc_concat.parameters(), 'lr': opt.lr},
                        ], lr=1, momentum=0.9, weight_decay=1e-5)
'''
writer = SummaryWriter(log_dir='avg_loss') 
'''我想把log_dir='绝对路径'就各种报错'''
for epoch in range(1, 81):
    avg_train_loss = train(epoch, model, criterion, optimizer, trainloader, device)
    writer.add_scalar('avg_train_loss_per_epoch', avg_train_loss, epoch)
    if epoch % 5 == 0:
        avg_test_loss, accuracy = test(model, criterion, testloader, device)
        #print(test(model, criterion, testloader, device))
        writer.add_scalar('avg_test_loss',  avg_test_loss, int(epoch/5))
        writer.add_scalar('test_accuracy', accuracy, int(epoch/5))
    if epoch % 40 == 0:
        adjust_learning_rate(optimizer)
writer.close()

torch.save(model.state_dict(), 'firststep_aircarft.pth')
