from utils.utils import progress_bar


def train(epoch, model, criterion, optimizer, trainloader, device):
    model.train()
    print('---------------------------------------------Epoch: %d--------------------------------------------------'%epoch)
    total_loss_train = 0  # 用于tensorboard
    # python中的两种标准化输出方式
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data, device)  # data作为参数传递给了model实例中的forward了，并且forward会自动执行
        # 这一句可以理解为output = forward(data)
        loss = criterion(output, target)  # loss已经是这一个batch的总体损失啦，没有必要loss.item()*batch_size
        total_loss_train += loss.item()
        '''output是二维的，target是一维的这都能行？得打印出来看看。因为NLLoss自带softmax，所以这里没有用
        logsoftmax也会把output这个二维的转化成一维的'''
        loss.backward()
        optimizer.step()
        '''一个batch就正反方向一次'''
        progress_bar(batch_idx, len(trainloader), 'loss: ' + str('%.4f'%loss.item()) + ' | train')
        '''
        不是loss.item()就行了吗？.data干嘛？这是因为loss.item()只适用含有一个数值的tensor,而.data.item()
        ，最好是.detach().item()什么情况都适用
        '''
         
    # return '%.4f' % (total_loss_train / len(trainloader.dataset))
    return total_loss_train / len(trainloader.dataset)


def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:  # 固定用法，记住记住
        param_group['lr'] = param_group['lr'] * 0.1
        '''
        [{'params': model.sample_128.parameters(), 'lr': lr},
        {'params': model.sample_256.parameters(), 'lr': lr},
        {'params': model.fc_concat.parameters(), 'lr': lr}]就是optimizer.param_groups
        '''