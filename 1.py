from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='avg_loss') 
'''我想把log_dir='绝对路径'就各种报错'''
for epoch in range(1, 81):
    avg_loss_train = epoch
    writer.add_scalar('avg_train_loss_per_epoch',  avg_loss_train, epoch)
writer.close()