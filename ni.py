from torch.utils.tensorboard import SummaryWriter
import numpy as np
'''
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
writer = SummaryWriter()
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
writer.close()
'''
import torch.full