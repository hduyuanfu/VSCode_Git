import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

transform_train = transforms.Compose([
                                    transforms.Resize((500, 480), Image.BILINEAR),
                                    transforms.RandomHorizontalFlip(),  #可以后面再接一个竖直翻转效果(概率自己设定)
                                    # 水平翻转就像水中倒影一样，上下折叠能重合；
                                    # 竖直翻转是左右折叠能重合
                                    transforms.RandomCrop(448),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ])
'''可以看出来，训练时故意加大了各种随机性，想着法子增大辨别难度，以使程序更健壮'''
transform_test = transforms.Compose([
                                    transforms.Resize((500, 480), Image.BILINEAR),
                                    transforms.CenterCrop(448),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ])

def default_loader(path):
    return Image.open(path).convert('RGB')

'''下面这个改造自己的数据读取类是个难点'''
class MyDataset(Dataset):
    def __init__(self, txt, transform, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader('/data/guijun/aircraft/images/' + fn)
        # print(fn)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)