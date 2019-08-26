import torch
from torchvision import datasets, transforms


def data_loader(opt):
    transform = transforms.Compose([    # 评估不需要加载真实图片
                                        transforms.Resize(opt.image_size), #重新设置图片大小，opt.image_size默认值为96
                                        transforms.CenterCrop(opt.image_size), #从中心截取大小为opt.image_size的图片
                                        transforms.ToTensor(), #转为Tensor格式，并将值取在[0,1]中
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #标准化，得到在[-1,1]的值
                                    ])
    dataset = datasets.ImageFolder(opt.data_path, transform=transform) #从data中读取图片，图片类别会设置为文件夹名faces
    dataloader = torch.utils.data.DataLoader(dataset, #然后对得到的图片进行批处理，默认一批为256张图，使用4个进程读取数据
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                num_workers=opt.num_workers,
                                                drop_last=True
                                                )
    # dataloader只是获得一个索引，并没有真正导入图片数据，真正导入图片是在for循环那里一个batch一个batch的导入
    print('%d images were found there!' % len(dataloader))

    #a = next(iter(dataloader))
    #print(a[1])
    return dataloader