import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms
from PIL import Image
from torch.optim import SGD, Adam
import torch.nn.functional as F

from networks.vgg import VGG, VGG_gray


def def_args():
    parser = argparse.ArgumentParser(description='粗糙度检测')
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--img_shape', default=(256, 256), help='vgg默认的输入形状为256*256，如需更改输入图像尺寸，需对vgg网络结构进行调整')
    parser.add_argument('--training_dataset_ratio', default=0.8, help='数据集中训练集所占比例，测试集则为剩余部分')
    parser.add_argument('--file_path', default='data/data', help='数据集路径，下一级为图像')
    parser.add_argument('--dropout', default=0., help='当模型过拟合时，提高dropout，一般不超过0.5')
    parser.add_argument('--weight_decay', default=0., help='当模型过拟合时，提高weight_decay，一般不超过0.5')
    parser.add_argument('--lr', default=1e-3,
                        help='学习率lr与batch_size的大小成正比，即当学习率已经能使得模型训练到一个稳定的精度时，改变batch_size时只需遵循'
                             '线性缩放原则改变lr。观察训练曲线，当训练曲线不稳定时，降低学习率')
    parser.add_argument('--batch_size', default=16, help='如何报告电脑显存不够时，降低batchsize')
    parser.add_argument('--epochs', default=100)
    return parser.parse_args()


def get_img_paths(file_path):
    sub_paths = os.listdir(file_path)
    paths = []
    for sub_path in sub_paths:
        path = os.path.join(file_path, sub_path)
        paths.append(path)
    random.shuffle(paths)
    return paths


class my_dataset(Dataset):
    def __init__(self, img_paths):
        super(my_dataset, self).__init__()
        self.paths = img_paths
        self.resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Resize(args.img_shape)
        ])
        self.flip = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])


class my_train_dataset(my_dataset):
    def __init__(self, img_paths):
        super(my_train_dataset, self).__init__(img_paths=img_paths)
        n = len(self.paths)
        self.paths = self.paths[:int(n * args.training_dataset_ratio)]

    def __getitem__(self, item):
        path = self.paths[item]
        img = Image.open(path).convert('L')
        img = self.resize(img)
        img = self.flip(img)
        sub_path = os.path.split(path)[-1]
        label = np.float32(float(sub_path[:-4]))
        return img, label

    def __len__(self):
        return len(self.paths)


class my_test_dataset(my_dataset):
    def __init__(self, img_paths):
        super(my_test_dataset, self).__init__(img_paths=img_paths)
        n = len(self.paths)
        self.paths = self.paths[int(n * args.training_dataset_ratio):]

    def __getitem__(self, item):
        path = self.paths[item]
        img = Image.open(path).convert('L')
        img = self.resize(img)
        sub_path = os.path.split(path)[-1]
        label = np.float32(float(sub_path[:-4]))
        return img, label

    def __len__(self):
        return len(self.paths)


def test(net, loader, epoch):
    net.eval()
    test_loss = 0
    for x, label in loader:
        x, label = x.to(args.device), label.to(args.device)
        with torch.no_grad():
            y = net(x).reshape([-1])
            # loss = F.l1_loss(y, label, reduction='mean')
            loss = F.mse_loss(y, label, reduction='mean')
        test_loss += loss.item() * x.shape[0]
    test_loss /= len(loader.dataset)
    # test_loss = np.sqrt(test_loss)
    return test_loss


def train(net, optim, loader, epoch):
    net.eval()
    train_loss = 0
    for x, label in loader:
        x, label = x.to(args.device), label.to(args.device)
        y = net(x).reshape([-1])
        # loss = F.l1_loss(y, label, reduction='mean')
        loss = F.mse_loss(y, label, reduction='mean')
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item() * x.shape[0]
    train_loss /= len(loader.dataset)
    return train_loss


def main():
    # 初始化网络
    net = VGG_gray(num_classes=1, dropout=args.dropout).to(args.device)
    optim = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 加载预训练模型
    net.load_state_dict(torch.load('networks/parameters/vgg'), strict=False)
    # 制作训练集和测试集
    img_paths = get_img_paths(args.file_path)
    train_dataset = my_train_dataset(img_paths)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 训练集保证随机性
    test_dataset = my_test_dataset(img_paths)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)  # 测试集保证不变性
    # 训练模型
    train_losses, test_losses = [], []
    for epoch in range(args.epochs):
        train_loss = train(net, optim, train_loader, epoch)
        test_loss = test(net, test_loader, epoch)
        print('第%d轮的训练损失为%.5f，测试损失为%.5f，测试集的RMSE损失为%.5f' % (epoch, train_loss, test_loss, np.sqrt(test_loss)))
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.show()


if __name__ == '__main__':
    args = def_args()
    main()
