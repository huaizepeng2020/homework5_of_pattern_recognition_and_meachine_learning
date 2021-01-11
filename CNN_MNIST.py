# -*- coding: utf-8 -*-
# coding=utf-8

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import random
import scipy.sparse
import pickle
import torch
import pandas as pd

seed = 20200803
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 训练集文件
train_images_idx3_ubyte_file = './data/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = './data/train-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'  # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(
        image_size) + 'B'  # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    # plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        # print(images[i])
        offset += struct.calcsize(fmt_image)
    #        plt.imshow(images[i],'gray')
    #        plt.pause(0.00001)
    #        plt.show()
    # plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


class CNNN(nn.Module):
    def __init__(self, num_layer, channel, kernal, pad, dim_in, dim_out, dim_out1):
        super(CNNN, self).__init__()
        self.num_layer = num_layer

        self.CNNN = nn.ModuleList()
        for i in range(num_layer):
            self.CNNN.append(nn.Conv2d(channel[i], channel[i + 1], kernal, padding=pad))
            nn.init.xavier_uniform(self.CNNN[-1].weight)
            # self.CNNN.append(nn.ReLU())
            # self.CNNN.append(nn.MaxPool2d(kernel_size=kernal, stride=1, padding=pad))
            self.CNNN.append(nn.AvgPool2d(kernel_size=kernal, stride=1, padding=pad))

        self.out = nn.Linear(channel[i + 1] * dim_in * dim_in, dim_out1, bias=True)
        self.out1 = nn.Linear(dim_out1, dim_out, bias=True)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.xavier_normal_(self.out1.weight)

    def forward(self, h):

        for i in range(len(self.CNNN)):
            h = self.CNNN[i](h)

        out = self.out(h.reshape(len(h), -1))
        out = self.out1(out)

        return out


if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    num_choose = 1000

    train_images = load_train_images()
    a = np.random.randint(0, len(train_images), num_choose)
    train_images = train_images[a]
    train_labels = load_train_labels()
    train_labels = train_labels[a]
    dim_out = len(set(train_labels))

    a = np.zeros(shape=(len(train_labels), dim_out))
    for i in range(len(train_labels)):
        a[i, int(train_labels[i])] = 1
    train_out = torch.LongTensor(a).to(device)

    train_data = torch.Tensor(train_images).unsqueeze(1).to(device) / 256
    # train_out = torch.Tensor(train_labels).to(device)

    num_layer = 3
    channel = [1, 32, 64, 32]
    channel = [1, 64, 128, 64]
    channel = [1, 32, 128, 32]
    channel = [1, 32, 256, 32]

    # num_layer = 4
    # channel = [1, 32, 64, 128, 64]
    # channel = [1, 64, 128, 256, 64]
    # #
    # num_layer = 5
    # channel = [1, 32, 64, 128, 64, 32]
    # channel = [1, 32, 64, 256, 64, 32]
    # #
    # num_layer = 6
    # channel = [1, 32, 64, 128, 256, 64,32]
    # channel = [1, 32, 64, 256, 256, 64,32]

    kernal = 3
    pad = (1, 1)

    dim_in = train_data.shape[-1]
    dim_out1 = 512

    net = CNNN(num_layer, channel, kernal, pad, dim_in, dim_out, dim_out1)
    net.to(device)

    num_epoch = 100
    lr = 0.001
    weight_decay = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    acc = []
    for epoch in range(num_epoch):
        net.train()

        logit = net(train_data)

        train_loss = torch.mean(torch.abs(logit - train_out))

        print('--------------------------------------')
        print('train_epoch:{}  loss:{}'.format(epoch, train_loss))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        logit = net(train_data)

        a = 0
        for ii, i in enumerate(logit):
            if torch.argmax(i) == torch.argmax(train_out[ii]):
                a += 1
        acc.append(a / logit.shape[0])
        print('train_epoch:{}  acc:{}'.format(epoch, a / logit.shape[0]))

acc = np.array(acc)
q = np.where(acc == acc.max())[0][0]
print('best acc:{}'.format(acc[q]))
a = 111111
