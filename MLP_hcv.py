# 此文件用于MLP分类

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入数据
data_url = ("./data/hcvdat0.csv")
usecol = [1] + [i for i in range(4, 14)]
dataset = pd.read_csv(data_url, index_col=0, header=0, usecols=usecol)
data1 = dataset.values

for i, ii in enumerate(data1):
    for j, jj in enumerate(ii):
        if math.isnan(jj):
            data1[i, j] = 0

data1 = np.array(data1, dtype=float)
data_label = dataset.index

a = []
for i in data_label:
    if i[0] == '0':
        a.append(0)
    elif i[0] == '1':
        a.append(1)
    elif i[0] == '2':
        a.append(2)
    elif i[0] == '3':
        a.append(3)

dim_out=len(set(a))

a = np.zeros(shape=(data1.shape[0], dim_out))
for ii, i in enumerate(data_label):
    if i[0] == '0':
        a[ii, 0] = 1
    elif i[0] == '1':
        a[ii, 1] = 1
    elif i[0] == '2':
        a[ii, 2] = 1
    elif i[0] == '3':
        a[ii, 3] = 1

data_label = np.array(a)


class MLP(nn.Module):
    def __init__(self, num_layer, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.num_layer = num_layer

        self.feature = nn.Linear(dim_in, dim_hidden, bias=True)
        nn.init.xavier_normal_(self.feature.weight, gain=1.414)
        self.a=nn.LeakyReLU()

        self.MLP = nn.ModuleList()
        for i in range(num_layer - 1):
            self.MLP.append(nn.Linear(dim_hidden, dim_hidden, bias=True))
            nn.init.xavier_normal_(self.MLP[-1].weight, gain=1.414)
            self.MLP.append(nn.LeakyReLU())

        self.out = nn.Linear(dim_hidden, dim_out, bias=True)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, data):
        h = self.feature(data)
        # h = nn.LeakyReLU(h)
        h = self.a(h)

        for i in range(self.num_layer - 1):
            h = self.MLP[i](h)
            # h = nn.LeakyReLU(h)

        out = self.out(h)
        return out


train_data = torch.Tensor(data1)
train_out = torch.LongTensor(data_label)

num_epoch = 500
num_layer = 7
dim_in = data1.shape[1]
dim_hidden = 32

net = MLP(num_layer, dim_in, dim_hidden, dim_out)

lr = 0.01
weight_decay = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

acc=[]
for epoch in range(num_epoch):
    net.train()

    logit = net(train_data)

    # logp = F.log_softmax(logit, 1)
    # train_loss = F.nll_loss(logp, train_out)

    train_loss=torch.mean(torch.abs(logit-train_out))

    print('--------------------------------------')
    print('train_epoch:{}  loss:{}'.format(epoch, train_loss))

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    net.eval()
    logit = net(train_data)

    a = 0
    for ii,i in enumerate(logit):
        if torch.argmax(i) == torch.argmax(train_out[ii]):
            a += 1
    acc.append(a / logit.shape[0])
    print('train_epoch:{}  acc:{}'.format(epoch, a / logit.shape[0]))

acc=np.array(acc)
q=np.where(acc==acc.max())[0][0]
print('best acc:{}'.format(acc[q]))
a = 111111
