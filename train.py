import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from net import CNN
import ShannonAndBirch

# 神经网络参数
batch_size = 128
learning_rate = 1e-3
num_epoches = 40
USE_GPU = torch.cuda.is_available()
datas = ShannonAndBirch.getdata()
dataset = ShannonAndBirch.trainAndtest(datas, datas[41], batch_size)
print(type(dataset[0]))
model = CNN(1, 2)
if USE_GPU:
    model = model.cuda()


def train():

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(dataset[0], 1):
            img, label = data
            if USE_GPU:
                img = img.cuda()
                label = label.cuda()
            img = Variable(img)
            label = Variable(label)
            # 向前传播
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()
            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / dataset[2], running_acc / dataset[2]))
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for data in dataset[1]:
            img, label = data
            if USE_GPU:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / dataset[3], eval_acc / dataset[3]))
        print()


def predict(data, multiple=False):
    _data = dataset.encode(data)
    _data = torch.from_numpy(
        np.pad(_data, (0, 64 - len(_data)), 'constant').astype(np.float32)
    ).reshape(-1, 1, 8, 8).cuda()
    _out = int(torch.max(model(_data).data, 1)[1].cpu().numpy())
    return dataset.decode(_out, label=True)


if __name__ == '__main__':
    train()