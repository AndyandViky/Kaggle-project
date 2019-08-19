# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: spatial-transformer-networks.py
@time: 2019/5/26 下午10:38
@desc: 空间变换网络，允许神经网络学习如何在输入图像上执行空间变换，以增强模型的几何不变性。
例如，它可以裁剪感兴趣的区域，缩放并校正图像的方向。
只需要在现有CNN上做微小的调整就能实现空间变换网络。
https://zhuanlan.zhihu.com/p/37110107
https://blog.csdn.net/qq_39422642/article/details/78870629
https://blog.csdn.net/xbinworld/article/details/65660665
https://blog.csdn.net/xiaqunfeng123/article/details/17362881
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from config import Model_DIR

plt.ion() # 开启交互模式
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loading data
train_data = torch.utils.data.DataLoader(
    datasets.MNIST(root='./datasets', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]), download=True),
    batch_size=64,
    shuffle=True,
    num_workers=4
)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST(root='./datasets', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64,
    shuffle=True,
    num_workers=4
)


class SpacialTransformer(nn.Module):

    def __init__(self):
        super(SpacialTransformer, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),

            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], device=device, dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):

        return self.stn(x)


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        # usual network program
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


model = Net().to(device)
transform = SpacialTransformer().to(device)

# train
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    for index, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        model.zero_grad()()

        t_data = transforms(data)
        output = model(t_data)
        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        if index % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, index * len(data), len(train_data.dataset),
                       100. * index / len(train_data), loss.item()))


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_data.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_data.dataset),
                      100. * correct / len(test_data.dataset)))


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_data))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


for epoch in range(1, 20 + 1):
    train(epoch)
    test()
torch.save(transform.state_dict(), os.path.join(Model_DIR, "pre_transform.pkl"))

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()

