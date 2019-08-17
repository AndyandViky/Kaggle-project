# ======================================= #
# from google.colab import drive
# drive.mount('/content/drive/')
# os.chdir('/content/drive/My Drive/Digit-Recongnizer')

import os

import pandas as pd
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


Digit_DIR = os.path.dirname(os.path.abspath(__file__))

Data_DIR = os.path.join(Digit_DIR, 'datasets')

Model_DIR = os.path.join(Digit_DIR, 'model')
os.makedirs(Model_DIR, exist_ok=True)

Log_DIR = os.path.join(Digit_DIR, 'log')
os.makedirs(Log_DIR, exist_ok=True)

train_data_path = os.path.join(Data_DIR, 'train.csv')
test_data_path = os.path.join(Data_DIR, 'test.csv')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# util functions
def csv2tensor(data):
    return torch.from_numpy(data.to_numpy())


def init_weights(Net):
    for net in Net.modules():
        if isinstance(net, nn.Conv2d):
            net.weight.data.normal_(0, 0.02)
            net.bias.data.zero_()
        if isinstance(net, nn.Linear):
            net.weight.data.normal_(0, 0.02)
            net.bias.data.zero_()


def caculate_accuracy(pred, target):

    correct = 0
    correct += (pred == target).sum()

    acc = correct.numpy() / len(pred.numpy())
    return acc


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.shape
        )


class DigitData(Dataset):

    def __init__(self, datapath, transform=None, train=True):
        # read csv file data
        self.datapath = datapath
        self.transform = transform
        self.datas = csv2tensor(pd.read_csv(self.datapath, skiprows=1))
        self.train = train

        train_length = int(len(self.datas) * 0.9)
        if self.train:
            self.image, self.label = self.datas[:train_length, 1:], self.datas[:train_length, 0]
        else:
            self.image, self.label = self.datas[train_length:, 1:], self.datas[train_length:, 0]

    def __getitem__(self, index):

        img, target = self.image.data[index], int(self.label.data[index])
        img = img.view(28, 28)
        # img = Image.fromarray(img.numpy(), mode='L')

        # plt.imshow(img)
        # plt.show()

        if self.transform is not None:
            img = self.transform(img.numpy())

        return torch.as_tensor(img, dtype=torch.float32), target

    def __len__(self):

        return len(self.image)


class Classifier_CNN(nn.Module):

    def __init__(self, feature_dim=784, latent_dim=10, input_size=(1, 28, 28), verbose=False):
        super(Classifier_CNN, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.input_size = input_size

        self.cshape = (128, 7, 7)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.vervose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, self.latent_dim),
            nn.Softmax(),
        )

        init_weights(self)

        if self.vervose:
            print(self.model)

    def forward(self, x):

        return self.model(x)


class Trainer:

    def __init__(self, batch_size=64,
                 test_batch_size=1200,
                 lr=1e-4,
                 feature_dim=784,
                 latent_dim=10,
                 input_size=(1, 28, 28),
                 betas=(0.5, 0.99),
                 decay=2.5*1e-5,
                 epochs=200):
        super(Trainer, self).__init__()

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.betas = betas
        self.decay = decay
        self.epochs = epochs

    def getdataloader(self, path, batch_size, train=True):

        dataloader = torch.utils.data.DataLoader(
            DigitData(path, train=train, transform=transforms.Compose([
                transforms.ToTensor(),
            ])),
            batch_size=batch_size,
            shuffle=True,
        )
        return dataloader

    def train(self):
        print("begin training ......")
        clf = Classifier_CNN(feature_dim=self.feature_dim,
                             latent_dim=self.latent_dim,
                             input_size=self.input_size)
        clf_ops = torch.optim.Adam(clf.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.decay)
        # clf_ops = torch.optim.SGD(clf.parameters(), lr=0.01, momentum=0.9)
        mls_loss = nn.CrossEntropyLoss()

        # to device: cuda or cpu
        clf.to(device)
        mls_loss.to(device)

        dataloader = self.getdataloader(train_data_path, self.batch_size, train=True)
        testdataloader = self.getdataloader(train_data_path, self.test_batch_size, train=False)

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        for epoch in range(self.epochs):
            tra_los = []
            tra_acc = []
            for index, (data, target) in enumerate(dataloader):

                data, target = data.to(device), target.to(device)

                # from torchvision.utils import save_image
                # save_image(data[:25], './test.png', nrow=5, normalize=False)

                clf.train()
                clf.zero_grad()
                clf_ops.zero_grad()

                logit = clf(data)

                loss = mls_loss(logit, target)

                loss.backward(retain_graph=True)
                clf_ops.step()

                tra_los.append(loss.data.cpu().numpy())

                # caculate accuracy
                pred = torch.argmax(logit, dim=1)
                acc = caculate_accuracy(pred.data.cpu(), target.data.cpu())
                tra_acc.append(acc)

            # test
            clf.eval()
            _data, _target = next(iter(testdataloader))
            _data, _target = _data.to(device), _target.to(device)

            _logit = clf(_data)
            _loss = mls_loss(_logit, _target)
            test_loss.append(_loss.data.cpu().numpy())
            _pred = torch.argmax(_logit, dim=1)
            _acc = caculate_accuracy(_pred.data.cpu(), _target.data.cpu())
            test_acc.append(_acc)

            train_loss.append(np.mean(tra_los))
            train_acc.append(np.mean(tra_acc))

            print("[Digit-Recongnizer] epoch: {}, train_loss: {}, train_acc: {}, test_loss: {}, test_acc: {}"
                  .format(epoch, train_loss[epoch], train_acc[epoch], test_loss[epoch], test_acc[epoch]))

            logger = open(os.path.join(Log_DIR, "log.txt"), 'a')
            logger.write(
                "[Digit-Recongnizer] epoch: {}, train_loss: {}, train_acc: {}, test_loss: {}, test_acc: {}"
                  .format(epoch, train_loss[epoch], train_acc[epoch], test_loss[epoch], test_acc[epoch])
            )
            logger.close()

        # predict
        predict_data = csv2tensor(pd.read_csv(test_data_path))
        predict_data = predict_data.view(-1, 1, 28, 28)
        predict_data = torch.as_tensor(predict_data, dtype=torch.float32, device=device)
        predict_result = torch.argmax(clf(predict_data), dim=1)

        predict_result = predict_result.data.cpu()
        image_id = [i for i in range(1, len(predict_result) + 1)]
        predict_result = pd.DataFrame({'ImageId': image_id, 'Label': predict_result})
        predict_result.to_csv('predict.csv', index=False)

        # save model
        torch.save(clf.state_dict(), os.path.join(Model_DIR, 'cls.pkl'))


if __name__ == "__main__":

    # training var
    lr = 0.01
    batch_size = 64
    test_batch_size = 1200
    feature_dim = 784
    latent_dim = 10
    input_size = (1, 28, 28)
    betas = (0.5, 0.99)
    decay = 2.5 * 1e-5
    epochs = 20

    trainer = Trainer(batch_size=batch_size,
                      test_batch_size=test_batch_size,
                      lr=lr,
                      feature_dim=feature_dim,
                      latent_dim=latent_dim,
                      input_size=input_size,
                      betas=betas,
                      decay=decay,
                      epochs=epochs)
    trainer.train()


