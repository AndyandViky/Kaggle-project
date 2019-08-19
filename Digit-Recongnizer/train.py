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
import torchvision.transforms as transforms
from config import Data_DIR, Model_DIR, Digit_DIR
from utils import csv2tensor, caculate_accuracy
from datasets import DigitData
from model import Classifier_CNN, SpacialTransformer
import matplotlib.pyplot as plt


os.makedirs(Model_DIR, exist_ok=True)

Log_DIR = os.path.join(Digit_DIR, 'log')
os.makedirs(Log_DIR, exist_ok=True)

train_data_path = os.path.join(Data_DIR, 'train.csv')
test_data_path = os.path.join(Data_DIR, 'test.csv')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        print("begin training1 ......")
        clf = Classifier_CNN(feature_dim=self.feature_dim,
                             latent_dim=self.latent_dim,
                             input_size=self.input_size)
        transform_net = SpacialTransformer()
        transform_net.load_state_dict(torch.load(os.path.join(Model_DIR, 'pre_transform.pkl'), map_location='cpu'))

        clf_ops = torch.optim.Adam(clf.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.decay)
        # clf_ops = torch.optim.SGD(clf.parameters(), lr=0.01, momentum=0.9)
        mls_loss = nn.CrossEntropyLoss()

        # to device: cuda or cpu
        clf.to(device)
        transform_net.to(device)
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

                # get extend data
                # transform_data = transform_net(data)

                # from torchvision.utils import save_image
                # save_image(data[:25], './test1.png', nrow=5, normalize=False)
                # save_image(transform_data[:25], './test.png', nrow=5, normalize=False)

                clf.train()
                clf.zero_grad()

                # according origin data to update parameters
                clf_ops.zero_grad()
                logit = clf(data)
                loss = mls_loss(logit, target)
                loss.backward()
                clf_ops.step()

                # according transform data to update parameters
                # clf_ops.zero_grad()
                # t_logit = clf(transform_data)
                # t_loss = mls_loss(t_logit, target)
                # t_loss.backward()
                # clf_ops.step()

                tra_los.append(loss.data.cpu().numpy())

                # caculate origin data accuracy
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
        with torch.no_grad():
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
    betas = (0.5, 0.9)
    decay = 1.5 * 1e-5
    epochs = 40

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


