import cv2
import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import random
from network.ToySynNet import ToySynNet, ToySynNetOnlyClassification
from dataloader.dataloader import ToyDataset
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

clip_length = 10

dataset_transform = transforms.Compose([
  transforms.ToPILImage(),
  # transforms.Resize((224, 224)),
  transforms.ToTensor()
])

train_dataset = ToyDataset(data_folder='', clip_length=clip_length, train=True, transform=dataset_transform)
train_loader = DataLoader(train_dataset, batch_size=1)

test_dataset = ToyDataset(data_folder='', clip_length=clip_length, train=False, transform=dataset_transform)
test_loader = DataLoader(test_dataset, batch_size=1)


model = ToySynNet(clip_length).cuda()
criterion = nn.CrossEntropyLoss().cuda()
# model = ToySynNet()
# criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# checkpoint_save_path = '/scratch/heyizhuo/checkpoints/'

def main():
    # result_txt_file = open("/scratch/heyizhuo/result_text3.txt","w") 
    start_time = time.time()
    tb = SummaryWriter()
    # epochs = 5

    # for epoch in range(epochs):
    #     train_total_loss = 0
    #     train_total_correct_d0 = 0
    #     train_total_correct_d1 = 0
    #     train_total_correct_d2 = 0
    #     train_total_correct_d3 = 0

    #     # Run the training batches
    #     for b, (clip1, clip2, label) in enumerate(train_loader):

    #         clip1 = clip1.cuda()
    #         clip2 = clip2.cuda()
    #         label = label.cuda()
    #         # clip1 = clip1
    #         # clip2 = clip2
    #         # label = label

    #         y_pred = model(torch.squeeze(clip1), torch.squeeze(clip2)).cuda()
    #         # y_pred = model(torch.squeeze(clip1), torch.squeeze(clip2), 10)
    #         # print('y_pred: ', torch.argmax(y_pred) - 5)
    #         # print('label: ', label)
    #         y_pred = torch.unsqueeze(y_pred, 0)
    #         loss = criterion(y_pred, label.long())
    #         train_total_loss += loss.item()
    #         predicted = torch.max(y_pred.data, 1)[1].item()
    #         label = label.item()

    #         print('predicted: ', predicted)
    #         print('label: ', label)

    #         if predicted == label:
    #             train_total_correct_d0 += 1
    #         if abs(predicted - label) <= 1:
    #             train_total_correct_d1 += 1
    #         if abs(predicted - label) <= 2:
    #             train_total_correct_d2 += 1
    #         if abs(predicted - label) <= 3:
    #             train_total_correct_d3 += 1
            
    #         # batch_corr = (predicted == label)
    #         # # print('correct: ', batch_corr)

    #         # Update parameters
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # train_total_correct += batch_corr.item()

    #         # # Print interim results
    #         # if b != 0:
    #         #     print(f'epoch: {epoch}, batch: {b}, train_total_correct: {train_total_correct}, total_loss: {train_total_loss}')

    #         # Print interim results
    #         if b != 0:
    #             text = f'epoch: {epoch}, batch: {b}, train_total_correct_d0: {train_total_correct_d0}, train_total_correct_d1: {train_total_correct_d1}, train_total_correct_d2: {train_total_correct_d2}, train_total_correct_d3: {train_total_correct_d3}, train_total_loss: {train_total_loss}\n'
    #             result_txt_file.write(text) 
    #             print(text)

        # tb.add_scalar('Train/Loss', train_total_loss, epoch)
        # tb.add_scalar('Train/Number Correct', train_total_correct, epoch)

        # train_losses.append(loss)
        # train_correct.append(trn_corr)

        # torch.save({'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'train_total_loss': train_total_loss,
        #             'train_total_correct': train_total_correct},
        #            checkpoint_save_path + 'checkpoint.tar')

    # Run the testing batches
    with torch.no_grad():
        test_total_loss = 0
        test_total_correct_d0 = 0
        test_total_correct_d1 = 0
        test_total_correct_d2 = 0
        test_total_correct_d3 = 0

        for b, (clip1, clip2, label) in enumerate(test_loader):

            clip1 = clip1.cuda()
            clip2 = clip2.cuda()
            label = label.cuda()

            y_pred = model(torch.squeeze(clip1), torch.squeeze(clip2)).cuda()
            y_pred = torch.unsqueeze(y_pred, 0)
            loss = criterion(y_pred, label.long())
            test_total_loss += loss.item()

            predicted = torch.max(y_pred.data, 1)[1].item()
            label = label.item()

            print('predicted: ', predicted)
            print('label: ', label)

            if predicted == label:
                test_total_correct_d0 += 1
            if abs(predicted - label) <= 1:
                test_total_correct_d1 += 1
            if abs(predicted - label) <= 2:
                test_total_correct_d2 += 1
            if abs(predicted - label) <= 3:
                test_total_correct_d3 += 1

            # Print interim results
            text = f'batch: {b}, test_total_correct_d0: {test_total_correct_d0}, test_total_correct_d1: {test_total_correct_d1}, test_total_correct_d2: {test_total_correct_d2}, test_total_correct_d3: {test_total_correct_d3}, test_total_loss: {test_total_loss}'
            # result_txt_file.write(text)
            print(text)

        # tb.add_scalar('Test/Loss', test_total_loss, epoch)
        # tb.add_scalar('Test/Number Correct', test_total_correct, epoch)

    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
    # result_txt_file.close()

def main_only_classification():
    model_only_classification = ToySynNetOnlyClassification(clip_length).cuda()

    # checkpoint = torch.load(checkpoint_save_path + 'checkpoint.tar')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # model.eval()

    # Run the testing batches
    with torch.no_grad():
        test_total_loss = 0
        test_total_correct_d0 = 0
        test_total_correct_d1 = 0
        test_total_correct_d2 = 0
        test_total_correct_d3 = 0

        for b, (clip1, clip2, label) in enumerate(test_loader):

            clip1 = clip1.cuda()
            clip2 = clip2.cuda()
            label = label.cuda()

            y_pred = model_only_classification(torch.squeeze(clip1), torch.squeeze(clip2)).cuda()
            # y_pred = model(torch.squeeze(clip1), torch.squeeze(clip2)).cuda()
            y_pred = torch.unsqueeze(y_pred, 0)
            loss = criterion(y_pred, label.long())
            test_total_loss += loss.item()
            predicted = torch.max(y_pred.data, 1)[1].item()
            label = label.item()

            if predicted == label:
                test_total_correct_d0 += 1
            if abs(predicted - label) <= 1:
                test_total_correct_d1 += 1
            if abs(predicted - label) <= 2:
                test_total_correct_d2 += 1
            if abs(predicted - label) <= 3:
                test_total_correct_d3 += 1

            print('predicted: ', predicted)
            print('label: ', label)

            # Print interim results
            if b != 0:
                print(f'batch: {b}, test_total_correct_d0: {test_total_correct_d0}, test_total_correct_d1: {test_total_correct_d1}, test_total_correct_d2: {test_total_correct_d2}, test_total_correct_d3: {test_total_correct_d3}, test_total_loss: {test_total_loss}')

        # tb.add_scalar('Test/Loss', test_total_loss, epoch)
        # tb.add_scalar('Test/Number Correct', test_total_correct, epoch)

    # print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

if __name__ == '__main__':
    # main_only_classification()
    main()