import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def mse(imageA, imageB):
  return -torch.mean((imageA - imageB)**2) * 10000000000000

# only calculate 11 (5*2 + 1) diagonals
def calculate_diagonals_partial(data_frame, sequence_len):
    diagonal_arr = []

    # diagonals of the left half + centre

    for i in range(int(sequence_len / 2 - 1), sequence_len):
        temp = sequence_len - 1 - i
        sum = 0
        for j in range(i + 1):
            sum += data_frame.iloc[temp, j]
            temp += 1
        diagonal_arr.append(sum / (i + 1))

    # diagonals of the right half
    for i in range(int(sequence_len / 2)):
        temp = i + 1
        sum = 0
        for j in range(sequence_len - 1 - i):
            sum += data_frame.iloc[j, j + temp]
        diagonal_arr.append(sum / (sequence_len - 1 - i))
    return diagonal_arr
    # return np.argmin(diagonal_arr) - 5


# mse_loss = nn.MSELoss(reduction='none')

class Classification(nn.Module):
    def __init__(self, sequence_length):
        super(Classification, self).__init__()
        self.sequence_length = sequence_length

    def forward(self, clip1, clip2):
        zero_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        df = pd.DataFrame({'c1_f1': zero_arr,
                           'c1_f2': zero_arr,
                           'c1_f3': zero_arr,
                           'c1_f4': zero_arr,
                           'c1_f5': zero_arr,
                           'c1_f6': zero_arr,
                           'c1_f7': zero_arr,
                           'c1_f8': zero_arr,
                           'c1_f9': zero_arr,
                           'c1_f10': zero_arr, 
                           'c1_f11': zero_arr,
                           'c1_f12': zero_arr,
                           'c1_f13': zero_arr,
                           'c1_f14': zero_arr,
                           'c1_f15': zero_arr,
                           'c1_f16': zero_arr,
                           'c1_f17': zero_arr,
                           'c1_f18': zero_arr,
                           'c1_f19': zero_arr,
                           'c1_f20': zero_arr},
                          index=['c2_f1', 'c2_f2', 'c2_f3', 'c2_f4', 'c2_f5', 'c2_f6', 'c2_f7', 'c2_f8', 'c2_f9',
                                 'c2_f10', 'c2_f11', 'c2_f12', 'c2_f13', 'c2_f14', 'c2_f15', 'c2_f16', 'c2_f17', 'c2_f18', 'c2_f19',
                                 'c2_f20'])
        for i in range(self.sequence_length):
            for j in range(self.sequence_length):
                # print('clip1: ', clip1[j])
                # print('clip2: ', clip2[i])
                df.iloc[i][j] = mse(clip1[j], clip2[i])
        # print(df)
        return torch.Tensor(calculate_diagonals_partial(df, self.sequence_length))

######################

def calculate_diagonals(matrix, sequence_length):
  # cuda()
  # arr = torch.zeros([sequence_length, sequence_length * 2 - 1]).cuda()
  arr = torch.zeros([sequence_length, sequence_length * 2 - 1])
  for i in range(sequence_length):
    arr[i, (sequence_length - 1 - i):(sequence_length * 2 - 1 - i)] = matrix[i, :]
  sum = arr.sum(dim = 0)

  # divide by array: e.g. when sequence_length = 10, divide_by_arr = torch.Tensor([1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1])
  second_half = np.arange(1, sequence_length + 1)[::-1].copy()
  # cuda()
  # second_half = torch.from_numpy(second_half).cuda()
  # first_half = torch.arange(1, sequence_length).cuda()
  second_half = torch.from_numpy(second_half)
  first_half = torch.arange(1, sequence_length)

  divide_by_arr = torch.cat((first_half, second_half))

  result = sum / divide_by_arr
  return result

class Classification2(nn.Module):
  def __init__(self, sequence_length):
    super(Classification2, self).__init__()
    self.sequence_length = sequence_length

  def forward(self, clip1, clip2):
    # cuda()
    # matrix = torch.zeros([self.sequence_length, self.sequence_length]).cuda()
    # xvalues = torch.arange(0, self.sequence_length).cuda()
    # yvalues = torch.arange(0, self.sequence_length).cuda()
    matrix = torch.zeros([self.sequence_length, self.sequence_length])
    xvalues = torch.arange(0, self.sequence_length)
    yvalues = torch.arange(0, self.sequence_length)

    xx, yy = torch.meshgrid(xvalues, yvalues)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    for i in range(len(xx)):
        matrix[xx[i]][yy[i]] = mse(clip1[yy[i]], clip2[xx[i]])

    # return calculate_diagonals(matrix, self.sequence_length)[4:15]
    return calculate_diagonals(matrix, self.sequence_length)[int(self.sequence_length / 2 - 1) : int(self.sequence_length * 3 / 2)]