import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

class ToyDataset(Dataset):
    def __init__(self, data_folder, clip_length, train=True, transform=None, epoch=None, batch=None):
        self.train = train
        self.data_folder = data_folder
        self.transform = transform
        self.epoch = epoch
        self.batch = batch
        self.clip_length = clip_length
        self.transform = transform
        self.cam1_path = '/scratch/heyizhuo/cube_sphere_frames/sphere_cube_square_frame_cam1/'
        self.cam2_path = '/scratch/heyizhuo/cube_sphere_frames/sphere_cube_square_frame_cam2/'
        # self.cam1_path = '/Users/yizhuohe/Desktop/sphere_cube_square_frame_cam1/'
        # self.cam2_path = '/Users/yizhuohe/Desktop/sphere_cube_square_frame_cam2/'

        if self.train:
            self.cam1_path += 'train/'
            self.cam2_path += 'train/'
        else:
            self.cam1_path += 'test/'
            self.cam2_path += 'test/'

        self.img_num = len(os.listdir(self.cam1_path))
        self.first_possible_index = self.clip_length / 2
        self.last_possible_index = self.img_num - self.clip_length * 3 / 2
        self.cam1_imgs_name_list = sorted(os.listdir(self.cam1_path))
        self.cam2_imgs_name_list = sorted(os.listdir(self.cam2_path))

    def __getitem__(self, idx):
        time_offset = random.randint(- self.clip_length / 2, self.clip_length / 2)
        cam1_clip_start_index = random.randint(self.first_possible_index, self.last_possible_index)

        clip_cam1 = []
        clip_cam2 = []

        for i in range(self.clip_length):
            img_cam1_path = os.path.join(self.cam1_path, self.cam1_imgs_name_list[
                                                         cam1_clip_start_index: cam1_clip_start_index + self.clip_length][i])
            img_cam2_path = os.path.join(self.cam2_path, self.cam2_imgs_name_list[
                                                         cam1_clip_start_index + time_offset: cam1_clip_start_index + self.clip_length + time_offset][i])
            img_cam1 = cv2.cvtColor(cv2.imread(img_cam1_path), cv2.COLOR_BGR2RGB)
            img_cam2 = cv2.cvtColor(cv2.imread(img_cam2_path), cv2.COLOR_BGR2RGB)
            img_cam1 = self.transform(img_cam1)
            img_cam2 = self.transform(img_cam2)
            clip_cam1.append(img_cam1)
            clip_cam2.append(img_cam2)

        clip_cam1 = torch.squeeze(torch.stack(clip_cam1))
        clip_cam2 = torch.squeeze(torch.stack(clip_cam2))

        label = time_offset + self.clip_length / 2

        return clip_cam1, clip_cam2, label

    def __len__(self):
        if self.train:
            # return int(200)
            return int(round(self.last_possible_index - self.first_possible_index + 1))  # ????
        else:
            # return int(200)
            return int(round(self.last_possible_index - self.first_possible_index + 1))
