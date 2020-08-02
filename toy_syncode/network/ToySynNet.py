import torch
import torch.nn as nn
import torch.nn.functional as F
from network.classification import Classification, Classification2
import torchvision.models as models


class ToySynNet_Pretrained_ResNet(nn.Module):
    def __init__(self, clip_length):
        super().__init__()
        self.clip_length = clip_length

        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # self.fc1 = nn.Linear(54*54*16, 120)
        self.fc1 = nn.Linear(198 * 198 * 16, 120)
        self.fc2 = nn.Linear(120, 84)


        # self.classification1 = Classification(self.clip_length)
        self.classification2 = Classification2(self.clip_length)
        # self.conv_encoder = ConvEncoder()

    def forward(self, clip1, clip2):
        # clip1 = self.conv_encoder(clip1)
        # clip2 = self.conv_encoder(clip2)
        clip1 = F.relu(self.conv1(clip1)) # [10, 3, 800, 800]
        clip2 = F.relu(self.conv1(clip2))

        clip1 = F.max_pool2d(clip1, 2, 2)
        clip2 = F.max_pool2d(clip2, 2, 2)

        clip1 = F.relu(self.conv2(clip1))
        clip2 = F.relu(self.conv2(clip2))

        clip1 = F.max_pool2d(clip1, 2, 2)
        clip2 = F.max_pool2d(clip2, 2, 2)

        # clip1 = clip1.view(-1, 54*54*16)
        # clip2 = clip2.view(-1, 54*54*16)
        clip1 = clip1.view(-1, 198 * 198 * 16)
        clip2 = clip2.view(-1, 198 * 198 * 16)

        clip1 = F.relu(self.fc1(clip1))
        clip2 = F.relu(self.fc1(clip2))

        clip1 = F.relu(self.fc2(clip1))
        clip2 = F.relu(self.fc2(clip2))  # torch.Size([10, 84])

        # print('classification result: ', self.classification(clip1, clip2))
        # start_time1 = time.time()
        # classification1 = self.classification1(clip1, clip2)
        # end_time1 = time.time()
        # print('1111111 spent time: ', end_time1 - start_time1)
        # print('classification1111: ', classification1)

        # start_time2 = time.time()
        classification2 = self.classification2(clip1, clip2)
        # end_time2 = time.time()
        # print('2222222 spent time: ', end_time2 - start_time2)
        # print('classification2222: ', classification2)

        return classification2

class ToySynNet(nn.Module):
    def __init__(self, clip_length):
        super().__init__()
        self.clip_length = clip_length

        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # self.fc1 = nn.Linear(54*54*16, 120)
        self.fc1 = nn.Linear(198 * 198 * 16, 120)
        self.fc2 = nn.Linear(120, 84)

        # self.classification1 = Classification(self.clip_length)
        self.classification2 = Classification2(self.clip_length)
        # self.conv_encoder = ConvEncoder()
    
    def encoder(self, frame):
        frame = F.relu(self.conv1(frame))
        frame = F.max_pool2d(frame, 2, 2)
        frame = F.relu(self.conv2(frame))
        frame = F.max_pool2d(frame, 2, 2)
        frame = frame.view(-1, 198 * 198 * 16)
        frame = F.relu(self.fc1(frame))
        frame = F.relu(self.fc2(frame))
        return frame

    def forward(self, clip1, clip2):
        clip1 = torch.stack([self.encoder(torch.unsqueeze(frame, 0)) for frame in clip1], dim=0)
        clip2 = torch.stack([self.encoder(torch.unsqueeze(frame, 0)) for frame in clip2], dim=0)
        clip1 = torch.squeeze(clip1)
        clip2 = torch.squeeze(clip2)

        classification2 = self.classification2(clip1, clip2)

        return classification2

class ToySynNetOnlyClassification(nn.Module):
    def __init__(self, clip_length):
        super().__init__()
        self.clip_length = clip_length
        self.classification = Classification2(clip_length)

    def forward(self, clip1, clip2):
        classification = self.classification(clip1, clip2)
        return classification