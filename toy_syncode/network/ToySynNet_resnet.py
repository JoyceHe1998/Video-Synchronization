import torch
import torch.nn as nn
import torch.nn.functional as F
from network.classification import Classification, Classification2
import torchvision.models as models

# resNetModel = models.resnet50(pretrained=True)

# for param in resNetModel.parameters():
#     param.requires_grad = False

# print(resNetModel)

# resNetModel.fc = nn.Linear(2048, 84)

# print(resNetModel)

class ToySynNet_resnet(nn.Module):
    def __init__(self, clip_length):
        super().__init__()
        self.clip_length = clip_length
        # self.classification1 = Classification(self.clip_length)
        self.resnet = models.resnet50(pretrained=True)
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        self.resnet.fc = nn.Linear(2048, 84)
        self.classification2 = Classification2(self.clip_length)
    

    def forward(self, clip1, clip2):
        clip1 = torch.stack([F.relu(self.resnet(torch.unsqueeze(frame, 0))) for frame in clip1], dim=0)
        clip2 = torch.stack([F.relu(self.resnet(torch.unsqueeze(frame, 0))) for frame in clip2], dim=0)
        clip1 = torch.squeeze(clip1)
        clip2 = torch.squeeze(clip2)

        classification2 = self.classification2(clip1, clip2)

        return classification2