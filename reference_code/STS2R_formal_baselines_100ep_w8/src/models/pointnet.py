import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=6):
        super(PointNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.stn = STN3d(3) # T-Net always works on XYZ (3)
        self.conv1 = torch.nn.Conv1d(input_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, num_classes, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        # x: (B, channel, N) -> We expect channel=9 now
        # channel=9 structure: [Local_XYZ (3), RGB (3), Global_XYZ (3)]
        n_pts = x.size()[2]
        
        # Split features
        xyz = x[:, 0:3, :]  # Local XYZ
        other_features = x[:, 3:, :] # RGB + Global XYZ (6 channels)
        
        trans = self.stn(xyz)
        xyz = xyz.transpose(2, 1)
        xyz = torch.bmm(xyz, trans)
        xyz = xyz.transpose(2, 1)
        
        # Concatenate transformed XYZ with other features
        # Total channels should be 3 (transformed xyz) + 6 (others) = 9
        # This matches self.conv1 input channels if initialized with channel=9
        x = torch.cat([xyz, other_features], 1)
        
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        pointfeat = x
        
        x = F.relu(self.bn2(self.conv2(x))) # (B, 128, N)
        x = self.bn3(self.conv3(x)) # (B, 1024, N)
        x = torch.max(x, 2, keepdim=True)[0] # (B, 1024, 1)
        
        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts) # (B, 1024, N)
        x = torch.cat([x, pointfeat], 1) # (B, 1088, N)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x) # (B, k, N)
        
        # Return raw logits [B, k, N] for CrossEntropyLoss
        return x, trans
