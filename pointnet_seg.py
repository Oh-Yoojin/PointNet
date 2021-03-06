import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNet_seg(nn.Module):
    def __init__(self, n_class=4):
        super(PointNet_sef, self).__init__()
        # stn3d
        self.stn_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.stn_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.stn_conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.stn_fc1 = nn.Linear(1024, 512)
        self.stn_fc2 = nn.Linear(512, 256)
        self.stn_fc3 = nn.Linear(256, 9)
        self.stn_bn1 = nn.BatchNorm1d(64)
        self.stn_bn2 = nn.BatchNorm1d(128)
        self.stn_bn3 = nn.BatchNorm1d(1024)
        self.stn_bn4 = nn.BatchNorm1d(512)
        self.stn_bn5 = nn.BatchNorm1d(256)

        # feat
        self.feat_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.feat_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.feat_conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.feat_bn1 = nn.BatchNorm1d(64)
        self.feat_bn2 = nn.BatchNorm1d(128)
        self.feat_bn3 = nn.BatchNorm1d(1024)

        # seg
        self.seg_conv1 = nn.Conv1d(1088, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, n_class, 1)
        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)


    def forward(self, x): # if x size (batch, channel, num_points) == (32, 3, 2048),
        ori_input = copy.deepcopy(x)
        # stn3d
        batchsize = x.size()[0]
        x = F.relu(self.stn_bn1(self.stn_conv1(x))) # (32, 64, 2048)
        x = F.relu(self.stn_bn2(self.stn_conv2(x))) # (32, 128, 2048)
        x = F.relu(self.stn_bn3(self.stn_conv3(x))) # (32, 1024, 2048)
        x = torch.max(x, 2, keepdim=True)[0] # (32, 1024, 1)
        x = x.view(-1, 1024) # (32, 1024)
        x = F.relu(self.stn_bn4(self.stn_fc1(x))) # (32, 512)
        x = F.relu(self.stn_bn5(self.stn_fc2(x))) # (32, 256)
        x = self.stn_fc3(x) # (32, 9)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize, 1) # (32, 9)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # (32, 9)
        trans = x.view(-1, 3, 3) # (32, 3, 3)

        # feat
        x = ori_input.transpose(2, 1) # (32, 2048, 3)
        x = torch.bmm(x, trans) # (32, 2048, 3)
        x = x.transpose(2, 1) # (32, 3, 2048)
        x = F.relu(self.feat_bn1(self.feat_conv1(x))) # (32, 64, 2048)
        x = F.relu(self.feat_bn2(self.feat_conv2(x))) # (32, 128, 2048)
        x = self.feat_bn3(self.feat_conv3(x)) # (32, 1024, 2048)
        x = torch.max(x, 2, keepdim=True)[0] # (32, 1024, 1)
        x = x.view(-1, 1024) # (32, 1024)

        # seg
        x = F.relu(self.seg_bn1(self.seg_conv1(x)))
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x = F.relu(self.seg_bn3(self.seg_conv3(x)))
        x = self.seg_conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1, n_class), dim=-1)
        x = x.view(batchsize, num_points, n_class)
        return x, trans
