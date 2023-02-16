import torch.nn as nn
import torch
import torch.nn.functional as F
from models.deform_conv_v2 import DeformableConv2d
from models.Trans_atten import *

class Deformable_Attention(nn.Module):

    def __init__(self, nc, K=8, downsample=4):
        # Attention_module(nc=128, K=8)
        super(Deformable_Attention, self).__init__()
        self.K = K

        nm = [512,256,128]

        self.fc_k = nn.Linear(192, 192, bias=False)
        self.fc_v = nn.Linear(192, 192, bias=False)

        atten_0_0 = nn.Sequential()
        atten_0_0.add_module('conv_a_0',nn.Conv2d(nc, nm[1], 3, 1, 1))
        atten_0_0.add_module('bn_a_0', nn.BatchNorm2d(nm[1]))
        atten_0_0.add_module('relu_a_0', nn.ReLU(True))
        atten_0_1 = nn.Sequential()
        atten_0_1.add_module('conv_a_1',nn.Conv2d(nm[1], nm[1], 3, 1, 1))
        atten_0_1.add_module('bn_a_1', nn.BatchNorm2d(nm[1]))
        atten_0_1.add_module('relu_a_1', nn.ReLU(True))
        atten_0_2 = nn.Sequential()
        atten_0_2.add_module('conv_a_2',nn.Conv2d(nm[1], nm[1], 3, 1, 1))
        atten_0_2.add_module('bn_a_2', nn.BatchNorm2d(nm[1]))
        atten_0_2.add_module('relu_a_2', nn.ReLU(True))
        atten_0_2.add_module('pooling_a_2',nn.MaxPool2d((2, 2)))
        atten_0_3 = nn.Sequential()
        atten_0_3.add_module('conv_a_3',nn.Conv2d(nm[1], nm[2], 3, 1, 1))
        atten_0_3.add_module('bn_a_3', nn.BatchNorm2d(nm[2]))
        atten_0_3.add_module('relu_a_3', nn.ReLU(True))
        atten_0_4 = nn.Sequential()
        atten_0_4.add_module('conv_a_4',nn.Conv2d(nm[2], nm[2], 3, 1, 1))
        atten_0_4.add_module('bn_a_4', nn.BatchNorm2d(nm[2]))
        atten_0_4.add_module('relu_a_4', nn.ReLU(True))
        atten_0_5 = nn.Sequential()
        atten_0_5.add_module('conv_a_5',nn.Conv2d(nm[2], nm[2], 3, 1, 1))
        atten_0_5.add_module('bn_a_5', nn.BatchNorm2d(nm[2]))
        atten_0_5.add_module('relu_a_5', nn.ReLU(True))
        atten_0_5.add_module('pooling_a_5',nn.MaxPool2d((2, 2)))

        atten_0 = nn.Sequential()
        atten_0.add_module('conv_a_6',DeformableConv2d(nm[2], nm[2], 3, 1, 1))
        atten_0.add_module('bn_a_6', nn.BatchNorm2d(nm[2]))
        atten_0.add_module('relu_a_6', nn.ReLU(True))

        atten_1 = nn.Sequential()
        atten_1.add_module('conv_a_7',DeformableConv2d(nm[2], nm[2], 3, 1, 1))
        atten_1.add_module('bn_a_7', nn.BatchNorm2d(nm[2]))
        atten_1.add_module('relu_a_7', nn.ReLU(True))

        self.atten_0_0 = atten_0_0
        self.atten_0_1 = atten_0_1
        self.atten_0_2 = atten_0_2
        self.atten_0_3 = atten_0_3
        self.atten_0_4 = atten_0_4
        self.atten_0_5 = atten_0_5
        self.atten_0 = atten_0
        self.atten_1 = atten_1

        Fc_dimension = int(96*32/downsample/downsample/16)
        self.atten_fc1 = nn.Linear(Fc_dimension, Fc_dimension)
        self.atten_fc2 = nn.Linear(Fc_dimension, Fc_dimension)

        self.cnn_1_1 = DeformableConv2d(nm[1],64,1,1,0)

        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid    = nn.Sigmoid()

        self.deconv1 = nn.ConvTranspose2d(nm[2], 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, self.K, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(self.K)

    def forward(self, input):
        # conv features
        batch_size = input.size(0)
        conv_out = input

        fc_1 = conv_out.reshape(batch_size, input.size(1), -1)
        conv_k = self.fc_k(fc_1)
        conv_v = self.fc_v(fc_1)
        conv_k = conv_k.reshape(batch_size, input.size(1), input.size(2), input.size(3))
        conv_v = conv_v.reshape(batch_size, input.size(1), input.size(2), input.size(3))

        x00 = self.atten_0_0(conv_k)
        x01 = self.atten_0_1(x00)
        x02 = self.atten_0_2(x01)
        x03 = self.atten_0_3(x02)
        x04 = self.atten_0_4(x03)
        x05 = self.atten_0_5(x04)
        x0 = self.atten_0(x05)
        x1 = self.atten_1(x0)

        channel = x1.size(1)
        height = x1.size(2)
        width = x1.size(3)
        fc_x = x1.view(batch_size, channel, -1)

        fc_atten = self.atten_fc2(self.atten_fc1(fc_x))
        fc_atten = fc_atten.reshape(batch_size, channel, height, width)

        score = self.relu(self.deconv1(fc_atten))
        score = self.bn1(score+self.cnn_1_1(x02))
        atten = self.sigmoid(self.deconv2(score))

        atten_list = torch.chunk(atten, atten.shape[0], 0)
        atten = atten.reshape(batch_size, self.K, -1)
        conv_v = conv_v.reshape(conv_v.size(0), conv_v.size(1), -1)

        conv_v = conv_v.permute(0,2,1)

        atten_out = torch.bmm(atten, conv_v)
        atten_out = atten_out.view(batch_size, self.K, -1)
        
        return atten_list, atten_out