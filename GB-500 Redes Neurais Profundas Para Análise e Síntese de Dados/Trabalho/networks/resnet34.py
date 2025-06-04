# -*- coding: utf-8 -*-

import torch
import torch.nn as  nn
import torch.nn.functional as F
from torchvision.models import resnet34

from networks.biggan import discriminator

'''
Original ResNet34 architecture
'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = None, stride = 1):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1, stride = stride, bias = False),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(num_features = out_channels),
        )
 
        self.downsample = downsample
        self.relu = nn.ReLU(inplace = True)
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv_block(x)
       
        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add identity
        out += identity
        out = self.relu(out)
        
        return out

'''
34 layers
[2x(3x3,  64)] x 3
[2x(3x3, 128)] x 4
[2x(3x3, 256)] x 6
[2x(3x3, 512)] x 3
'''
class ResNet34(nn.Module):
    def __init__(self, ResBlock, pretrained_weights = None):
        super(ResNet34, self).__init__()

        self.in_channels = 64

        self.block_list = [3, 4, 6, 3]

        self.conv_block_0 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(num_features = self.in_channels),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.conv_block_1 = self.make_conv_block(ResBlock, self.block_list[0], planes =  64)
        self.conv_block_2 = self.make_conv_block(ResBlock, self.block_list[1], planes = 128, stride = 2)
        self.conv_block_3 = self.make_conv_block(ResBlock, self.block_list[2], planes = 256, stride = 2)
        self.conv_block_4 = self.make_conv_block(ResBlock, self.block_list[3], planes = 512, stride = 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(in_features = 512, out_features = 2)

        if pretrained_weights == 'ResNet34':
            self.init_resnet34_weights()

        if pretrained_weights == 'BigGAN':
            self.init_biggan_weights()

        if pretrained_weights == 'hybrid':
            self.init_hybrid_weights()
    
    def make_conv_block(self, ResBlock, blocks, planes, stride = 1):
        downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels = self.in_channels, out_channels = planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(num_features = planes)
            )
            
        layers.append(ResBlock(self.in_channels, planes, downsample = downsample, stride = stride))
        self.in_channels = planes
        
        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_block_0(x)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    '''
    Transfer learning using weights of the ResNet34 pretrained with ImageNet
    '''
    def init_resnet34_weights(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print('Loading ResNet34 weights...')
        resnet34_pretrained = resnet34(pretrained = True).to(device)
        resnet34_parameters = resnet34_pretrained.named_parameters()

        resnet34_conv_weights = []

        # Picks up weights
        for _, (name, parameters) in enumerate(resnet34_parameters):
            split_name = name.split('.')

            # Convolutional layer of initial block
            if split_name[0] == 'conv1':
                resnet34_conv_weights.append(parameters)

            if len(split_name) > 2:
                # Convolutional block layers
                if split_name[2] == 'conv1' or split_name[2] == 'conv2':
                    resnet34_conv_weights.append(parameters)

                # Convolutional downsample layers
                if split_name[2] == 'downsample' and split_name[3] == '0':
                    resnet34_conv_weights.append(parameters)
        print('Done!')

        print('Transfering weights from the ResNet34 to the classification network...')
        # 0 - Conv2d [0]
        self.conv_block_0[0].weight.data.copy_(resnet34_conv_weights[0].data)
        # 0 - BatchNorm2d [1]
        # 0 - ReLU [2]
        # 0 - MaxPool2d [3]
        ###################################################################################
        # 1.0 - Conv2d [0]
        self.conv_block_1[0].conv_block[0].weight.data.copy_(resnet34_conv_weights[1].data)
        # 1.0 - BatchNorm2d [1]
        # 1.0 - ReLU [2]
        # 1.0 - Conv2d [3]
        self.conv_block_1[0].conv_block[3].weight.data.copy_(resnet34_conv_weights[2].data)
        # 1.0 - BatchNorm2d [4]
        # 1.0 - ReLU [5]
        # 1.1 - Conv2d [6]
        self.conv_block_1[1].conv_block[0].weight.data.copy_(resnet34_conv_weights[3].data)
        # 1.1 - BatchNorm2d [7]
        # 1.1 - ReLU [8]
        # 1.1 - Conv2d [9]
        self.conv_block_1[1].conv_block[3].weight.data.copy_(resnet34_conv_weights[4].data)
        # 1.1 - BatchNorm2d [10]
        # 1.1 - ReLU [11]
        # 1.2 - Conv2d [12]
        self.conv_block_1[2].conv_block[0].weight.data.copy_(resnet34_conv_weights[5].data)
        # 1.2 - BatchNorm2d [13]
        # 1.2 - ReLU [14]
        # 1.2 - Conv2d [15]
        self.conv_block_1[2].conv_block[3].weight.data.copy_(resnet34_conv_weights[6].data)
        # 1.2 - BatchNorm2d [16]
        # 1.2 - ReLU [17]
        ###################################################################################
        # 2.0 - Conv2d [0]
        self.conv_block_2[0].conv_block[0].weight.data.copy_(resnet34_conv_weights[7].data)
        # 2.0 - BatchNorm2d [1]
        # 2.0 - ReLU [2]
        # 2.0 - Conv2d [3]
        self.conv_block_2[0].conv_block[3].weight.data.copy_(resnet34_conv_weights[8].data)
        # 2.0 - BatchNorm2d [4]
        # 2.0 - Conv2d (downsample) [5]
        self.conv_block_2[0].downsample[0].weight.data.copy_(resnet34_conv_weights[9].data)
        # 2.0 - BatchNorm2d (downsample) [6]
        # 2.0 - ReLU [7]
        # 2.1 - Conv2d [8]
        self.conv_block_2[1].conv_block[0].weight.data.copy_(resnet34_conv_weights[10].data)
        # 2.1 - BatchNorm2d [9]
        # 2.1 - ReLU [10]
        # 2.1 - Conv2d [11]
        self.conv_block_2[1].conv_block[3].weight.data.copy_(resnet34_conv_weights[11].data)
        # 2.1 - BatchNorm2d [12]
        # 2.1 - ReLU [13]
        # 2.2 - Conv2d [14]
        self.conv_block_2[2].conv_block[0].weight.data.copy_(resnet34_conv_weights[12].data)
        # 2.2 - BatchNorm2d [15]
        # 2.2 - ReLU [16]
        # 2.2 - Conv2d [17]
        self.conv_block_2[2].conv_block[3].weight.data.copy_(resnet34_conv_weights[13].data)
        # 2.2 - BatchNorm2d [18]
        # 2.2 - ReLU [19]
        # 2.3 - Conv2d [20]
        self.conv_block_2[3].conv_block[0].weight.data.copy_(resnet34_conv_weights[14].data)
        # 2.3 - BatchNorm2d [21]
        # 2.3 - ReLU [22]
        # 2.3 - Conv2d [23]
        self.conv_block_2[3].conv_block[3].weight.data.copy_(resnet34_conv_weights[15].data)
        # 2.3 - BatchNorm2d [24]
        # 2.3 - ReLU [25]
        ####################################################################################
        # 3.0 - Conv2d [0]
        self.conv_block_3[0].conv_block[0].weight.data.copy_(resnet34_conv_weights[16].data)
        # 3.0 - BatchNorm2d [1]
        # 3.0 - ReLU [2]
        # 3.0 - Conv2d [3]
        self.conv_block_3[0].conv_block[3].weight.data.copy_(resnet34_conv_weights[17].data)
        # 3.0 - BatchNorm2d [4]
        # 3.0 - Conv2d (downsample) [5]
        self.conv_block_3[0].downsample[0].weight.data.copy_(resnet34_conv_weights[18].data)
        # 3.0 - BatchNorm2d (downsample) [6]
        # 3.0 - ReLU [7]
        # 3.1 - Conv2d [8]
        self.conv_block_3[1].conv_block[0].weight.data.copy_(resnet34_conv_weights[19].data)
        # 3.1 - BatchNorm2d [9]
        # 3.1 - ReLU [10]
        # 3.1 - Conv2d [11]
        self.conv_block_3[1].conv_block[3].weight.data.copy_(resnet34_conv_weights[20].data)
        # 3.1 - BatchNorm2d [12]
        # 3.1 - ReLU [13]
        # 3.2 - Conv2d [14]
        self.conv_block_3[2].conv_block[0].weight.data.copy_(resnet34_conv_weights[21].data)
        # 3.2 - BatchNorm2d [15]
        # 3.2 - ReLU [16]
        # 3.2 - Conv2d [17]
        self.conv_block_3[2].conv_block[3].weight.data.copy_(resnet34_conv_weights[22].data)
        # 3.2 - BatchNorm2d [18]
        # 3.2 - ReLU [19]
        # 3.3 - Conv2d [20]
        self.conv_block_3[3].conv_block[0].weight.data.copy_(resnet34_conv_weights[23].data)
        # 3.3 - BatchNorm2d [21]
        # 3.3 - ReLU [22]
        # 3.3 - Conv2d [23]
        self.conv_block_3[3].conv_block[3].weight.data.copy_(resnet34_conv_weights[24].data)
        # 3.3 - BatchNorm2d [24]
        # 3.3 - ReLU [25]
        # 3.4 - Conv2d [26]
        self.conv_block_3[4].conv_block[0].weight.data.copy_(resnet34_conv_weights[25].data)
        # 3.4 - BatchNorm2d [27]
        # 3.4 - ReLU [28]
        # 3.4 - Conv2d [29]
        self.conv_block_3[4].conv_block[3].weight.data.copy_(resnet34_conv_weights[26].data)
        # 3.4 - BatchNorm2d [30]
        # 3.4 - ReLU [31]
        # 3.5 - Conv2d [32]
        self.conv_block_3[5].conv_block[0].weight.data.copy_(resnet34_conv_weights[27].data)
        # 3.5 - BatchNorm2d [33]
        # 3.5 - ReLU [34]
        # 3.5 - Conv2d [35]
        self.conv_block_3[5].conv_block[3].weight.data.copy_(resnet34_conv_weights[28].data)
        # 3.5 - BatchNorm2d [36]
        # 3.5 - ReLU [37]
        ####################################################################################
        # 4.0 - Conv2d [0]
        self.conv_block_4[0].conv_block[0].weight.data.copy_(resnet34_conv_weights[29].data)
        # 4.0 - BatchNorm2d [1]
        # 4.0 - ReLU [2]
        # 4.0 - Conv2d [3]
        self.conv_block_4[0].conv_block[3].weight.data.copy_(resnet34_conv_weights[30].data)
        # 4.0 - BatchNorm2d [4]
        # 4.0 - Conv2d (downsample) [5]
        self.conv_block_4[0].downsample[0].weight.data.copy_(resnet34_conv_weights[31].data)
        # 4.0 - BatchNorm2d (downsample) [6]
        # 4.0 - ReLU [7]
        # 4.1 - Conv2d [8]
        self.conv_block_4[1].conv_block[0].weight.data.copy_(resnet34_conv_weights[32].data)
        # 4.1 - BatchNorm2d [9]
        # 4.1 - ReLU [10]
        # 4.1 - Conv2d [11]
        self.conv_block_4[1].conv_block[3].weight.data.copy_(resnet34_conv_weights[33].data)
        # 4.1 - BatchNorm2d [12]
        # 4.1 - ReLU [13]
        # 4.2 - Conv2d [14]
        self.conv_block_4[2].conv_block[0].weight.data.copy_(resnet34_conv_weights[34].data)
        # 4.2 - BatchNorm2d [15]
        # 4.2 - ReLU [16]
        # 4.2 - Conv2d [17]
        self.conv_block_4[2].conv_block[3].weight.data.copy_(resnet34_conv_weights[35].data)
        # 4.2 - BatchNorm2d [18]
        # 4.2 - ReLU [19]
        ####################################################################################
        # AdaptiveAvgPool2d
        # Linear
        print('Done!')

        del resnet34_pretrained
        del resnet34_parameters
        del resnet34_conv_weights

    '''
    Transfer learning using weights of the BigGAN Discriminator
    '''
    def init_biggan_weights(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print('Loading Discriminator weights...')
        D = discriminator.Discriminator().to(device)
        D.load_state_dict(torch.load('networks/biggan/D48.pth'), strict = True)
        D.optim.load_state_dict(torch.load('networks/biggan/D48_optim.pth'))
        
        D_parameters = D.named_parameters()

        D_conv_weights = []

        # Picks up weights
        for _, (name, parameters) in enumerate(D_parameters):
            split_name = name.split('.')

            # Excludes non-convolutional layers
            if len(split_name) > 2:
                # Excludes the fifth convolutional block 
                if int(split_name[1]) < 4:
                    # On initial convolutional block, only second convolutional
                    if int(split_name[1]) == 0:
                        if split_name[3] == 'conv2':
                            if split_name[4] == 'weight':
                                D_conv_weights.append(parameters)
                    else:
                        # Excludes 'conv_sc' layers
                        if split_name[3] == 'conv1' or split_name[3] == 'conv2':
                            if split_name[4] == 'weight':
                                D_conv_weights.append(parameters)
        print('Done!')

        print('Transfering weights from the Discriminator to the classification network...')
        # 0 - Conv2d [0]
        # 0 - BatchNorm2d [1]
        # 0 - ReLU [2]
        # 0 - MaxPool2d [3]
        ############################################################################
        # 1.0 - Conv2d [0]
        self.conv_block_1[0].conv_block[0].weight.data.copy_(D_conv_weights[0].data)
        # 1.0 - BatchNorm2d [1]
        # 1.0 - ReLU [2]
        # 1.0 - Conv2d [3]
        self.conv_block_1[0].conv_block[3].weight.data.copy_(D_conv_weights[0].data)
        # 1.0 - BatchNorm2d [4]
        # 1.0 - ReLU [5]
        # 1.1 - Conv2d [6]
        self.conv_block_1[1].conv_block[0].weight.data.copy_(D_conv_weights[0].data)
        # 1.1 - BatchNorm2d [7]
        # 1.1 - ReLU [8]
        # 1.1 - Conv2d [9]
        self.conv_block_1[1].conv_block[3].weight.data.copy_(D_conv_weights[0].data)
        # 1.1 - BatchNorm2d [10]
        # 1.1 - ReLU [11]
        # 1.2 - Conv2d [12]
        self.conv_block_1[2].conv_block[0].weight.data.copy_(D_conv_weights[0].data)
        # 1.2 - BatchNorm2d [13]
        # 1.2 - ReLU [14]
        # 1.2 - Conv2d [15]
        self.conv_block_1[2].conv_block[3].weight.data.copy_(D_conv_weights[0].data)
        # 1.2 - BatchNorm2d [16]
        # 1.2 - ReLU [17]
        ############################################################################
        # 2.0 - Conv2d [0]
        self.conv_block_2[0].conv_block[0].weight.data.copy_(D_conv_weights[1].data)
        # 2.0 - BatchNorm2d [1]
        # 2.0 - ReLU [2]
        # 2.0 - Conv2d [3]
        self.conv_block_2[0].conv_block[3].weight.data.copy_(D_conv_weights[2].data)
        # 2.0 - BatchNorm2d [4]
        # 2.0 - Conv2d (downsample) [5]
        # 2.0 - BatchNorm2d (downsample) [6]
        # 2.0 - ReLU [7]
        # 2.1 - Conv2d [8]
        self.conv_block_2[1].conv_block[0].weight.data.copy_(D_conv_weights[2].data)
        # 2.1 - BatchNorm2d [9]
        # 2.1 - ReLU [10]
        # 2.1 - Conv2d [11]
        self.conv_block_2[1].conv_block[3].weight.data.copy_(D_conv_weights[2].data)
        # 2.1 - BatchNorm2d [12]
        # 2.1 - ReLU [13]
        # 2.2 - Conv2d [14]
        self.conv_block_2[2].conv_block[0].weight.data.copy_(D_conv_weights[2].data)
        # 2.2 - BatchNorm2d [15]
        # 2.2 - ReLU [16]
        # 2.2 - Conv2d [17]
        self.conv_block_2[2].conv_block[3].weight.data.copy_(D_conv_weights[2].data)
        # 2.2 - BatchNorm2d [18]
        # 2.2 - ReLU [19]
        # 2.3 - Conv2d [20]
        self.conv_block_2[3].conv_block[0].weight.data.copy_(D_conv_weights[2].data)
        # 2.3 - BatchNorm2d [21]
        # 2.3 - ReLU [22]
        # 2.3 - Conv2d [23]
        self.conv_block_2[3].conv_block[3].weight.data.copy_(D_conv_weights[2].data)
        # 2.3 - BatchNorm2d [24]
        # 2.3 - ReLU [25]
        ############################################################################
        # 3.0 - Conv2d [0]
        self.conv_block_3[0].conv_block[0].weight.data.copy_(D_conv_weights[3].data)
        # 3.0 - BatchNorm2d [1]
        # 3.0 - ReLU [2]
        # 3.0 - Conv2d [3]
        self.conv_block_3[0].conv_block[3].weight.data.copy_(D_conv_weights[4].data)
        # 3.0 - BatchNorm2d [4]
        # 3.0 - Conv2d (downsample) [5]
        # 3.0 - BatchNorm2d (downsample) [6]
        # 3.0 - ReLU [7]
        # 3.1 - Conv2d [8]
        self.conv_block_3[1].conv_block[0].weight.data.copy_(D_conv_weights[4].data)
        # 3.1 - BatchNorm2d [9]
        # 3.1 - ReLU [10]
        # 3.1 - Conv2d [11]
        self.conv_block_3[1].conv_block[3].weight.data.copy_(D_conv_weights[4].data)
        # 3.1 - BatchNorm2d [12]
        # 3.1 - ReLU [13]
        # 3.2 - Conv2d [14]
        self.conv_block_3[2].conv_block[0].weight.data.copy_(D_conv_weights[4].data)
        # 3.2 - BatchNorm2d [15]
        # 3.2 - ReLU [16]
        # 3.2 - Conv2d [17]
        self.conv_block_3[2].conv_block[3].weight.data.copy_(D_conv_weights[4].data)
        # 3.2 - BatchNorm2d [18]
        # 3.2 - ReLU [19]
        # 3.3 - Conv2d [20]
        self.conv_block_3[3].conv_block[0].weight.data.copy_(D_conv_weights[4].data)
        # 3.3 - BatchNorm2d [21]
        # 3.3 - ReLU [22]
        # 3.3 - Conv2d [23]
        self.conv_block_3[3].conv_block[3].weight.data.copy_(D_conv_weights[4].data)
        # 3.3 - BatchNorm2d [24]
        # 3.3 - ReLU [25]
        # 3.4 - Conv2d [26]
        self.conv_block_3[4].conv_block[0].weight.data.copy_(D_conv_weights[4].data)
        # 3.4 - BatchNorm2d [27]
        # 3.4 - ReLU [28]
        # 3.4 - Conv2d [29]
        self.conv_block_3[4].conv_block[3].weight.data.copy_(D_conv_weights[4].data)
        # 3.4 - BatchNorm2d [30]
        # 3.4 - ReLU [31]
        # 3.5 - Conv2d [32]
        self.conv_block_3[5].conv_block[0].weight.data.copy_(D_conv_weights[4].data)
        # 3.5 - BatchNorm2d [33]
        # 3.5 - ReLU [34]
        # 3.5 - Conv2d [35]
        self.conv_block_3[5].conv_block[3].weight.data.copy_(D_conv_weights[4].data)
        # 3.5 - BatchNorm2d [36]
        # 3.5 - ReLU [37]
        ############################################################################
        # 4.0 - Conv2d [0]
        self.conv_block_4[0].conv_block[0].weight.data.copy_(D_conv_weights[5].data)
        # 4.0 - BatchNorm2d [1]
        # 4.0 - ReLU [2]
        # 4.0 - Conv2d [3]
        self.conv_block_4[0].conv_block[3].weight.data.copy_(D_conv_weights[6].data)
        # 4.0 - BatchNorm2d [4]
        # 4.0 - Conv2d (downsample) [5]
        # 4.0 - BatchNorm2d (downsample) [6]
        # 4.0 - ReLU [7]
        # 4.1 - Conv2d [8]
        self.conv_block_4[1].conv_block[0].weight.data.copy_(D_conv_weights[6].data)
        # 4.1 - BatchNorm2d [9]
        # 4.1 - ReLU [10]
        # 4.1 - Conv2d [11]
        self.conv_block_4[1].conv_block[3].weight.data.copy_(D_conv_weights[6].data)
        # 4.1 - BatchNorm2d [12]
        # 4.1 - ReLU [13]
        # 4.2 - Conv2d [14]
        self.conv_block_4[2].conv_block[0].weight.data.copy_(D_conv_weights[6].data)
        # 4.2 - BatchNorm2d [15]
        # 4.2 - ReLU [16]
        # 4.2 - Conv2d [17]
        self.conv_block_4[2].conv_block[3].weight.data.copy_(D_conv_weights[6].data)
        # 4.2 - BatchNorm2d [18]
        # 4.2 - ReLU [19]
        ############################################################################
        # AdaptiveAvgPool2d
        # Linear
        print("Done!")

        del D
        del D_parameters
        del D_conv_weights

    def init_hybrid_weights(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print('Loading ResNet34 weights...')
        resnet34_pretrained = resnet34(pretrained = True).to(device)
        resnet34_parameters = resnet34_pretrained.named_parameters()

        resnet34_conv_weights = []

        # Picks up weights
        for _, (name, parameters) in enumerate(resnet34_parameters):
            split_name = name.split('.')

            # Convolutional layer of initial block
            if split_name[0] == 'conv1':
                resnet34_conv_weights.append(parameters)

            if len(split_name) > 2:
                # Convolutional block layers
                if split_name[2] == 'conv1' or split_name[2] == 'conv2':
                    resnet34_conv_weights.append(parameters)

                # Convolutional downsample layers
                if split_name[2] == 'downsample' and split_name[3] == '0':
                    resnet34_conv_weights.append(parameters)
        print('Done!')

        print('Loading Discriminator weights...')
        D = discriminator.Discriminator().to(device)
        D.load_state_dict(torch.load('networks/biggan/D48.pth'), strict = True)
        D.optim.load_state_dict(torch.load('networks/biggan/D48_optim.pth'))

        D_parameters = D.named_parameters()

        D_conv_weights = []

        # Picks up BigGAN weights
        for _, (name, parameters) in enumerate(D_parameters):
            split_name = name.split('.')

            # Excludes non-convolutional layers
            if len(split_name) > 2:
                # Excludes the fifth convolutional block 
                if int(split_name[1]) < 4:
                    # On initial convolutional block, only second convolutional
                    if int(split_name[1]) == 0:
                        if split_name[3] == 'conv2':
                            if split_name[4] == 'weight':
                                D_conv_weights.append(parameters)
                    else:
                        # Excludes 'conv_sc' layers
                        if split_name[3] == 'conv1' or split_name[3] == 'conv2':
                            if split_name[4] == 'weight':
                                D_conv_weights.append(parameters)
        print('Done!')

        print('Transfering weights from the ResNet34 and BigGAN Discriminator to the classification network...')
        # 0 - Conv2d [0]
        self.conv_block_0[0].weight.data.copy_(resnet34_conv_weights[0].data)
        # 0 - BatchNorm2d [1]
        # 0 - ReLU [2]
        # 0 - MaxPool2d [3]
        ############################################################################################################
        # 1.0 - Conv2d [0]
        self.conv_block_1[0].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[1].data, .5),
                                                                       torch.mul(D_conv_weights[0].data, .5)))
        # 1.0 - BatchNorm2d [1]
        # 1.0 - ReLU [2]
        # 1.0 - Conv2d [3]
        self.conv_block_1[0].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[2].data, .5),
                                                                       torch.mul(D_conv_weights[0].data, .5)))
        # 1.0 - BatchNorm2d [4]
        # 1.0 - ReLU [5]
        # 1.1 - Conv2d [6]
        self.conv_block_1[1].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[3].data, .5),
                                                                       torch.mul(D_conv_weights[0].data, .5)))
        # 1.1 - BatchNorm2d [7]
        # 1.1 - ReLU [8]
        # 1.1 - Conv2d [9]
        self.conv_block_1[1].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[4].data, .5),
                                                                       torch.mul(D_conv_weights[0].data, .5)))
        # 1.1 - BatchNorm2d [10]
        # 1.1 - ReLU [11]
        # 1.2 - Conv2d [12]
        self.conv_block_1[2].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[5].data, .5),
                                                                       torch.mul(D_conv_weights[0].data, .5)))
        # 1.2 - BatchNorm2d [13]
        # 1.2 - ReLU [14]
        # 1.2 - Conv2d [15]
        self.conv_block_1[2].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[6].data, .5),
                                                                       torch.mul(D_conv_weights[0].data, .5)))
        # 1.2 - BatchNorm2d [16]
        # 1.2 - ReLU [17]
        ############################################################################################################
        # 2.0 - Conv2d [0]
        self.conv_block_2[0].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[7].data, .5),
                                                                       torch.mul(D_conv_weights[1].data, .5)))
        # 2.0 - BatchNorm2d [1]
        # 2.0 - ReLU [2]
        # 2.0 - Conv2d [3]
        self.conv_block_2[0].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[8].data, .5),
                                                                       torch.mul(D_conv_weights[2].data, .5)))
        # 2.0 - BatchNorm2d [4]
        # 2.0 - Conv2d (downsample) [5]
        self.conv_block_2[0].downsample[0].weight.data.copy_(resnet34_conv_weights[9].data)
        # 2.0 - BatchNorm2d (downsample) [6]
        # 2.0 - ReLU [7]
        # 2.1 - Conv2d [8]
        self.conv_block_2[1].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[10].data, .5),
                                                                       torch.mul(D_conv_weights[2].data, .5)))
        # 2.1 - BatchNorm2d [9]
        # 2.1 - ReLU [10]
        # 2.1 - Conv2d [11]
        self.conv_block_2[1].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[11].data, .5),
                                                                       torch.mul(D_conv_weights[2].data, .5)))
        # 2.1 - BatchNorm2d [12]
        # 2.1 - ReLU [13]
        # 2.2 - Conv2d [14]
        self.conv_block_2[2].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[12].data, .5),
                                                                       torch.mul(D_conv_weights[2].data, .5)))
        # 2.2 - BatchNorm2d [15]
        # 2.2 - ReLU [16]
        # 2.2 - Conv2d [17]
        self.conv_block_2[2].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[13].data, .5),
                                                                       torch.mul(D_conv_weights[2].data, .5)))
        # 2.2 - BatchNorm2d [18]
        # 2.2 - ReLU [19]
        # 2.3 - Conv2d [20]
        self.conv_block_2[3].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[14].data, .5),
                                                                       torch.mul(D_conv_weights[2].data, .5)))
        # 2.3 - BatchNorm2d [21]
        # 2.3 - ReLU [22]
        # 2.3 - Conv2d [23]
        self.conv_block_2[3].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[15].data, .5),
                                                                       torch.mul(D_conv_weights[2].data, .5)))
        # 2.3 - BatchNorm2d [24]
        # 2.3 - ReLU [25]
        #############################################################################################################
        # 3.0 - Conv2d [0]
        self.conv_block_3[0].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[16].data, .5),
                                                                       torch.mul(D_conv_weights[3].data, .5)))
        # 3.0 - BatchNorm2d [1]
        # 3.0 - ReLU [2]
        # 3.0 - Conv2d [3]
        self.conv_block_3[0].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[17].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.0 - BatchNorm2d [4]
        # 3.0 - Conv2d (downsample) [5]
        self.conv_block_3[0].downsample[0].weight.data.copy_(resnet34_conv_weights[18].data)
        # 3.0 - BatchNorm2d (downsample) [6]
        # 3.0 - ReLU [7]
        # 3.1 - Conv2d [8]
        self.conv_block_3[1].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[19].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.1 - BatchNorm2d [9]
        # 3.1 - ReLU [10]
        # 3.1 - Conv2d [11]
        self.conv_block_3[1].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[20].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.1 - BatchNorm2d [12]
        # 3.1 - ReLU [13]
        # 3.2 - Conv2d [14]
        self.conv_block_3[2].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[21].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.2 - BatchNorm2d [15]
        # 3.2 - ReLU [16]
        # 3.2 - Conv2d [17]
        self.conv_block_3[2].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[22].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.2 - BatchNorm2d [18]
        # 3.2 - ReLU [19]
        # 3.3 - Conv2d [20]
        self.conv_block_3[3].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[23].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.3 - BatchNorm2d [21]
        # 3.3 - ReLU [22]
        # 3.3 - Conv2d [23]
        self.conv_block_3[3].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[24].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.3 - BatchNorm2d [24]
        # 3.3 - ReLU [25]
        # 3.4 - Conv2d [26]
        self.conv_block_3[4].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[25].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.4 - BatchNorm2d [27]
        # 3.4 - ReLU [28]
        # 3.4 - Conv2d [29]
        self.conv_block_3[4].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[26].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.4 - BatchNorm2d [30]
        # 3.4 - ReLU [31]
        # 3.5 - Conv2d [32]
        self.conv_block_3[5].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[27].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.5 - BatchNorm2d [33]
        # 3.5 - ReLU [34]
        # 3.5 - Conv2d [35]
        self.conv_block_3[5].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[28].data, .5),
                                                                       torch.mul(D_conv_weights[4].data, .5)))
        # 3.5 - BatchNorm2d [36]
        # 3.5 - ReLU [37]
        #############################################################################################################
        # 4.0 - Conv2d [0]
        self.conv_block_4[0].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[29].data, .5),
                                                                       torch.mul(D_conv_weights[5].data, .5)))
        # 4.0 - BatchNorm2d [1]
        # 4.0 - ReLU [2]
        # 4.0 - Conv2d [3]
        self.conv_block_4[0].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[30].data, .5),
                                                                       torch.mul(D_conv_weights[6].data, .5)))
        # 4.0 - BatchNorm2d [4]
        # 4.0 - Conv2d (downsample) [5]
        self.conv_block_4[0].downsample[0].weight.data.copy_(resnet34_conv_weights[31].data)
        # 4.0 - BatchNorm2d (downsample) [6]
        # 4.0 - ReLU [7]
        # 4.1 - Conv2d [8]
        self.conv_block_4[1].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[32].data, .5),
                                                                       torch.mul(D_conv_weights[6].data, .5)))
        # 4.1 - BatchNorm2d [9]
        # 4.1 - ReLU [10]
        # 4.1 - Conv2d [11]
        self.conv_block_4[1].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[33].data, .5),
                                                                       torch.mul(D_conv_weights[6].data, .5)))
        # 4.1 - BatchNorm2d [12]
        # 4.1 - ReLU [13]
        # 4.2 - Conv2d [14]
        self.conv_block_4[2].conv_block[0].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[34].data, .5),
                                                                       torch.mul(D_conv_weights[6].data, .5)))
        # 4.2 - BatchNorm2d [15]
        # 4.2 - ReLU [16]
        # 4.2 - Conv2d [17]
        self.conv_block_4[2].conv_block[3].weight.data.copy_(torch.add(torch.mul(resnet34_conv_weights[35].data, .5),
                                                                       torch.mul(D_conv_weights[6].data, .5)))
        # 4.2 - BatchNorm2d [18]
        # 4.2 - ReLU [19]
        #############################################################################################################
        # AdaptiveAvgPool2d
        # Linear
        print("Done!")

        del resnet34_pretrained
        del resnet34_parameters
        del resnet34_conv_weights
        del D
        del D_parameters
        del D_conv_weights

def resnet_34():
    return ResNet34(ResBlock, pretrained_weights = 'hybrid')