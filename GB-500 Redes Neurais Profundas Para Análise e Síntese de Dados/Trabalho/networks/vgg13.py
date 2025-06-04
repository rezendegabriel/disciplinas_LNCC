#%% LIBRARIES

#from torchvision.models import vgg13_bn as vgg13_bn_arch
#from torchvision.models import VGG13_BN_Weights
from torchvision.models import vgg13 as vgg13_arch
from torchvision.models import VGG13_Weights

import torch
import torch.nn as nn

#%% LOCAL FILES

from networks.biggan import discriminator

#%% ORIGINAL VGG13 ARCHITECTURE

class VGG13(nn.Module):
    def __init__(self, pretrained_weights='VGG13'):
        super(VGG13, self).__init__()

        self.features_depth = 7
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_depth*self.features_depth*512, 4096),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(4096, 2)
        )

        if pretrained_weights == 'VGG13':
            self.init_vgg13_weights()

        if pretrained_weights == 'BigGAN':
            self.init_biggan_weights()
        
        if pretrained_weights == 'hybrid':
            self.init_hybrid_weights()

    def forward(self, input):
        out = self.conv_block_1(input)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = self.conv_block_5(out)

        out = out.view(-1, self.features_depth*self.features_depth*512)
        out = self.fc(out)

        return out
    
    #%% TRANSFER LEARNING (IMAGENET WEIGHTS)

    def init_vgg13_weights(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print('Loading VGG13 weights...', end=' ')
        #vgg13_pretrained = vgg13_bn_arch(weights=VGG13_BN_Weights.IMAGENET1K_V1).to(device)
        vgg13_pretrained = vgg13_arch(weights=VGG13_Weights.IMAGENET1K_V1).to(device)
        print('done!')

        print('Transfering weights from the VGG13 to the classification network...', end=' ')
        # 1 - Conv2d [0]
        self.conv_block_1[0].weight.data.copy_(vgg13_pretrained.features[0].weight.data)
        self.conv_block_1[0].bias.data.copy_(vgg13_pretrained.features[0].bias.data)
        # 1 - BatchNorm2d [1]
        #self.conv_block_1[1].weight.data.copy_(vgg13_pretrained.features[1].weight.data)
        #self.conv_block_1[1].bias.data.copy_(vgg13_pretrained.features[1].bias.data)
        # 1 - ReLU [2]
        # 1 - Conv2d [3]
        self.conv_block_1[3].weight.data.copy_(vgg13_pretrained.features[2].weight.data)
        self.conv_block_1[3].bias.data.copy_(vgg13_pretrained.features[2].bias.data)
        # 1 - BatchNorm2d [4]
        #self.conv_block_1[4].weight.data.copy_(vgg13_pretrained.features[4].weight.data)
        #self.conv_block_1[4].bias.data.copy_(vgg13_pretrained.features[4].bias.data)
        # 1 - ReLU [5]
        # 1 - MaxPool2d [6]

        # 2 - Conv2d [0]
        self.conv_block_2[0].weight.data.copy_(vgg13_pretrained.features[5].weight.data)
        self.conv_block_2[0].bias.data.copy_(vgg13_pretrained.features[5].bias.data)
        # 2 - BatchNorm2d [1]
        #self.conv_block_2[1].weight.data.copy_(vgg13_pretrained.features[8].weight.data)
        #self.conv_block_2[1].bias.data.copy_(vgg13_pretrained.features[8].bias.data)
        # 2 - ReLU [2]
        # 2 - Conv2d [3]
        self.conv_block_2[3].weight.data.copy_(vgg13_pretrained.features[7].weight.data)
        self.conv_block_2[3].bias.data.copy_(vgg13_pretrained.features[7].bias.data)
        # 2 - BatchNorm2d [4]
        #self.conv_block_2[4].weight.data.copy_(vgg13_pretrained.features[11].weight.data)
        #self.conv_block_2[4].bias.data.copy_(vgg13_pretrained.features[11].bias.data)
        # 2 - ReLU [5]
        # 2 - MaxPool2d [6]

        # 3 - Conv2d [0]
        self.conv_block_3[0].weight.data.copy_(vgg13_pretrained.features[10].weight.data)
        self.conv_block_3[0].bias.data.copy_(vgg13_pretrained.features[10].bias.data)
        # 3 - BatchNorm2d [1]
        #self.conv_block_3[1].weight.data.copy_(vgg13_pretrained.features[15].weight.data)
        #self.conv_block_3[1].bias.data.copy_(vgg13_pretrained.features[15].bias.data)
        # 3 - ReLU [2]
        # 3 - Conv2d [3]
        self.conv_block_3[3].weight.data.copy_(vgg13_pretrained.features[12].weight.data)
        self.conv_block_3[3].bias.data.copy_(vgg13_pretrained.features[12].bias.data)
        # 3 - BatchNorm2d [4]
        #self.conv_block_3[4].weight.data.copy_(vgg13_pretrained.features[18].weight.data)
        #self.conv_block_3[4].bias.data.copy_(vgg13_pretrained.features[18].bias.data)
        # 3 - ReLU [5]
        # 3 - MaxPool2d [6]

        # 4 - Conv2d [0]
        self.conv_block_4[0].weight.data.copy_(vgg13_pretrained.features[15].weight.data)
        self.conv_block_4[0].bias.data.copy_(vgg13_pretrained.features[15].bias.data)
        # 4 - BatchNorm2d [1]
        #self.conv_block_4[1].weight.data.copy_(vgg13_pretrained.features[22].weight.data)
        #self.conv_block_4[1].bias.data.copy_(vgg13_pretrained.features[22].bias.data)
        # 4 - ReLU [2]
        # 4 - Conv2d [3]
        self.conv_block_4[3].weight.data.copy_(vgg13_pretrained.features[17].weight.data)
        self.conv_block_4[3].bias.data.copy_(vgg13_pretrained.features[17].bias.data)
        # 4 - BatchNorm2d [4]
        #self.conv_block_4[4].weight.data.copy_(vgg13_pretrained.features[25].weight.data)
        #self.conv_block_4[4].bias.data.copy_(vgg13_pretrained.features[25].bias.data)
        # 4 - ReLU [5]
        # 4 - MaxPool2d [6]

        # 5 - Conv2d [0]
        self.conv_block_5[0].weight.data.copy_(vgg13_pretrained.features[20].weight.data)
        self.conv_block_5[0].bias.data.copy_(vgg13_pretrained.features[20].bias.data)
        # 5 - BatchNorm2d [1]
        #self.conv_block_5[1].weight.data.copy_(vgg13_pretrained.features[29].weight.data)
        #self.conv_block_5[1].bias.data.copy_(vgg13_pretrained.features[29].bias.data)
        # 5 - ReLU [2]
        # 5 - Conv2d [3]
        self.conv_block_5[3].weight.data.copy_(vgg13_pretrained.features[22].weight.data)
        self.conv_block_5[3].bias.data.copy_(vgg13_pretrained.features[22].bias.data)
        # 5 - BatchNorm2d [4]
        #self.conv_block_5[4].weight.data.copy_(vgg13_pretrained.features[32].weight.data)
        #self.conv_block_5[4].bias.data.copy_(vgg13_pretrained.features[32].bias.data)
        # 5 - ReLU [5]
        # 5 - MaxPool2d [6]
        print('done!')
        
        del vgg13_pretrained

    #%% TRANSFER LEARNING (BIGAN DISCRIMINATOR)

    def init_biggan_weights(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print('Loading Discriminator weights...', end=' ')
        D = discriminator.Discriminator().to(device)
        D.load_state_dict(torch.load('networks/biggan/D_best.pth'), strict=True)
        D.optim.load_state_dict(torch.load('networks/biggan/D_optim_best.pth'))

        D_parameters = D.named_parameters()

        D_conv_weights = []
        D_conv_biases = []

        # Picks up weights and biases
        for _, (name, parameters) in enumerate(D_parameters):
            split_name = name.split('.')

            # Excludes non-convolutional layers
            if len(split_name) > 2:
                # Excludes 'conv_sc' layers (skip-connections)
                if split_name[3] == 'conv1' or split_name[3] == 'conv2':
                    if split_name[4] == 'weight':
                        D_conv_weights.append(parameters)
                    if split_name[4] == 'bias':
                        D_conv_biases.append(parameters)
        print('done!')

        print('Transfering weights from the Discriminator to the classification network...', end=' ')
        # 1 - Conv2d [0]
        self.conv_block_1[0].weight.data.copy_(D_conv_weights[0].data)
        self.conv_block_1[0].bias.data.copy_(D_conv_biases[0].data)
        # 1 - BatchNorm2d [1]
        # 1 - ReLU [2]
        # 1 - Conv2d [3]
        self.conv_block_1[2].weight.data.copy_(D_conv_weights[1].data)
        self.conv_block_1[2].bias.data.copy_(D_conv_biases[1].data)
        # 1 - BatchNorm2d [4]
        # 1 - ReLU [5]
        # 1 - MaxPool2d [6]

        # 2 - Conv2d [0]
        self.conv_block_2[0].weight.data.copy_(D_conv_weights[2].data)
        self.conv_block_2[0].bias.data.copy_(D_conv_biases[2].data)
        # 2 - BatchNorm2d [1]
        # 2 - ReLU [2]
        # 2 - Conv2d [3]
        self.conv_block_2[2].weight.data.copy_(D_conv_weights[3].data)
        self.conv_block_2[2].bias.data.copy_(D_conv_biases[3].data)
        # 2 - BatchNorm2d [4]
        # 2 - ReLU [5]
        # 2 - MaxPool2d [6]

        # 3 - Conv2d [0]
        self.conv_block_3[0].weight.data.copy_(D_conv_weights[4].data[:, :128, :, :]) # BigGAN_BMs
        #self.conv_block_3[0].weight.data.copy_(D_conv_weights[4].data) # original BigGAN
        self.conv_block_3[0].bias.data.copy_(D_conv_biases[4].data)
        # 3 - BatchNorm2d [1]
        # 3 - ReLU [2]
        # 3 - Conv2d [3]
        self.conv_block_3[2].weight.data.copy_(D_conv_weights[5].data)
        self.conv_block_3[2].bias.data.copy_(D_conv_biases[5].data)
        # 3 - BatchNorm2d [4]
        # 3 - ReLU [5]
        # 3 - MaxPool2d [6]

        # 4 - Conv2d [0]
        self.conv_block_4[0].weight.data.copy_(D_conv_weights[6].data)
        self.conv_block_4[0].bias.data.copy_(D_conv_biases[6].data)
        # 4 - BatchNorm2d [1]
        # 4 - ReLU [2]
        # 4 - Conv2d [3]
        self.conv_block_4[2].weight.data.copy_(D_conv_weights[7].data)
        self.conv_block_4[2].bias.data.copy_(D_conv_biases[7].data)
        # 4 - BatchNorm2d [4]
        # 4 - ReLU [5]
        # 4 - MaxPool2d [6]

        # 5 - Conv2d [0]
        self.conv_block_5[0].weight.data.copy_(D_conv_weights[7].data)
        self.conv_block_5[0].bias.data.copy_(D_conv_biases[7].data)
        # 5 - BatchNorm2d [1]
        # 5 - ReLU [2]
        # 5 - Conv2d [3]
        self.conv_block_5[2].weight.data.copy_(D_conv_weights[7].data)
        self.conv_block_5[2].bias.data.copy_(D_conv_biases[7].data)
        # 5 - BatchNorm2d [4]
        # 5 - ReLU [5]
        # 5 - MaxPool2d [6]
        print('done!')

        del D
        del D_parameters
        del D_conv_weights
        del D_conv_biases
    
    #%% TRANSFER LEARNING (BIGAN DISCRIMINATOR C2D AND IMAGENET BN WEIGHTS)

    def init_hybrid_weights(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print('Loading Discriminator weights...', end=' ')
        D = discriminator.Discriminator().to(device)
        D.load_state_dict(torch.load('networks/biggan/D_best.pth'), strict=True)
        D.optim.load_state_dict(torch.load('networks/biggan/D_optim_best.pth'))

        D_parameters = D.named_parameters()

        D_conv_weights = []
        D_conv_biases = []

        # Picks up weights and biases
        for _, (name, parameters) in enumerate(D_parameters):
            split_name = name.split('.')

            # Excludes non-convolutional layers
            if len(split_name) > 2:
                # Excludes 'conv_sc' layers (skip-connections)
                if split_name[3] == 'conv1' or split_name[3] == 'conv2':
                    if split_name[4] == 'weight':
                        D_conv_weights.append(parameters)
                    if split_name[4] == 'bias':
                        D_conv_biases.append(parameters)
        print('done!')

        print('Loading VGG13 weights...', end=' ')
        vgg13_pretrained = vgg13_bn(weights=VGG13_BN_Weights.IMAGENET1K_V1).to(device)
        print('done!')

        print('Transfering weights from the VGG13 to the classification network...', end=' ')
        # 1 - Conv2d [0]
        self.conv_block_1[0].weight.data.copy_(D_conv_weights[0].data)
        self.conv_block_1[0].bias.data.copy_(D_conv_biases[0].data)
        # 1 - BatchNorm2d [1]
        self.conv_block_1[1].weight.data.copy_(vgg13_pretrained.features[1].weight.data)
        self.conv_block_1[1].bias.data.copy_(vgg13_pretrained.features[1].bias.data)
        # 1 - ReLU [2]
        # 1 - Conv2d [3]
        self.conv_block_1[3].weight.data.copy_(D_conv_weights[1].data)
        self.conv_block_1[3].bias.data.copy_(D_conv_biases[1].data)
        # 1 - BatchNorm2d [4]
        self.conv_block_1[4].weight.data.copy_(vgg13_pretrained.features[4].weight.data)
        self.conv_block_1[4].bias.data.copy_(vgg13_pretrained.features[4].bias.data)
        # 1 - ReLU [5]
        # 1 - MaxPool2d [6]

        # 2 - Conv2d [0]
        self.conv_block_2[0].weight.data.copy_(D_conv_weights[2].data)
        self.conv_block_2[0].bias.data.copy_(D_conv_biases[2].data)
        # 2 - BatchNorm2d [1]
        self.conv_block_2[1].weight.data.copy_(vgg13_pretrained.features[8].weight.data)
        self.conv_block_2[1].bias.data.copy_(vgg13_pretrained.features[8].bias.data)
        # 2 - ReLU [2]
        # 2 - Conv2d [3]
        self.conv_block_2[3].weight.data.copy_(D_conv_weights[3].data)
        self.conv_block_2[3].bias.data.copy_(D_conv_biases[3].data)
        # 2 - BatchNorm2d [4]
        self.conv_block_2[4].weight.data.copy_(vgg13_pretrained.features[11].weight.data)
        self.conv_block_2[4].bias.data.copy_(vgg13_pretrained.features[11].bias.data)
        # 2 - ReLU [5]
        # 2 - MaxPool2d [6]

        # 3 - Conv2d [0]
        #self.conv_block_3[0].weight.data.copy_(D_conv_weights[4].data[:, :128, :, :]) # BigGAN_BMs
        self.conv_block_3[0].weight.data.copy_(D_conv_weights[4].data) # original BigGAN
        self.conv_block_3[0].bias.data.copy_(D_conv_biases[4].data)
        # 3 - BatchNorm2d [1]
        self.conv_block_3[1].weight.data.copy_(vgg13_pretrained.features[15].weight.data)
        self.conv_block_3[1].bias.data.copy_(vgg13_pretrained.features[15].bias.data)
        # 3 - ReLU [2]
        # 3 - Conv2d [3]
        self.conv_block_3[3].weight.data.copy_(D_conv_weights[5].data)
        self.conv_block_3[3].bias.data.copy_(D_conv_biases[5].data)
        # 3 - BatchNorm2d [4]
        self.conv_block_3[4].weight.data.copy_(vgg13_pretrained.features[18].weight.data)
        self.conv_block_3[4].bias.data.copy_(vgg13_pretrained.features[18].bias.data)
        # 3 - ReLU [5]
        # 3 - MaxPool2d [6]

        # 4 - Conv2d [0]
        self.conv_block_4[0].weight.data.copy_(D_conv_weights[6].data)
        self.conv_block_4[0].bias.data.copy_(D_conv_biases[6].data)
        # 4 - BatchNorm2d [1]
        self.conv_block_4[1].weight.data.copy_(vgg13_pretrained.features[22].weight.data)
        self.conv_block_4[1].bias.data.copy_(vgg13_pretrained.features[22].bias.data)
        # 4 - ReLU [2]
        # 4 - Conv2d [3]
        self.conv_block_4[3].weight.data.copy_(D_conv_weights[7].data)
        self.conv_block_4[3].bias.data.copy_(D_conv_biases[7].data)
        # 4 - BatchNorm2d [4]
        self.conv_block_4[4].weight.data.copy_(vgg13_pretrained.features[25].weight.data)
        self.conv_block_4[4].bias.data.copy_(vgg13_pretrained.features[25].bias.data)
        # 4 - ReLU [5]
        # 4 - MaxPool2d [6]

        # 5 - Conv2d [0]
        self.conv_block_5[0].weight.data.copy_(D_conv_weights[7].data)
        self.conv_block_5[0].bias.data.copy_(D_conv_biases[7].data)
        # 5 - BatchNorm2d [1]
        self.conv_block_5[1].weight.data.copy_(vgg13_pretrained.features[29].weight.data)
        self.conv_block_5[1].bias.data.copy_(vgg13_pretrained.features[29].bias.data)
        # 5 - ReLU [2]
        # 5 - Conv2d [3]
        self.conv_block_5[3].weight.data.copy_(D_conv_weights[7].data)
        self.conv_block_5[3].bias.data.copy_(D_conv_biases[7].data)
        # 5 - BatchNorm2d [4]
        self.conv_block_5[4].weight.data.copy_(vgg13_pretrained.features[32].weight.data)
        self.conv_block_5[4].bias.data.copy_(vgg13_pretrained.features[32].bias.data)
        # 5 - ReLU [5]
        # 5 - MaxPool2d [6]
        print('done!')
        
        del vgg13_pretrained

def vgg13():
    return VGG13(), ''