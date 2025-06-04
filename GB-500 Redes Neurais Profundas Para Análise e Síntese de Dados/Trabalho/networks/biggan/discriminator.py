#%% LIBRARIES

import functools
import torch
import torch.nn as nn
import torch.optim as optim

#%% LOCAL FILES

from . import layers

#%% SPATIAL TRANSFORM NET

def stn(image, transf_matrix, size):
    grid = torch.nn.functional.affine_grid(transf_matrix, torch.Size(size))
    out_image = torch.nn.functional.grid_sample(image, grid)

    return out_image

#%% DISCRIMINATOR ARCHITECTURES

def D_arch(ch=64, num_objects=5):
    arch = {}

    arch[112] = {'in_channels':  [3] + [ch*item for item in [1, 2, 4, 8]],
                 'out_channels': [item*ch for item in [1, 2, 4, 8, 16]],
                 'downsample': [True]*4 + [False],
                 'resolution': [56, 28, 14, 7, 7],
                 'attention': {56: False, 28: False, 14: False, 7: False, 7: False},

                 'in_channels_middle':  [3+(num_objects+1)] + [ch*item for item in [1]],
                 'out_channels_middle': [item*ch for item in [1, 2]],
                 'downsample_middle': [False]*2,
                 'resolution_middle': [28, 28]}

    return arch

#%% DISCRIMINATOR

class Discriminator(nn.Module):
    def __init__(self, D_ch=64, D_wide=True, resolution=112, D_kernel_size=3, n_classes=1,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, output_dim=1, D_param='SN',
                 use_mo=False, num_objects=5, use_object_pathway=False, **kwargs):
        super(Discriminator, self).__init__()
        self.ch = D_ch # width multiplier
        self.D_wide = D_wide # use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.resolution = resolution # Resolution
        self.kernel_size = D_kernel_size # kernel size
        self.n_classes = n_classes # number of classes
        self.activation = D_activation # activation
        self.D_param = D_param # parameterization style
        self.SN_eps = SN_eps # epsilon for Spectral Norm
        self.use_mo = use_mo # use multiple-objects generation?
        self.num_objects = num_objects # number of objects in the proposal
        self.use_object_pathway = use_object_pathway # use object pathway? 
        self.arch = D_arch(self.ch, num_objects=self.num_objects)[self.resolution] # architecture

        #%% LAYERS CONFIGURATION

        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            self.which_embedding = nn.Embedding

        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim) # linear output layer
        
        # Embedding for projection discrimination
        if self.n_classes > 1:
            self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        #%% BLOCKS CONFIGURATION

        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            if self.use_object_pathway:
                if index == 2:
                    self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index] + self.arch['out_channels_middle'][-1], # concat
                                                   out_channels=self.arch['out_channels'][index],
                                                   which_conv=self.which_conv,
                                                   wide=self.D_wide,
                                                   activation=self.activation,
                                                   preactivation=(index > 0),
                                                   downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
                elif index == 4:
                    self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index] + self.num_objects+1, # concat
                                                   out_channels=self.arch['out_channels'][index],
                                                   which_conv=self.which_conv,
                                                   wide=self.D_wide,
                                                   activation=self.activation,
                                                   preactivation=(index > 0),
                                                   downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
                else:
                    self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                                   out_channels=self.arch['out_channels'][index],
                                                   which_conv=self.which_conv,
                                                   wide=self.D_wide,
                                                   activation=self.activation,
                                                   preactivation=(index > 0),
                                                   downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            else:
                self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                               out_channels=self.arch['out_channels'][index],
                                               which_conv=self.which_conv,
                                               wide=self.D_wide,
                                               activation=self.activation,
                                               preactivation=(index > 0),
                                               downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # Residual blocks for object pathway of the multiple-object generation
        if self.use_object_pathway:
            self.blocks_middle = []
            for index in range(len(self.arch['out_channels_middle'])):
                self.blocks_middle += [[layers.DBlock(in_channels=self.arch['in_channels_middle'][index],
                                                      out_channels=self.arch['out_channels_middle'][index],
                                                      which_conv=self.which_conv,
                                                      wide=self.D_wide,
                                                      activation=self.activation,
                                                      preactivation=(index > 0),
                                                      downsample=(nn.AvgPool2d(2) if self.arch['downsample_middle'][index] else None))]]
            
            self.blocks_middle = nn.ModuleList([nn.ModuleList(block_middle) for block_middle in self.blocks_middle])

        #%% Optimizer

        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    #%% FORWARD

    def forward(self, x, y=None, transf_matrices=None, transf_matrices_inv=None, labels_one_hot=None, num_objects=None):
        h = x # stick x into h for cleaner for loops without flow control

        if self.use_object_pathway:
            # Empty canvas on which the features are added at the locations given by the bounding-box
            h_locals_middle = torch.cuda.FloatTensor(h.shape[0],
                                                     self.arch['out_channels_middle'][-1],
                                                     self.arch['resolution_middle'][-1], self.arch['resolution_middle'][-1]).fill_(0)

            for i in range(num_objects):
                # Get bounding-box label and replicate spatially
                curr_labels_one_hot = labels_one_hot[:, i].view(labels_one_hot.shape[0], num_objects+1, 1, 1)
                curr_labels_one_hot = curr_labels_one_hot.repeat(1, 1, self.arch['resolution_middle'][-1], self.arch['resolution_middle'][-1])

                # Extract features from bounding-box and concatenate with the bounding-box label
                h_local_middles = []
                for j, transf_matrix in enumerate(transf_matrices):
                    h_local_middle = stn(h, transf_matrix[:, i],
                                         (h.shape[0], h.shape[1], self.arch['resolution_middle'][-1], self.arch['resolution_middle'][-1]))
                    h_local_middle = torch.cat((h_local_middle, curr_labels_one_hot), 1)
                    h_local_middles.append(h_local_middle)

                # Loop over blocks
                for blocklist_middle in self.blocks_middle:
                    for block_middle in blocklist_middle:
                        for j, h_local_middle in enumerate(h_local_middles):
                            h_local_middles[j] = block_middle(h_local_middle)

                # Reshape extracted features to bounding-box layout and add to empty canvas
                for j, transf_matrix_inv in enumerate(transf_matrices_inv):
                    h_local_middles[j] = stn(h_local_middles[j], transf_matrix_inv[:, i],
                                             (h_local_middles[j].shape[0], h_local_middles[j].shape[1], self.arch['resolution_middle'][-1], self.arch['resolution_middle'][-1]))
                    h_locals_middle += h_local_middles[j]

        if self.use_mo:
            labels_one_hot = labels_one_hot.detach()

            sum_all_loh = torch.cuda.FloatTensor(labels_one_hot.shape[0], self.num_objects+1).fill_(0)
            for i in range(num_objects):
                sum_all_loh += labels_one_hot[:, i, :]

            for i in range(sum_all_loh.shape[0]):
                for j in range(1, self.num_objects+1):
                    if sum_all_loh[i][j] > 1:
                        sum_all_loh[i][j] = 1
                        sum_all_loh[i][0] += 1

            sum_all_loh = sum_all_loh.view(-1, self.num_objects+1, 1, 1)
            sum_all_loh = sum_all_loh.repeat(1, 1, self.arch['resolution'][4], self.arch['resolution'][4])

        for index, blocklist in enumerate(self.blocks):
            if self.use_object_pathway:
                if index == 2:
                    h = torch.cat((h, h_locals_middle), 1) # combine global and local pathway
                if index == 4:
                    h = torch.cat((h, sum_all_loh), 1) # combine global pathway and sum of all labels one-hot

            for block in blocklist:
                h = block(h)

        h = torch.sum(self.activation(h), [2, 3]) # apply global sum pooling as in SN-GAN
        out = self.linear(h) # get initial class-unconditional output

        # Get projection of final featureset onto class vectors and add to evidence
        if self.n_classes > 1:
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)

        return out