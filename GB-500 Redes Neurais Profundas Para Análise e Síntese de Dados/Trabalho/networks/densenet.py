# -*- coding: utf-8 -*-

import torch

from networks.biggan import discriminator

def densenet121(pretrained_weights = 'DenseNet'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Loading DenseNet121 architecture...')
    densenet121_scratch = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained = False).to(device)
    densenet121_scratch.classifier = torch.nn.Linear(1024, 2)
    print('Done!')

    if pretrained_weights == 'DenseNet':
        densenet121_scratch_params = densenet121_scratch.named_parameters()

        print('Loading DenseNet121 weights...')
        densenet121_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained = True).to(device)
        densenet121_pretrained_params = densenet121_pretrained.named_parameters()

        densenet121_weights = []

        # Picks up weights
        for (_, params) in densenet121_pretrained_params:
            densenet121_weights.append(params.data)
        print('Done!')

        print('Transfering weights from the DenseNet121 to the classification network...')
        for l, (name, params) in enumerate(densenet121_scratch_params):
            split_name = name.split('.')

            if 'conv' in split_name[1] or 'norm' in split_name[1]:
                params.data.copy_(densenet121_weights[l].data)
            if len(split_name) > 3:
                if 'conv' in split_name[2] or 'norm' in split_name[2]:
                    params.data.copy_(densenet121_weights[l].data)
                if 'conv' in split_name[3] or 'norm' in split_name[3]:
                    params.data.copy_(densenet121_weights[l].data)
        print('Done!')

    return densenet121_scratch, pretrained_weights