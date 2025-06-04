#%% LIBRARIES

from networks import vgg13, resnet34, densenet

#%% NETWORKS

nets = {'vgg13': vgg13.vgg13,
        'resnet34': resnet34.resnet_34,
        'densenet121': densenet.densenet121}

def setup_network(hps):
    net, pretrained_weights = nets[hps['network']]()

    return net, pretrained_weights