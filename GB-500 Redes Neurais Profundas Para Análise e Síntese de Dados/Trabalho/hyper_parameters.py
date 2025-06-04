hps = {
    'network': '', # which network do you want to train
    'epochs': 50,
    'bs': 16, # batchsize
    'input_size': 224, # input size
    'processes': 8, # number of workers of DataLoader function
    'lr': .005,
    'freeze_layers': 1, # freeze (> 0) or not (0) convolutional layers
    'cross_val': 1,
    'test': '',
    'data_split': '',
    'n_splits': 3,
    'verbose': 5
}

def setup_hparams(args):
    for arg in args:
        key, value = arg.split('=')
        if key not in hps:
            raise ValueError(key + ' is not a valid hyper parameter')
        else:
            hps[key] = value

    # Invalid parameter check
    try:
        hps['epochs'] = int(hps['epochs'])
        hps['bs'] = int(hps['bs'])
        hps['input_size'] = int(hps['input_size'])
        hps['processes'] = int(hps['processes'])
        hps['lr'] = float(hps['lr'])
        hps['freeze_layers'] = int(hps['freeze_layers'])
        hps['cross_val'] = int(hps['cross_val'])
        hps['data_split'] = int(hps['data_split'])
        hps['n_splits'] = int(hps['n_splits'])
        hps['verbose'] = int(hps['verbose'])
        
    except Exception as e:
        raise ValueError('Invalid input parameters')

    return hps
