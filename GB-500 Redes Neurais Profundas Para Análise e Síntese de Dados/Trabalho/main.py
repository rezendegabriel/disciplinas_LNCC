# %% LIBRARIES AND HYPER-PARAMETERS

from data_loaders import Plain_Dataset
from hyper_parameters import setup_hparams
from networks.setup_network import setup_network
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler

import numpy  as np
import os
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import warnings

warnings.filterwarnings('ignore')

hps = setup_hparams(sys.argv[1:]) # hyper-parameters

epochs, batch_size, lr, test = hps['epochs'], hps['bs'], hps['lr'], hps['test'] # model variables
data_split, n_splits =  hps['data_split'], hps['n_splits'] # cross validation
net_name = hps['network'] # net name

img_size = hps['input_size'] # image size

freeze_layers = hps['freeze_layers'] # freeze or not convolutional layers
cross_val = hps['cross_val'] # cross validation
verbose = hps['verbose']

data_path = 'data/cross_validation_{}/data_split_{}'.format(cross_val, data_split) # path of the data
classes_names = ['girolando', 'holandes'] # classes names
oversampling = False # oversampling

# set the processing device to GPU, if it exists
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# memory usage
memory_usage = []

# %% TRAINING HISTORY

train_acc_hist = []
val_acc_hist = []
train_loss_hist = []
val_loss_hist = []

val_acc_class_hist = []

for split in range(n_splits):
    train_acc_hist.append([])
    val_acc_hist.append([])
    train_loss_hist.append([])
    val_loss_hist.append([])

    for label in range(len(classes_names)):
        val_acc_class_hist.append([])

# %% TRAINING LOOP

def Train(net, epochs, train_loader, val_loader, split):
    print('======================================== Training start ========================================')
    
    for epoch in range(epochs):
        t_start = time.time()
        
        train_correct = 0
        val_correct = 0
        train_loss = 0
        val_loss = 0
        val_correct_class = [] 
        n_samples = 0
        iter = 0

        for label in range(len(classes_names)):
            val_correct_class.append(0)
        
        # training
        net.train()
        for data, labels, _ in train_loader:
            # moves the data and labels to the GPU
            data, labels = data.to(device), labels.to(device)

            with autocast():
                # forward + backward + optimize
                mu = torch.cuda.memory_allocated()*1e-9
                memory_usage.append(mu)
                if verbose == 6:
                    print('Train iteration: {} || GPU Memory usage before forward: {:.2f} GB'.format(iter, mu))

                outputs = net(data)

                mu = torch.cuda.memory_allocated()*1e-9
                memory_usage.append(mu)
                if verbose == 6:
                    print('Train iteration: {} || GPU Memory usage after forward: {:.2f} GB'.format(iter, mu))

                loss = criterion(outputs, labels)

                mu = torch.cuda.memory_allocated()*1e-9
                memory_usage.append(mu)
                if verbose == 6:
                    print('Train iteration: {} || GPU Memory usage before backward: {:.2f} GB'.format(iter, mu))
                
                scaler.scale(loss).backward()

                mu = torch.cuda.memory_allocated()*1e-9
                memory_usage.append(mu)
                if verbose == 6:
                    print('Train iteration: {} || GPU Memory usage after backward: {:.2f} GB'.format(iter, mu))

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none = True)

                # performance metrics
                train_loss += float(loss.item())
                
                _, preds = torch.max(outputs, 1)
                train_correct += float(torch.sum(preds == labels.data).item())

                n_samples += labels.size(0)

                iter+=1

                del data
                del labels
                del outputs
                del loss
                del preds

                mu = torch.cuda.memory_allocated()*1e-9
                memory_usage.append(mu)
                if verbose == 6:
                    print('Train iteration: {} || GPU Memory usage: {:.2f} GB'.format(iter, mu))
        
        # update learning rate
        exp_lr_scheduler.step()

        train_acc = train_correct/n_samples
        train_loss = train_loss/n_samples
            
        train_loss_hist[split-1].append(train_loss)
        train_acc_hist[split-1].append(train_acc)

        iter = 0
        n_samples = 0
        
        # disables gradient calculation
        with torch.no_grad():
            # validation
            net.eval()
            for data, labels, _ in val_loader:
                # moves the data and labels to the GPU
                data, labels = data.to(device), labels.to(device)

                # forward
                outputs = net(data)

                # backward
                loss = criterion(outputs, labels)

                # performance metrics
                val_loss += float(loss.item())
                
                _, preds = torch.max(outputs, 1)
                val_correct += float(torch.sum(preds == labels.data).item())

                n_samples += labels.size(0)

                for i in range(len(labels.data)):
                    val_correct_class[labels.data[i]] += float(torch.sum(preds[i] == labels.data[i]).item())
                
                iter+=1
                
                del data
                del labels
                del outputs
                del loss
                del preds
                
                mu = torch.cuda.memory_allocated()*1e-9
                memory_usage.append(mu)
                if verbose == 6:
                    print('Val iteration: {} || GPU Memory usage: {:.2f} GB'.format(iter, mu))
            
            val_acc = val_correct/n_samples
            val_loss = val_loss/n_samples
            
            val_acc_hist[split-1].append(val_acc)
            val_loss_hist[split-1].append(val_loss)

            for label in range(len(classes_names)):
                val_correct_class[label] = val_correct_class[label]/counts_val[label]
                
                val_acc_class_hist[label+((split-1)*len(classes_names))].append(val_correct_class[label])

        if verbose == 1: # epoch
            print('Epoch: {}'.format(epoch+1))
        elif verbose == 2: # epochs and loss
            print('Epoch: {} || '
                  'Train Loss: {:.4f} | '
                  'Val Loss: {:.4f}'.format(epoch+1,
                                            train_loss, val_loss))
        elif verbose == 3: # epoch, loss and accuracy
            print('Epoch: {} || '
                  'Train Loss: {:.4f} | '
                  'Val Loss: {:.4f} || '
                  'Train Accuracy: {:.4f}% | '
                  'Val Accuracy: {:.4f}%'.format(epoch+1,
                                                 train_loss, val_loss,
                                                 train_acc*100, val_acc*100))
        elif verbose == 4: # epoch, loss, accuracy and time
            print('Epoch: {} || '
                  'Train Loss: {:.4f} | '
                  'Val Loss: {:.4f} || '
                  'Train Accuracy: {:.4f}% | '
                  'Val Accuracy: {:.4f}% || '
                  'Time: {:.2f}s'.format(epoch+1,
                                         train_loss, val_loss,
                                         train_acc*100, val_acc*100,
                                         time.time()-t_start))
        elif verbose >= 5: # epoch, loss, accuracy, time and GPU memory usage
            mu = torch.cuda.memory_allocated()*1e-9 # byte yo gigabyte
            memory_usage.append(mu)

            print('Epoch: {} || '
                  'Train Loss: {:.4f} | '
                  'Val Loss: {:.4f} || '
                  'Train Accuracy: {:.4f}% | '
                  'Val Accuracy: {:.4f}% || '
                  'Time: {:.2f}s || ' 
                  'GPU memory usage: {:.2f} GB'.format(epoch+1,
                                                       train_loss, val_loss,
                                                       train_acc*100, val_acc*100,
                                                       time.time()-t_start,
                                                       mu))

    if not os.path.exists('models'):
        os.mkdir('models')
        
        torch.save(net.state_dict(), 'models/{}-{}-{}-{}-{}-{}-{}.pt'.format(net_name, epochs, batch_size, lr, test, data_split, split))
    else:
        torch.save(net.state_dict(), 'models/{}-{}-{}-{}-{}-{}-{}.pt'.format(net_name, epochs, batch_size, lr, test, data_split, split))

    print('======================================== Training finished ========================================')

# %% DATA AUGMENTATION

mean, std = 0.5, 0.5

norm = T.Compose([
    T.Resize((img_size, img_size)),
    
    T.ToTensor(),
    T.Normalize(mean = (mean, mean, mean), std = (std, std, std))
])

transform = T.Compose([
    T.Resize((img_size, img_size)),

    T.RandomHorizontalFlip(),
    T.RandomRotation(4), 
    T.RandomPerspective(distortion_scale = 0.4),
    
    T.ToTensor(),
    T.Normalize(mean = (mean, mean, mean), std = (std, std, std))
])

# %% CROSS VALIDATION

for split in range(1, n_splits+1):

    # %% MODEL

    # model architecture
    net, pretrained_weights = setup_network(hps)

    print('Model architecture:')
    print(net)

    if freeze_layers > 0:
        # VGG's freeze layers
        if 'vgg' in hps['network']:
            for _, (name, param) in enumerate(net.named_parameters()):
                split_name = name.split('.')
                
                # Excludes non-convolutional layers
                if 'conv' in split_name[0]:
                    if freeze_layers == 1:
                        if '1' in split_name[0]:
                        # Only convolutional layers
                            if split_name[1] == '0' or split_name[1] == '3':
                                param.requires_grad = False

        # ResNet's freeze layers
        if 'resnet' in hps['network']:
            for _, (name, params) in enumerate(net.named_parameters()):
                split_name = name.split('.')

                if 'ResNet' in pretrained_weights or pretrained_weights == 'hybrid':
                    # Initial convolutional block
                    if '0' in split_name[0]:
                        # Only convolutional layer
                        if split_name[1] == '0':
                            params.require_grade = False

                    # First convolutional block
                    if freeze_layers == 1:
                        if '1' in split_name[0]:
                            # Only convolutional layers
                            if split_name[3] == '0' or split_name[3] == '3' or split_name[3] == '6':
                                params.require_grade = False

                if pretrained_weights == 'BigGAN':
                    if freeze_layers == 1:
                        # First convolutional block
                        if '1' in split_name[0]:
                            # Exclude identity_blocks
                            if split_name[2] == 'conv_block':
                                # Only convolutional layers 
                                if split_name[3] == '0' or split_name[3] == '3' or split_name[3] == '6':
                                    params.require_grade = False

        # DenseNet's freeze layers
        if 'densenet' in hps['network']:
            for _, (name, params) in enumerate(net.named_parameters()):
                split_name = name.split('.')

                if pretrained_weights == 'DenseNet':
                    if freeze_layers == 1:
                        # Initial convolutional block (only convolutional layer)
                        if split_name[1] == 'conv0':
                            params.require_grade = False
                        
                        # First convolutional block
                        if split_name[1] == 'denseblock1':
                            # Only convolutional layers
                            if 'conv' in split_name[3]:
                                params.require_grade = False

                if pretrained_weights == 'BigGAN':
                    if freeze_layers == 1:
                        # First convolutional block
                        if split_name[1] == 'denseblock1':
                            if split_name[2] == 'denselayer1':
                                # Only convolutional layers
                                if 'conv' in split_name[3]:
                                    params.require_grade = False
        
        # BigGAN Dicriminator freeze layers
        if 'biggan' in hps['network']:
            for _, (name, params) in enumerate(net.named_parameters()):
                split_name = name.split('.')

                if split_name[0] == 'blocks':
                    if freeze_layers == 1:
                        if split_name[1] == '0':
                            params.require_grade = False
                    if freeze_layers == 2:
                        if split_name[1] == '0' or split_name[1] == '1':
                            params.require_grade = False
                    if freeze_layers == 3:
                        if split_name[1] == '0' or split_name[1] == '1' or split_name[1] == '2':
                            params.require_grade = False
                    if freeze_layers == 4:
                        if split_name[1] == '0' or split_name[1] == '1' or split_name[1] == '2' or split_name[1] == '3':
                            params.require_grade = False
                    if freeze_layers == 5:
                        if split_name[1] == '0' or split_name[1] == '1' or split_name[1] == '2' or split_name[1] == '3' or split_name[1] == '4':
                            params.require_grade = False
        

    print('Total params: {}'.format(sum(p.numel() for p in net.parameters())))
    print('Trainable params: {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    mu = torch.cuda.memory_allocated()*1e-9
    memory_usage.append(mu)
    if verbose >= 5:
        print('GPU Memory usage (before net allocation): {:.2f} GB'.format(mu))

    # moves the model architecture to the GPU
    net.to(device)
    
    mu = torch.cuda.memory_allocated()*1e-9
    memory_usage.append(mu)
    if verbose >= 5:
        print('GPU Memory usage (after net allocation): {:.2f} GB'.format(mu))

    # %% DATA SPLIT
    
    train_csv_file = '{}/split_{}/train.csv'.format(data_path, split)
    val_csv_file = '{}/split_{}/val.csv'.format(data_path, split)
    img_dir = '{}/datasets/hcid'.format(os.path.abspath(''))

    train_dataset = Plain_Dataset(csv_file = train_csv_file, img_dir = img_dir, transform = transform)
    val_dataset = Plain_Dataset(csv_file = val_csv_file, img_dir = img_dir, transform = norm)

    # %% INITIAL OVERSAMPLING

    # number of images per class
    _, counts_val = np.unique(val_dataset.labels, return_counts = True)
    
    if not oversampling:
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

    if oversampling:
        # oversampling
        _, counts = np.unique(train_dataset.labels, return_counts = True)
        weights = [sum(counts)/c for c in counts]
        
        samples_weights = [weights[e] for e in train_dataset.labels]
        sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(train_dataset.labels))
        
        train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = sampler, num_workers = 0)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)

    # %% LOSS AND OPTIMIZER FUNCTIONS

    scaler = GradScaler()

    # loss and optimizer function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = lr, momentum = .9, nesterov = True, weight_decay = 1e-4)

    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = .1)

    # %% TRAIN

    Train(net, epochs, train_loader, val_loader, split)

    # deallocate model
    mu = torch.cuda.memory_allocated()*1e-9
    memory_usage.append(mu)
    if verbose >= 5:
        print('GPU Memory usage (before net deallocation): {:.2f} GB'.format(mu))

    del net
    del criterion
    del optimizer

    mu = torch.cuda.memory_allocated()*1e-9
    memory_usage.append(mu)
    if verbose >= 5:
        print('GPU Memory usage (after net deallocation): {:.2f} GB'.format(mu))

# %% RESULTS

def save_histories(): 
    # dataframe of acc and loss series
    df_hist = pd.DataFrame()

    # dataframe of val acc per class
    df_val_acc_class_hist = pd.DataFrame()
    
    for split in range(1, n_splits+1):
        df_hist['train_acc_hist_split_{}'.format(split)] = train_acc_hist[split-1]
        df_hist['train_loss_hist_split_{}'.format(split)] = train_loss_hist[split-1]
        df_hist['val_acc_hist_split_{}'.format(split)] = val_acc_hist[split-1]
        df_hist['val_loss_hist_split_{}'.format(split)] = val_loss_hist[split-1]

        for label in range(len(classes_names)):
            df_val_acc_class_hist['val_acc_hist_{}_split_{}'.format(classes_names[label], split)] = val_acc_class_hist[label+((split-1)*len(classes_names))]
 
    # save dataframes
    if not os.path.exists('results'):
        os.mkdir('results')
        
        df_hist.to_csv('results/histories_{}_{}.csv'.format(test, data_split), index = False)
        df_val_acc_class_hist.to_csv('results/val_acc_class_history_{}_{}.csv'.format(test, data_split), index = False)
    else:
        df_hist.to_csv('results/histories_{}_{}.csv'.format(test, data_split), index = False)
        df_val_acc_class_hist.to_csv('results/val_acc_class_history_{}_{}.csv'.format(test, data_split), index = False)

    # memory usage
    df_memory = pd.DataFrame()
    df_memory['memory usage'] = memory_usage

    df_memory.to_csv('results/memory_usage_{}_{}.csv'.format(test, data_split), index = False)

save_histories()
