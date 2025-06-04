# %% LIBRARIES AND HYPER-PARAMETERS

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sb
import shutil
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import warnings

from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from data_loaders import Plain_Dataset
from hyper_parameters import setup_hparams
from networks.setup_network import setup_network

warnings.filterwarnings('ignore')

hps = setup_hparams(sys.argv[1:]) # hyper-parameters
epochs, batch_size, lr, test = hps['epochs'], hps['bs'], hps['lr'], hps['test'] # model variables
net_name = hps['network'] # net name
img_size = hps['input_size'] # tencrop image size
data_split, n_splits =  hps['data_split'], hps['n_splits'] # cross validation
classes_names = ['girolando', 'holandes'] # classes names
cross_val = hps['cross_val'] # cross validation
data_path = 'data/cross_validation_{}/data_split_{}'.format(cross_val, data_split) # path of the data and pre-trained model
img_dir = '{}/datasets/hcid'.format(os.path.abspath('')) # path of the dataset
plots_folder = 'plots/test_{}'.format(test) # path to the plots folder
webcam_aplication = False # webcam aplication

# wrongly predicted images folder
wrong_pred_img_folder = os.path.join(os.getcwd(),
                                     'wrong_pred_imgs',
                                     'test_{}'.format(test),
                                     'data_split_{}'.format(data_split))

if not os.path.exists(wrong_pred_img_folder):
    os.makedirs(wrong_pred_img_folder)

img_x = 1

#set the processing device to GPU, if it exists
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transform and load images from test.csv
mean, std = 0.5, 0.5

norm = T.Compose([
    T.Resize((img_size, img_size)),
    
    T.ToTensor(),
    T.Normalize(mean = (mean, mean, mean), std = (std, std, std))
])

#%% CROSS VALIDATION

#model evaluation
total = []

#variables used to calculate test accuracy 
num_correct = 0
num_samples = 0

#inputs of the confusion matrix
targets = []
predictions = []

for split in range(1, n_splits+1):
    
#%% DATA SPLIT

    dataset = Plain_Dataset(csv_file = '{}/split_{}/test.csv'.format(data_path, split),
                            img_dir = img_dir,
                            transform = norm)

    test_loader =  DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

#%% MODEL

    # paths of the pre-trained models
    model_path = 'models/{}-{}-{}-{}-{}-{}-{}.pt'.format(net_name, epochs, batch_size, lr, test, data_split, split)

    # loads the network architecture to be used for prediction
    net, pretrained_weights = setup_network(hps)

    print('Model architecture:')
    print(net)
    print('Total params: {}'.format(sum(p.numel() for p in net.parameters())))
    print('Trainable params: {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()

# %% PREDICTION
    # makes the prediction and checks the model accuracy for the finaltest.csv
    with torch.no_grad(): # disables gradient calculation
        for data, labels, name in test_loader:
            # moves the data and labels to the GPU
            data, labels = data.to(device), labels.to(device)   

            # forward
            outputs = net(data)

            # prediction
            preds = torch.argmax(F.softmax(outputs, dim = 1), 1)

            # updates the number of correct predictions
            num_correct += (preds == labels).sum()

            # updates the total number of samples
            num_samples += preds.size(0)

            # adds the prediction and label information in two lists that will be used to plot the confusion matrix
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()

            for x in range(len(labels_cpu)):
                targets.append(labels_cpu[x])
                predictions.append(preds_cpu[x])

                # plot wrong predicted images
                if labels_cpu[x] != preds_cpu[x]:
                    img = Image.open('{}/{}'.format(img_dir, name[x]))
                    img = img.resize((img_size, img_size))
                    img.save('{}/img({})_target({})_pred({}).jpg'.format(wrong_pred_img_folder, img_x, labels_cpu[x], preds_cpu[x]))

                    img_x+=1

            del data
            del labels
            del outputs
            del preds

    del net

model_acc = (num_correct)/(num_samples)*100
print('Accuracy of the network on the test images: {:.2f}%'.format(model_acc))

# function for ploting the confusion matrix results
def ploting_confusion_matrix():
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times']
    })

    # confusion matrix with absolute values
    cf_matrix = confusion_matrix(targets, predictions)
    group_counts = ['{0:.0f}'.format(value) for value in cf_matrix.flatten()]
    
    # confusion matrix with relative values
    cf_matrix = 100*cf_matrix/cf_matrix.sum(axis = 1)[:, np.newaxis]    
    group_percentages = ['{0:.2f}'.format(value) for value in cf_matrix.flatten()]
    
    # confusion matrix with both absolute and relative values
    annot = [f'{v1}' + r'\%' + f'\n{v2}' for v1, v2 in zip(group_percentages, group_counts)]
    annot = np.asarray(annot).reshape(len(classes_names) , len(classes_names))
    
    # ploting
    plt.figure(figsize = (20, 20))
    
    sb.set(rc = {'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Times']}, font_scale = 5)
    ax = sb.heatmap(cf_matrix,
                    xticklabels = classes_names,
                    yticklabels = classes_names,
                    annot = annot,
                    fmt = '',
                    cmap = 'Blues',
                    cbar = False)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(length = 0)
    plt.xlabel('Acc.: {:.2f}'.format(model_acc) + r'\%', labelpad = 30, fontsize = 60)
    plt.xticks(plt.xticks()[0], [label._text for label in plt.xticks()[1]], fontsize = 60)
    plt.yticks(plt.yticks()[0], [label._text for label in plt.yticks()[1]], fontsize = 60, rotation = 0)
    
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        
        plt.savefig('{}/mc_{}_{}.pdf'.format(plots_folder, test, data_split), bbox_inches = 'tight', pad_inches = 0)
    else:
        plt.savefig('{}/mc_{}_{}.pdf'.format(plots_folder, test, data_split), bbox_inches = 'tight', pad_inches = 0)
    
ploting_confusion_matrix()