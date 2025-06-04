import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

class Plain_Dataset(Dataset):
    '''
    Pytorch dataset class used for load and transform the .csv information read from training to .jpg images
    '''
    def __init__(self, csv_file, img_dir, transform):
        '''
        Class constructor
        
        Parameters
        ----------
        csv_file: the path of the csv file (train, validation, test)
        
        img_dir: the directory of the images (train, validation, test)
        
        datatype: string for searching along the image_dir (train, val, test)
        
        transform: pytorch transformation over the data
        '''

        self.csv_file = pd.read_csv(csv_file)
        self.labels = self.csv_file['label']
        self.names = self.csv_file['name']
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        '''
        This method is used to return the lenght of the csv_file
        '''

        return len(self.csv_file)

    def __getitem__(self, id_x):
        '''
        This method is used for list indexing and accessing ranges of values
        '''

        name = self.names[id_x]
        img = Image.open('{}/{}'.format(self.img_dir, name))
        if self.transform :
            img = self.transform(img)

        label = torch.tensor(self.labels[id_x]).type(torch.long)

        sample = (img, label, name)

        return sample