import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

root_dir = './dataset'

class LungDataset(Dataset):
    """Base class for the lung dataset.

    The lung dataset is split up into train, test, and val datasets.
    
    The images in the lung dataset consist of 150x150 greyscale X-ray pictures of normal lungs or infected lungs with or without Covid.
    """
    dataset_name = 'Lung Dataset'
    
    # All images are of size 150 x 150
    img_size = (150, 150)
    
    # Number of images in each part of the dataset
    dataset_numbers = {
        'train_normal': 1341,
        'train_infected_noncovid': 2530,
        'train_infected_covid': 1345,
        'test_normal': 234,
        'test_infected_noncovid': 242,
        'test_infected_covid': 138,
        'val_normal': 8,
        'val_infected_noncovid': 8,
        'val_infected_covid': 8,
    }
    
    # Types of datasets
    dataset_types = ['train', 'test', 'val']
    
    def __init__(self, type_, root_dir=None, transform=None):
        """Instantiate the lung dataset.
        
        Parameters:
        - type_ should be 'train', 'test', or 'val'.
        - root_dir should be the path to the dataset root directory.
        - transform should be a callable to be applied on an image.
        """
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - type_ should be one of the following: " + str(LungDataset.dataset_types) + "."
        assert type_ in LungDataset.dataset_types, err_msg
        
        # Type of dataset
        self.type = type_
        
        # Root directory of dataset
        if root_dir is None:
            root_dir = globals()['root_dir']
        self.root_dir = root_dir

        # Three classes will be considered (normal, infected with no covid, infected with covid)
        self.classes = ['normal', 'infected_noncovid', 'infected_covid']
        
        # Class weights
        self.class_weights = torch.Tensor([LungDataset.dataset_numbers['train_normal'],
                                           LungDataset.dataset_numbers['train_infected_noncovid'],
                                           LungDataset.dataset_numbers['train_infected_covid']])
        
        # Number of images in each part of the dataset
        self.dataset_numbers = {'normal': LungDataset.dataset_numbers[type_ + '_normal'],
                                'infected_noncovid': LungDataset.dataset_numbers[type_ + '_infected_noncovid'],
                                'infected_covid': LungDataset.dataset_numbers[type_ + '_infected_covid']}
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {'normal': os.path.join(root_dir, type_, 'normal'),
                              'infected_noncovid': os.path.join(root_dir, type_, 'infected', 'non-covid'),
                              'infected_covid': os.path.join(root_dir, type_, 'infected', 'covid')}
        
        # Function for transforming samples
        self.transform = transform
    
    def describe(self):
        """Generate and print a description of the dataset."""
        msg = "This is the " + self.type + " dataset of the " + self.dataset_name + " used "
        msg += "in the Small Project for the 50.039 Deep Learning class in Feb-March 2021. \n"
        msg += "It contains a total of {} images ".format(len(self))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are split into the following classes and they are stored in the following locations:\n"
        for cls in self.classes:
            msg += " - {} ({} images), in {}.\n".format(cls, self.dataset_numbers[cls], repr(self.dataset_paths[cls]))
        print(msg)

    def _get_imgpath(self, class_val, index_val):
        """Return the path of the specified image from the dataset."""
        return '{}/{}.jpg'.format(self.dataset_paths[class_val], index_val)
        
    def open_img(self, class_val, index_val):
        """Open and return the specified image from the dataset.
        
        Parameters:
        - class_val should be 'normal', 'infected_noncovid', or 'infected_covid'.
        - index_val should be an integer between 0 and the total number of images under the specified class in the dataset minus 1.
        """
        err_msg = "Error - class_val should be one of the following: " + str(self.classes) + "."
        assert class_val in self.classes, err_msg
        
        max_val = self.dataset_numbers[class_val]
        err_msg = "Error - index_val should be an integer between 0 and the total number of images " + \
                  "under the specified class in the dataset minus 1 ({}).".format(max_val - 1)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val < max_val, err_msg
        
        img_path = self._get_imgpath(class_val, index_val)
        
        return Image.open(img_path)

    def show_img(self, class_val, index_val):
        """Open and display the specified image from the dataset.
        
        Parameters:
        - class_val should be 'normal', 'infected_noncovid', or 'infected_covid'.
        - index_val should be an integer between 0 and the total number of images under the specified class in the dataset minus 1.
        """
        # Open image
        im = self.open_img(class_val, index_val)
        
        # Display
        plt.imshow(im)

    def __len__(self):
        """Return the total number of images in dataset."""
        return sum(self.dataset_numbers[cls] for cls in self.classes)

    def _classify_index(self, index):
        """Classify and return the updated index, class value, and its corresponding label."""
        for label, class_val in enumerate(self.classes):
            max_idx = self.dataset_numbers[class_val]
            if index < max_idx:
                break
            else:
                index -= max_idx
        
        return index, class_val, label
    
    def __getitem__(self, index):
        """Return the image and its label from the dataset.
        
        Parameters:
        - index should be an integer value between 0 and the total number of images in the dataset minus 1.
        """
        index, class_val, label = self._classify_index(index)
        im = self.open_img(class_val, index)
        if self.transform:
            im = self.transform(im)
        return im, label

class LungInfectedDataset(LungDataset):
    """Base class for the lung (normal/infected) dataset.

    The lung (normal/infected) dataset is split up into train, test, and val datasets.
    
    The images in the lung (normal/infected) dataset consist of 150x150 greyscale X-ray pictures of normal lungs or infected lungs (with and without Covid).
    """
    dataset_name = 'Lung (Normal/Infected) Dataset'
    
    def __init__(self, type_, root_dir=None, transform=None):
        """Instantiate the lung (normal/infected) dataset.
        
        Parameters:
        - type_ should be 'train', 'test', or 'val'.
        - root_dir should be the path to the dataset root directory.
        - transform should be a callable to be applied on an image.
        """
        super().__init__(type_, root_dir, transform)
        
        # Only two classes will be considered (normal, infected)
        self.classes = ['normal', 'infected']
        self.class_weights = torch.Tensor([LungDataset.dataset_numbers['train_normal'],
                                           LungDataset.dataset_numbers['train_infected_noncovid'] + \
                                           LungDataset.dataset_numbers['train_infected_covid']])
        
        # Set dataset number and path for 'infected' class
        self.dataset_numbers['infected'] = self.dataset_numbers['infected_noncovid'] + self.dataset_numbers['infected_covid']
        self.dataset_paths['infected'] = [self.dataset_paths['infected_noncovid'], self.dataset_paths['infected_covid']]
    
    def _get_imgpath(self, class_val, index_val):
        """Return the path of the specified image from the dataset."""
        if class_val == 'normal':
            return '{}/{}.jpg'.format(self.dataset_paths['normal'], index_val)
        
        # else class_val == 'infected'
        elif index_val < self.dataset_numbers['infected_noncovid']:
            return '{}/{}.jpg'.format(self.dataset_paths['infected_noncovid'], index_val)
        
        else: # index into infected covid
            return '{}/{}.jpg'.format(self.dataset_paths['infected_covid'], index_val - self.dataset_numbers['infected_noncovid'])

class LungCovidDataset(LungDataset):
    """Base class for the lung (non-covid/covid) dataset.

    The lung (non-covid/covid) dataset is split up into train, test, and val datasets.
    
    The images in the lung (non-covid/covid) dataset consist of 150x150 greyscale X-ray pictures of infected lungs with or without Covid.
    """
    dataset_name = 'Lung (Non-Covid/Covid) Dataset'
    
    def __init__(self, type_, root_dir=None, transform=None):
        """Instantiate the lung (non-covid/covid) dataset.
        
        Parameters:
        - type_ should be 'train', 'test', or 'val'.
        - root_dir should be the path to the dataset root directory.
        - transform should be a callable to be applied on an image.
        """
        super().__init__(type_, root_dir, transform)
        
        # Only two classes will be considered (infected non-covid, infected covid)
        self.classes.remove('normal')
        self.class_weights = self.class_weights[1:]
        
        # Remove dataset number and path for 'normal' class
        del self.dataset_numbers['normal']
        del self.dataset_paths['normal']