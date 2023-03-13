#!/usr/bin/env python

import numpy as np
import torch 
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from pytorch_lightning.core import LightningModule
from pytorch_lightning.metrics.functional import  accuracy 
from pytorch_lightning import loggers as pl_loggers
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class HAR_torch_Dataset(Dataset):
    def __init__(self, data,labels, transform= None):
        """Reads source and target sequences from processing file ."""        
        self.input_tensor = data.float()
        self.label = labels
        self.transform =transform
        self.num_total_seqs = len(self.input_tensor)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        input_seq = self.input_tensor[index]
        input_labels = self.label[index]
        if self.transform:
            input_seq = self.transform(input_seq)
        return input_seq, input_labels

    def __len__(self):
        return self.num_total_seqs

def HAR_data_generator(data_dir,save=False):
    # dataloading 
    subject_data= np.loadtxt(f'{data_dir}/train/subject_train.txt')
    # Samples
    train_acc_x= np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_x_train.txt')
    train_acc_y= np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_y_train.txt')
    train_acc_z= np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_z_train.txt')
    train_gyro_x= np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_x_train.txt')
    train_gyro_y= np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_y_train.txt')
    train_gyro_z= np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_z_train.txt')
    train_tot_acc_x= np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_x_train.txt')
    train_tot_acc_y= np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_y_train.txt')
    train_tot_acc_z= np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_z_train.txt')

    test_acc_x= np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_x_test.txt')
    test_acc_y= np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_y_test.txt')
    test_acc_z= np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_z_test.txt')
    test_gyro_x= np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_x_test.txt')
    test_gyro_y= np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_y_test.txt')
    test_gyro_z= np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_z_test.txt')
    test_tot_acc_x= np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_x_test.txt')
    test_tot_acc_y= np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_y_test.txt')
    test_tot_acc_z= np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_z_test.txt')

    # Stacking channels together data 
    train_data= np.stack((train_acc_x,train_acc_y,train_acc_z,
                              train_gyro_x,train_gyro_y,train_gyro_z,
                              train_tot_acc_x, train_tot_acc_y,train_tot_acc_z),axis=1)
    test_data= np.stack((test_acc_x,test_acc_y,test_acc_z,
                              test_gyro_x,test_gyro_y,test_gyro_z,
                              test_tot_acc_x, test_tot_acc_y,test_tot_acc_z),axis=1)
    # labels 
    train_labels=  np.loadtxt(f'{data_dir}/train/y_train.txt')
    train_labels -= np.min(train_labels)
    test_labels=  np.loadtxt(f'{data_dir}/test/y_test.txt')
    test_labels -= np.min(test_labels)


    HAR_dataset_processed= {'train':{'samples':torch.from_numpy(train_data), 'labels':torch.LongTensor(train_labels)},
                               'test':{'samples':torch.from_numpy(test_data), 'labels':torch.LongTensor(test_labels)}}
    if save:
        for mode in ['train', 'test']:
            torch.save(HAR_dataset_processed[mode], f'{mode}.pt')
    return HAR_dataset_processed




def HAR_data_loader(dataset, batch_size, shuffle, drop_last) :
    train_dataset = HAR_torch_Dataset(dataset['train']['samples'], dataset['train']['labels'])
    test_dataset = HAR_torch_Dataset(dataset['test']['samples'], dataset['test']['labels'])
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_dl = DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

    return train_dl, test_dl

