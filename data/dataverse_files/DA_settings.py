
import numpy as np
import torch 
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class HAR_torch_Dataset(Dataset):
    def __init__(self, data,labels, transform= None):
        """Reads source and target sequences from processing file ."""        
        self.input_tensor = (torch.from_numpy(data)).float()
        self.label = torch.LongTensor(labels)
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

    # different subjects 
    subject_train= np.loadtxt(f'{data_dir}/train/subject_train.txt')
    subject_test= np.loadtxt(f'{data_dir}/test/subject_test.txt')
    # select subset i for train and subset j for testing

    all_subjects_data = np.concatenate((train_data, test_data))
    all_subjects_labels = np.concatenate((train_labels, test_labels))
    subject_indices =  np.concatenate((subject_train, subject_test))
    # arrange the subjects to different domains 
    
    domains_data, domains_labels= [],[]
    domain_names = ['a', 'b', 'c', 'd', 'e']
    for i in range(0, 25, 6):
        j= i+6 
        domains_data.append(all_subjects_data[np.where((i< subject_indices)&( subject_indices<=j))])
        domains_labels.append(all_subjects_labels[np.where((i< subject_indices)&( subject_indices<=j))])

    # split the domains to train_val_test
    HAR_dataset_processed = {}
    for domain_data, domain_labels, name in zip(domains_data, domains_labels, domain_names):
        # train, validation, test split of the data 
        X_train, X_test, y_train, y_test = train_test_split(domain_data, domain_labels, test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        HAR_dataset_processed[name]= {'train':{'samples':X_train, 'labels':y_train},
                                   'val':{'samples':X_val, 'labels':y_val},
                                   'test':{'samples':X_test, 'labels':y_test}}
    if save:
        torch.save(HAR_dataset_processed, 'HAR_DG_settings.numpy')
    return HAR_dataset_processed

def HAR_data_loader(full_data,domain_id, batch_size=32, shuffle=True, drop_last=True) :
    # datasets 
    dataset = full_data[domain_id]
    train_dataset = HAR_torch_Dataset(dataset['train']['samples'], dataset['train']['labels'])
    val_dataset = HAR_torch_Dataset(dataset['val']['samples'], dataset['val']['labels'])
    test_dataset = HAR_torch_Dataset(dataset['test']['samples'], dataset['test']['labels'])
    # dataloaders
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    valid_dl = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

    return train_dl, valid_dl, test_dl

