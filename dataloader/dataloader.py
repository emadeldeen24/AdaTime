import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

import os
import numpy as np
import random


class Load_Dataset(Dataset):
    def __init__(self, dataset, normalize):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()

        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train

        self.num_channels = X_train.shape[1]

        if normalize:
            # Assume datashape: num_samples, num_channels, seq_length
            data_mean = torch.FloatTensor(self.num_channels).fill_(0).tolist()  # assume min= number of channels
            data_std = torch.FloatTensor(self.num_channels).fill_(1).tolist()  # assume min= number of channels
            data_transform = transforms.Normalize(mean=data_mean, std=data_std)
            self.transform = data_transform
        else:
            self.transform = None

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        if self.transform is not None:
            output = self.transform(self.x_data[index].view(self.num_channels, -1, 1))
            self.x_data[index] = output.view(self.x_data[index].shape)

        return self.x_data[index].float(), self.y_data[index].long()

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset, dataset_configs.normalize)
    test_dataset = Load_Dataset(test_dataset, dataset_configs.normalize)

    # Dataloaders
    batch_size = hparams["batch_size"]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
    return train_loader, test_loader


def few_shot_data_generator(data_loader):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data
    if not isinstance(y_data, (np.ndarray)):
        y_data = y_data.numpy()

    NUM_SAMPLES_PER_CLASS = 5
    NUM_CLASSES = len(np.unique(y_data))

    samples_count_dict = {id: 0 for id in range(NUM_CLASSES)}

    # if the min number of samples in one class is less than NUM_SAMPLES_PER_CLASS
    y_list = y_data.tolist()
    counts = [y_list.count(i) for i in range(NUM_CLASSES)]

    for i in samples_count_dict:
        if counts[i] < NUM_SAMPLES_PER_CLASS:
            samples_count_dict[i] = counts[i]
        else:
            samples_count_dict[i] = NUM_SAMPLES_PER_CLASS

    # if min(counts) < NUM_SAMPLES_PER_CLASS:
    #     NUM_SAMPLES_PER_CLASS = min(counts)

    samples_ids = {}
    for i in range(NUM_CLASSES):
        samples_ids[i] = [np.where(y_data == i)[0]][0]

    selected_ids = {}
    for i in range(NUM_CLASSES):
        selected_ids[i] = random.sample(list(samples_ids[i]), samples_count_dict[i])

    # select the samples according to the selected random ids
    y = torch.from_numpy(y_data)
    selected_x = x_data[list(selected_ids[0])]
    selected_y = y[list(selected_ids[0])]

    for i in range(1, NUM_CLASSES):
        selected_x = torch.cat((selected_x, x_data[list(selected_ids[i])]), dim=0)
        selected_y = torch.cat((selected_y, y[list(selected_ids[i])]), dim=0)

    few_shot_dataset = {"samples": selected_x, "labels": selected_y}
    # Loading datasets
    few_shot_dataset = Load_Dataset(few_shot_dataset, None)

    # Dataloaders
    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=len(few_shot_dataset),
                                                  shuffle=False, drop_last=False, num_workers=0)
    return few_shot_loader


def generator_percentage_of_data(data_loader):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data

    X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=0)

    few_shot_dataset = {"samples": X_val, "labels": y_val}
    # Loading datasets
    few_shot_dataset = Load_Dataset(few_shot_dataset, None)

    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=32,
                                                  shuffle=True, drop_last=True, num_workers=0)
    return few_shot_loader
