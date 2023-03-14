import time
import time, os, h5py
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class MILDataset(Dataset):
    def __init__(self, dataset="MNIST", target_number=9, max_instances=15, mean_bag_length=10, var_bag_length=2,
                 seed=1, train=True):
        self.train = train
        self.dataset = dataset
        self.r = np.random.RandomState(seed)
        self.max_instances = max_instances
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.data, self.labels = self.__create_bags__()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        bag = self.data[index]
        label = self.labels[index]

        num_instances = bag.shape[0]

        padded_bag = torch.zeros(((self.max_instances,)+ bag.shape[1:]))
        padded_bag[:num_instances] = bag

        padded_label = torch.zeros(((self.max_instances)), dtype=bool)
        padded_label[:num_instances] = label

        return padded_bag, [max(padded_label), padded_label]

    def __create_bags__(self):
        self.data, self.labels = [], []
        if self.dataset == "MNIST":
            # channel = 1
            # im_size = (28, 28)
            # num_classes = 10
            mean = [0.1307]
            std = [0.3081]
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            if self.train == True:
                dataset = datasets.MNIST('data', train=True, download=True, transform=transform)  # no augmentation
            else:
                dataset = datasets.MNIST('data', train=False, download=True, transform=transform)


        elif self.dataset == "CIFAR10":
            # channel = 3
            # im_size = (32, 32)
            # num_classes = 10
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

            if self.train == True:
                dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)  # no augmentation
            else:
                dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)

        origin_data, origin_labels = dataset.data, dataset.targets
        origin_data = origin_data.unsqueeze(1)
        len_dst = len(dataset)
        index = list(range(len_dst))
        if self.train:
            shuffle(index)

        idx = 0
        bags = []

        labels = []
        while idx < len_dst:
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            bag_length = 1 if bag_length < 1 else self.max_instances if bag_length > self.max_instances else bag_length
            bag_index = torch.LongTensor(index[idx: min(len_dst, idx+bag_length)])
            images_in_bag = origin_data[bag_index]
            labels_in_bag = origin_labels[bag_index] == self.target_number
            idx += bag_length
            bags.append(images_in_bag)
            labels.append(labels_in_bag)

        return bags, labels

def get_mil_dataset(args):
    dst_train = MILDataset(dataset=args.dataset, target_number=args.target_number,  mean_bag_length=args.mean_bag_length,
                           var_bag_length=args.var_bag_length, seed=args.seed,
                           train=True)

    if args.distill_mode == "bag":
        dst_test = MILDataset(dataset=args.dataset, target_number=args.target_number, mean_bag_length=args.mean_bag_length,
                              var_bag_length=args.var_bag_length, seed=args.seed,
                              train=False)

    if args.dataset == "MNIST":
        channel = 1
        im_size = (28, 28)
        num_classes = 2
        mean = [0.1307]
        std = [0.3081]
        class_names = ['True', 'False']

        if args.distill_mode == "instance":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            dst_test = datasets.MNIST("data", train=False, download=True, transform=transform)
            dst_test.targets = (dst_test.targets == args.target_number)


    elif args.dataset == "CIFAR10":
        channel = 3
        im_size = (32, 32)
        num_classes = 2
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        class_names = ['True', 'False']

        if args.distill_mode == "instance":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            dst_test = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
            dst_test.targets = (dst_test.targets == args.target_number)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=1, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader


if __name__ == "__main__":
    train_loader = data_utils.DataLoader(MILDataset(dataset="MNIST",
                                                   target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   seed=1,
                                                   train=True),
                                         batch_size=4,
                                         shuffle=True,
                                         )

    test_loader = data_utils.DataLoader(MILDataset(dataset="MNIST",
                                                    target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  seed=1,
                                                  train=False),
                                        batch_size=4,
                                        shuffle=False,
                                        )

    for batch_idx, (bag, label) in enumerate(train_loader):
        print(bag, label)





