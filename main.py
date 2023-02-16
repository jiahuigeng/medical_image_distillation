import time, os, h5py
import numpy as np
from os import listdir
from pathlib import Path
import PIL.Image
import matplotlib.pyplot as plt
import shutil
import torch
import torchvision
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
from torchvision import transforms, datasets
import wandb

wandb.login()
wandb.init(project="pcam-pytorch-training")
wandb.run.name = "pcam-pytorch-experiment#-" + wandb.run.id
print("Staring experiment: ", wandb.run.name)


BATCH_SIZE = 16
data_path = "data/"
CHECKPOINT_DIR = 'checkpoint'
dataloader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 2}

class H5Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.dataset_x = None
        self.dataset_y = None
        self.transform = transform
        ### Going to read the X part of the dataset - it's a different file
        with h5py.File(self.file_path + '_x.h5', 'r') as filex:
            self.dataset_x_len = len(filex['x'])

        ### Going to read the y part of the dataset - it's a different file
        with h5py.File(self.file_path + '_y.h5', 'r') as filey:
            self.dataset_y_len = len(filey['y'])

    def __len__(self):
        assert self.dataset_x_len == self.dataset_y_len # Since we are reading from different sources, validating we are good in terms of size both X, Y
        return self.dataset_x_len

    def __getitem__(self, index):
        imgs_path = self.file_path + '_x.h5'
        labels_path = self.file_path + '_y.h5'

        if self.dataset_x is None:
            self.dataset_x = h5py.File(imgs_path, 'r')['x']
        if self.dataset_y is None:
            self.dataset_y = h5py.File(labels_path, 'r')['y']

        # get one pair of X, Y and return them, transform if needed
        image = self.dataset_x[index]
        label = self.dataset_y[index]

        if self.transform:
            image = self.transform(image)

        return (image, label)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CNN_V2(nn.Module):
    """Implemented by paper: http://cs230.stanford.edu/projects_winter_2019/posters/15813053.pdf"""

    def __init__(self, p=0.5):
        # log dropout parameter
        wandb.config.dropout = p
        """Init method for initializaing the CNN model"""
        super(CNN_V2, self).__init__()
        # 1. Convolutional layers
        # Single image is in shape: 3x96x96 (CxHxW, H==W), RGB images
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(p=p)

        # 2. FC layers to final output
        self.fc1 = nn.Linear(in_features=128 * 6 * 6, out_features=512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc_bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Convolution Layers, followed by Batch Normalizations, Maxpool, and ReLU
        x = self.bn1(self.conv1(x))  # batch_size x 96 x 96 x 16
        x = self.pool(F.relu(x))  # batch_size x 48 x 48 x 16
        x = self.bn2(self.conv2(x))  # batch_size x 48 x 48 x 32
        x = self.pool(F.relu(x))  # batch_size x 24 x 24 x 32
        x = self.bn3(self.conv3(x))  # batch_size x 24 x 24 x 64
        x = self.pool(F.relu(x))  # batch_size x 12 x 12 x 64
        x = self.bn4(self.conv4(x))  # batch_size x 12 x 12 x 128
        x = self.pool(F.relu(x))  # batch_size x  6 x  6 x 128
        # Flatten the output for each image
        x = x.reshape(-1, self.num_flat_features(x))  # batch_size x 6*6*128

        # Apply 4 FC Layers
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.fc_bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def sigmoid(x):
    """This method calculates the sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))


def training_accuracy(predicted, true, i, acc, tpr, tnr):
    """Taken from https://www.kaggle.com/krishanudb/cancer-detection-deep-learning-model-using-pytorch"""
    predicted = predicted.cpu()  # Taking the predictions, why cpu and not device?
    true = true.cpu()  # Taking the labels, why cpu and not device?

    predicted = (sigmoid(predicted.data.numpy()) > 0.5)  # Using sigmoid above, if prediction > 0.5 it is 1
    true = true.data.numpy()  # Numpy - can't combine above?
    accuracy = np.sum(predicted == true) / true.shape[0]  # Accuracy is: (TP + TN)/(TP + TN + FN + FP)
    true_positive_rate = np.sum((predicted == 1) * (true == 1)) / np.sum(true == 1)  # TPR: TP / (TP + FN) aka Recall
    true_negative_rate = np.sum((predicted == 0) * (true == 0)) / np.sum(true == 0)  # TNR: TN / (FP + TN)
    acc = acc * (i) / (i + 1) + accuracy / (i + 1)
    tpr = tpr * (i) / (i + 1) + true_positive_rate / (i + 1)
    tnr = tnr * (i) / (i + 1) + true_negative_rate / (i + 1)
    return acc, tpr, tnr


def dev_accuracy(predicted, target):
    """Taken from https://www.kaggle.com/krishanudb/cancer-detection-deep-learning-model-using-pytorch"""
    predicted = predicted.cpu()
    target = target.cpu()
    predicted = (sigmoid(predicted.data.numpy()) > 0.5)
    true = target.data.numpy()
    accuracy = np.sum(predicted == true) / true.shape[0]
    true_positive_rate = np.sum((predicted == 1) * (true == 1)) / np.sum(true == 1)
    true_negative_rate = np.sum((predicted == 0) * (true == 0)) / np.sum(true == 0)
    return accuracy, true_positive_rate, true_negative_rate


def fetch_state(epoch, model, optimizer, dev_loss_min, dev_acc_max):
    """Returns the state dictionary for a model and optimizer"""
    state = {
        'epoch': epoch,
        'dev_loss_min': dev_loss_min,
        'dev_acc_max': dev_acc_max,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict()
    }
    return state


def save_checkpoint(state, is_best=False, checkpoint=CHECKPOINT_DIR):
    """Taken from CS230 PyTorch Code Examples"""
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last_v2.pth.tar')
    if (not os.path.exists(checkpoint)):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if (is_best):
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_v2.pth.tar'))


def load_checkpoint(model, optimizer=None, checkpoint=CHECKPOINT_DIR):
    """Taken from CS230 PyTorch Code Examples"""
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        print("File doesn't exist {}".format(checkpoint))
        checkpoint = None
        return
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

if __name__ == "__main__":

    train_path = data_path + 'camelyonpatch_level_2_split_train'
    val_path = data_path + 'camelyonpatch_level_2_split_valid'
    test_path = data_path + 'camelyonpatch_level_2_split_test'

    test_dataset = H5Dataset(test_path, transform=test_transforms)
    test_loader = DataLoader(test_dataset, **dataloader_params)

    val_dataset = H5Dataset(val_path, transform=test_transforms)
    dev_loader = DataLoader(val_dataset, **dataloader_params)

    train_dataset = H5Dataset(train_path, transform=train_transforms)
    train_loader = DataLoader(train_dataset, **dataloader_params)

    USE_GPU = torch.cuda.is_available()

    model = CNN_V2()
    if (USE_GPU):
        model.cuda()

    # Hyperparameters
    lr = 5e-4
    wandb.config.learning_rate = lr

    # Parameters
    num_workers = 0
    total_epochs = 0
    num_epochs = 1
    early_stop_limit = 10
    bad_epoch_count = 0
    stop = False
    train_loss_min = np.Inf
    dev_loss_min = np.Inf
    dev_acc_max = 0

    # Optimizer + Loss Function
    # optimizer = optim.Adam(model.parameters(), lr = lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)  # SWATS
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy for binary classification - malignant / benign

    # Load best checkpoint
    best_checkpoint = os.path.join(CHECKPOINT_DIR, 'best_v2.pth.tar');
    # checkpoint = load_checkpoint(model = model, optimizer = optimizer, checkpoint = best_checkpoint)
    # checkpoint = load_checkpoint(model = model, optimizer = None, checkpoint = best_checkpoint) # SWATS
    # total_epochs = None if checkpoint is None else checkpoint['epoch']
    total_epochs = 0

    # Initialize arrays for plot
    train_loss_arr = []
    train_acc_arr = []
    train_tpr_arr = []
    train_tnr_arr = []

    dev_loss_arr = []
    dev_acc_arr = []
    dev_tpr_arr = []
    dev_tnr_arr = []

    # Loop over the dataset multiple times
    total_num_epochs = total_epochs + num_epochs
    for epoch in range(num_epochs):
        curr_epoch = total_epochs + epoch + 1
        # Keep track of training loss
        train_loss = []
        # Keep track of dev loss
        dev_loss = []
        # Keep track of accuracy measurements
        acc, tpr, tnr = 0.0, 0.0, 0.0

        # Train the model
        start_time = time.time()
        model.train()
        for batch_idx, (image, label) in enumerate(train_loader):
            if USE_GPU:
                data, target = image.cuda(), label.cuda()
            else:
                data, target = image, label
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Update target to be the same dimensions as output
            target = target.view(output.shape[0], 1).float()
            # Get accuracy measurements
            acc, tpr, tnr = training_accuracy(output, target, batch_idx, acc, tpr, tnr)
            # Calculate the batch's loss
            curr_train_loss = criterion(output, target)
            # Update the training loss
            train_loss.append(curr_train_loss.item())
            # Backward pass
            curr_train_loss.backward()
            # Perform a single optimization step to update parameters
            optimizer.step()
            # Print debug info every 64 batches
            if (batch_idx) % 64 == 0:
                print('Epoch {}/{}; Iter {}/{}; Loss: {:.4f}; Acc: {:.3f}; True Pos: {:.3f}; True Neg: {:.3f}'
                      .format(curr_epoch, total_num_epochs, batch_idx + 1, len(train_loader), curr_train_loss.item(),
                              acc, tpr, tnr))

        end_time = time.time()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(dev_loader):
                if USE_GPU:
                    data, target = image.cuda(), label.cuda()
                else:
                    data, target = image, label
                # Get predicted output
                output = model(data)
                # Update target to be the same dimensions as output
                target = target.view(output.shape[0], 1).float()
                # Get accuracy measurements
                dev_acc, dev_tpr, dev_tnr = dev_accuracy(output, target)
                # Calculate the batch's loss
                curr_dev_loss = criterion(output, target)
                # Update the dev loss
                dev_loss.append(curr_dev_loss.item())

        # Calculate average loss
        avg_train_loss = np.mean(np.array(train_loss))
        avg_dev_loss = np.mean(np.array(dev_loss))

        # Update dev loss arrays
        dev_loss_arr.append(avg_dev_loss)
        dev_acc_arr.append(dev_acc)
        dev_tpr_arr.append(dev_tpr)
        dev_tnr_arr.append(dev_tnr)

        # Update training loss arrays
        train_loss_arr.append(avg_train_loss)
        train_acc_arr.append(acc)
        train_tpr_arr.append(tpr)
        train_tnr_arr.append(tnr)

        print(
            'Epoch {}/{}; Avg. Train Loss: {:.4f}; Train Acc: {:.3f}; Train TPR: {:.3f}; Train TNR: {:.3f}; Epoch Time: {} mins; \nAvg. Dev Loss: {:.4f}; Dev Acc: {:.3f}; Dev TPR: {:.3f}; Dev TNR: {:.3f}\n'
            .format(curr_epoch, total_num_epochs, avg_train_loss, acc, tpr, tnr,
                    round((end_time - start_time) / 60., 2), avg_dev_loss, dev_acc, dev_tpr, dev_tnr))

        wandb.log({'epoch': curr_epoch, 'loss': avg_train_loss, 'accuracy': acc, 'tpr': tpr,
                   'time_per_epoch_min': round((end_time - start_time) / 60., 2)})

        if avg_dev_loss < dev_loss_min:
            print('Dev loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(dev_loss_min, avg_dev_loss))
            dev_loss_min = avg_dev_loss
            is_best = False
            if (dev_acc >= dev_acc_max):
                is_best = True
                dev_acc_max = dev_acc
            state = fetch_state(epoch=curr_epoch, model=model, optimizer=optimizer,
                                dev_loss_min=dev_loss_min,
                                dev_acc_max=dev_acc_max)
            save_checkpoint(state=state, is_best=is_best)
            bad_epoch_count = 0
        # If dev loss didn't improve, increase bad_epoch_count and stop if
        # bad_epoch_count >= early_stop_limit
        else:
            bad_epoch_count += 1
            print('{} epochs of increasing dev loss ({:.6f} --> {:.6f}).'
                  .format(bad_epoch_count, dev_loss_min, avg_dev_loss))
            if (bad_epoch_count >= early_stop_limit):
                print('Stopping training')
                stop = True

        if (stop):
            break

    print(model.state_dict)
    model.eval()
    torch.save(model, 'CNNv2_1.pt')

    # for x, y in test_dataloader:
    #   print(x.shape)
    #   print(y.shape)
    #   break
    #
    # for x, y in val_dataloader:
    #   print(x.shape)
    #   print(y.shape)
    #   break
    #
    # for x, y in train_dataloader:
    #   print(x.shape)
    #   print(y.shape)
    #   break