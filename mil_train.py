import os.path
import sys
import numpy as np
import argparse

import torch
import torch.utils.data as data_utils
import torch.optim as optim
from utils import get_loops,get_mil_dataset, get_mil_network
from torch.autograd import Variable
import logging

parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')

parser.add_argument("--distill_mode", type=str, default="bag", choices=["bag", "instance"])
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--batch_size', type=int, default=1)

# common with mil distillation
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=4, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=0, metavar='VL',
                    help='variance of bag length')



# may change
# parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
#                     help='number of bags in training set')
# parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
#                     help='number of bags in test set')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
# parser.add_argument('--save_path', type=str, required=True, help='path to save results')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

def train(model, train_loader):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        loss = loss.mean()
        train_loss += loss.data#[0]
        error, _ = model.calculate_classification_error(data, bag_label)
        # error = error.mean()
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    logger.info('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy(), train_error))
    # print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))

def test(model, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

        # if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
        #     bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
        #     instance_level = list(zip(instance_labels.numpy()[0].tolist(),
        #                          np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
        #
        #     print('\nTrue Bag Label, Predicted Bag Label: {}\n'
        #           'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    logger.info('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.mkdir("models")

    save_path = os.path.join("models", f"model_bagsize{args.mean_bag_length}_bs{args.batch_size}")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(save_path, 'logs.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = get_mil_dataset(args)
    net = get_mil_network(args.model, channel, num_classes, im_size).to(args.device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
    train_loader = data_utils.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(dst_test, batch_size=1, shuffle=False)
    test(net, test_loader)
    for epoch in range(1, args.epochs + 1):
        train(net, train_loader)
        test(net, test_loader)


    torch.save(net.state_dict(), os.path.join(save_path, "model.pth"))

    # new_net = get_mil_network(args.model, channel, num_classes, im_size).to(args.device)
    # new_net.load_state_dict(torch.load("./models/model.pth"))
    #
    # test(new_net, test_loader)



# train_loader = data_utils.DataLoader(MILDataset(dataset="MNIST",
#                                                 target_number=9,
#                                                 mean_bag_length=10,
#                                                 var_bag_length=2,
#                                                 seed=1,
#                                                 train=True),
#                                      batch_size=4,
#                                      shuffle=True,
#
#                                      )

# trainset = MILDataset(target_number=9,
#                       mean_bag_length=10,
#                       var_bag_length=0,
#                       num_bag=100,
#                       seed=1,
#                       train=True)
#
# train_loader = data_utils.DataLoader(trainset,
#                                      batch_size=3,
#                                      shuffle=True)