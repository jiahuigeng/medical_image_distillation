import os
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_mil_dataset, get_network, get_mil_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import logging
import datetime
import torchvision


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--distill_mode', type=str, default="bag", choices=["bag", "instance"])
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=3, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=5000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=8, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, required=True, help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')


    ##################### mil settings #####################
    parser.add_argument('--target_number', type=int, default=9, metavar='T',
                        help='bags have a positive labels if they contain at least one 9')
    parser.add_argument('--mean_bag_length', type=int, default=2, metavar='ML',
                        help='average bag length')
    parser.add_argument('--var_bag_length', type=int, default=0, metavar='VL',
                        help='variance of bag length')
    parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                        help='number of bags in training set')
    parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                        help='number of bags in test set')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(os.path.join("result", args.save_path)):
        os.mkdir(os.path.join("result", args.save_path))

    save_path = os.path.join("result", args.save_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
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

    for ipc_size in [10, 20, 50, 100]:
        args.ipc = ipc_size
        for bag_size in [2,4,8,16]:
            args.mean_bag_length = bag_size

            if args.distill_mode == "instance":
                channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, args.target_number)
                model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

                images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
                labels_all = [dst_train[i][1] for i in range(len(dst_train))]

            elif args.distill_mode == "bag":
                channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_mil_dataset(args)
                model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

                images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
                labels_all = [dst_train[i][1][0] for i in range(len(dst_train))]


            indices_class = [[] for c in range(num_classes)]
            for i, lab in enumerate(labels_all):
                indices_class[lab].append(i)

            if not os.path.exists(os.path.join("result", args.save_path)):
                os.mkdir(os.path.join("result", args.save_path))


            total_acc = []

            for exp in range(args.num_exp):
                logger.info('\n================== Exp %d ==================\n '%exp)
                logger.info('Hyper-parameters: %s \n' % args.__dict__)
                # print('Evaluation model pool: ', model_eval_pool)

                ''' organize the real dataset '''

                def get_images(c, n): # get random n images from class c
                    idx_shuffle = np.random.permutation(indices_class[c])[:n]
                    return torch.cat([images_all[i] for i in idx_shuffle], dim=0).to(args.device)


                #
                if args.distill_mode == "instance":
                    image_rnd = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float,
                                            requires_grad=True, device=args.device)
                else:
                    image_rnd = torch.randn(
                        size=(num_classes * args.ipc, args.mean_bag_length, channel, im_size[0], im_size[1]), dtype=torch.float,
                        device=args.device)

                for c in range(num_classes):
                    image_rnd.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
                    label_rnd = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long,
                                             requires_grad=False, device=args.device).view(-1)

                for model_eval in model_eval_pool:
                    logger.info('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s' % (
                    args.model, model_eval))

                    # print('DSA augmentation strategy: \n', args.dsa_strategy)
                    # print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    for it_eval in range(args.num_eval):
                        if args.distill_mode == "instance":
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                        else:
                            net_eval = get_mil_network(args.model, channel, num_classes, im_size).to(args.device)
                        # image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                        #     label_syn.detach())  # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_rnd, label_rnd, testloader,
                                                                 args)
                        accs.append(acc_test)
                    logger.info('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                    len(accs), model_eval, np.mean(accs), np.std(accs)))

                    total_acc += accs

            logger.info('Finally Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                len(total_acc), model_eval, np.mean(total_acc), np.std(total_acc)))



if __name__ == '__main__':
    main()
