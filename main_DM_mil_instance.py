import os, sys
import time, datetime
import copy
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_mil_dataset, get_network, get_mil_network, \
    get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import wandb
import torchvision

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    # parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=10000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    # TODO: change batch_real to 1
    parser.add_argument('--batch_real', type=int, default=2, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=8, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', choices=['color_crop_cutout_flip_scale_rotate', 'none'], help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, required=True, help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')


    ##################### mil settings #####################
    parser.add_argument("--distill_mode", type=str, default="instance", choices=["bag", "instance"])
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
    parser.add_argument('--clip_alpha', type=int, default=10, help="coefficient for clip loss")
    parser.add_argument('--scale_alpha', type=int, default=0, help="coefficient for clip loss")

    parser.add_argument('--use_pretrain', type=str, default="", help='load pretrained model for distillation')
    parser.add_argument('--use_wandb', action='store_true', help='')

    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if args.use_wandb:
        run = wandb.init(project=f"MIL", job_type="Instance", config=args)

        cur_file = os.path.join(os.getcwd(), __file__)
        wandb.save(cur_file)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists("result"):
        os.mkdir("result")

    args.save_path = f'batch_size-{args.batch_real}-mean_bag_length-{args.mean_bag_length}'
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

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    logger.info(f'eval_it_pool: {eval_it_pool}')


    # channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_mil_dataset(args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    for exp in range(args.num_exp):
        # print('\n================== Exp %d ==================\n '%exp)
        # print('Hyper-parameters: \n', args.__dict__)
        # print('Evaluation model pool: ', model_eval_pool)

        logger.info('\n================== Exp %d ==================\n '%exp)
        logger.info(f'Hyper-parameters: {args.__dict__}\n')
        logger.info(f'Evaluation model pool: {model_eval_pool}')

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1][0] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        # images_all = torch.cat(images_all, dim=0).to(args.device)
        # labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)


        for c in range(num_classes):
            logger.info('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            # For ConvNet
            # if args.model == 'Attention':
            xs = [images_all[i] for i in idx_shuffle]
            # else:
            #     xs = [images_all[i][0] for i in idx_shuffle]
            x = torch.cat(xs, dim=0).to(args.device)
            return x

        # for ch in range(channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        # image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        image_syn = torch.rand(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float,
                                requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        if args.init == 'real':
            logger.info('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            logger.info('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        logger.info('%s training begins'%get_time())

        for it in range(args.Iteration+1):
            # if args.model == 'Attention':
            net = get_mil_network(args.model, channel, num_classes, im_size, args).to(args.device)
            # else:
            #     net = get_network(args.model, channel, num_classes, im_size, args).to(args.device)
            if args.use_pretrain:
                net.load_state_dict(torch.load(os.path.join(args.use_pretrain, "model.pth")))

            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            loss_avg = 0

            ''' update synthetic data '''
            if 'BN' not in args.model: # for ConvNet
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000

                        # TODO: dsa on original dataset
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    # clip_loss = torch.square(img_syn - torch.clip(img_syn, 0, 1)).detach()
                    # syn_max, syn_min = torch.max(img_syn), torch.min(img_syn)
                    # scale = syn_max - syn_min
                    # scale_loss = torch.square((img_syn - syn_min)/scale - img_syn).detach()
                    # loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                    # loss += args.clip_alpha *torch.sum(clip_loss) + args.scale_alpha * torch.sum(scale_loss)
                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
            else: # for ConvNetBN
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)


                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                clip_loss = torch.square(img_syn - torch.clip(img_syn, 0, 1)).detach()
                syn_max, syn_min = torch.max(img_syn), torch.min(img_syn)
                scale = syn_max - syn_min
                scale_loss = torch.square((img_syn - syn_min) / scale - img_syn).detach()
                loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)
                loss += args.clip_alpha *torch.sum(clip_loss) + args.scale_alpha * torch.sum(scale_loss)



            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()


            loss_avg /= (num_classes)

            if it%10 == 0:
                logger.info('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))
                if args.use_wandb:
                    wandb.log({f'Loss': loss_avg}, step=it)

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    logger.info('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                    args.model, model_eval, it))

                    logger.info(f'DSA augmentation strategy: {args.dsa_strategy}\n')
                    logger.info(f'DSA augmentation parameters: {args.dsa_param.__dict__}\n')

                    accs = []
                    for it_eval in range(args.num_eval):
                        # TODO: use attention network

                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                            label_syn.detach())  # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader,
                                                                 args)

                        accs.append(acc_test)
                    logger.info('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                    len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if args.use_wandb:
                        wandb.log({f'Accuracy/{model_eval}': np.mean(accs)}, step=it)
                        wandb.log({f'Std/{model_eval}': np.std(accs)}, step=it)

                    if it == args.Iteration:  # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(save_path, 'vis_mil_instance_%s_%s_%s_%dipc_exp%d_iter%d.png' % (
                args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(image_syn_vis, save_name,
                           nrow=args.ipc)  # Trying normalize = True/False may get better visual effects.
                grid = torchvision.utils.make_grid(image_syn_vis, nrow=args.ipc, normalize=True, scale_each=True)
                wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

    logger.info('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logger.info('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()
