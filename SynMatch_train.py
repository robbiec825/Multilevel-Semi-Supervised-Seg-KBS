# This file is modified from https://github.com/HiLab-git/SSL4MIS
# Copyright (c) 2020 Xiangde Luo
# Licensed under the MIT License

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet_fp as ViT_seg_fp
from utils import losses, ramps
from val_2D import test_single_volume

from copy import deepcopy
from utils.transform import blur, obtain_cutmix_box
from utils.dynamic_thresholds import DynamicConfidenceThreshold_max_percentile

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/SynMatch', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UNet_feature_dropout', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--save_code_snapshot', type=int, default=1, choices=[0, 1],
                    help='whether to copy code into snapshot_path/code')
args = parser.parse_args()
config = get_config(args)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()
    # model2 = ViT_seg_fp(config, img_size=args.patch_size,
    #                  num_classes=args.num_classes).cuda()
    # model2.load_from(config)


    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    


    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss_with_threshold(n_classes = num_classes) 

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)


    auto_labeled_ratio = labeled_slice / float(total_slices) if total_slices > 0 else 0.0
    auto_labeled_ratio = max(1e-6, min(1.0, auto_labeled_ratio))
    dynamic_threshold_manager = DynamicConfidenceThreshold_max_percentile(
        alpha=0.9,
        percentile=90,
        labeled_ratio=auto_labeled_ratio,
        num_classes=num_classes
    )
    logging.info(
        "Dynamic threshold init: alpha=%.2f, percentile=%d, labeled_ratio=%.6f (%d/%d)",
        0.9, 90, auto_labeled_ratio, labeled_slice, total_slices
    )


    def split_and_shuffle_data(volume_batch, labeled_bs):
        labeled_data = volume_batch[:labeled_bs] 
        unlabeled_data = volume_batch[labeled_bs:]  
        random_idx = np.random.permutation(unlabeled_data.shape[0]) 
        unlabeled_data_shuffled = unlabeled_data[random_idx] 
        combined_data = np.concatenate((labeled_data, unlabeled_data_shuffled), axis=0)
        return combined_data


    for epoch_num in iterator:

        # loader = zip(trainloader, trainloader_u)

        for i_batch, sampled_batch in enumerate(trainloader):  


            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch_s1 = deepcopy(volume_batch) 
            volume_batch_s1_mix = deepcopy(volume_batch)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()


            volume_batch_s2 = deepcopy(volume_batch_s1) 


            volume_batch_s1_mix = torch.tensor(split_and_shuffle_data(volume_batch_s1_mix, args.labeled_bs), dtype=torch.float32)
            volume_batch_s2_mix = deepcopy(volume_batch_s1_mix) 

            volume_batch_u = deepcopy(volume_batch_s1_mix).cuda()


            if random.random() < 0.8:
                volume_batch_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(volume_batch_s1)
            volume_batch_s1 = blur(volume_batch_s1, p=0.5)
            cutmix_box1 = obtain_cutmix_box(args.patch_size[0], p=0.5)
            img_s1 = torch.from_numpy(np.array(volume_batch_s1)).float() 


                

            if random.random() < 0.8:
                volume_batch_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(volume_batch_s2)
            volume_batch_s2 = blur(volume_batch_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(args.patch_size[0], p=0.5)
            img_s2 = torch.from_numpy(np.array(volume_batch_s2)).float() 



            if random.random() < 0.8:
                volume_batch_s1_mix = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(volume_batch_s1_mix)
            volume_batch_s1_mix = blur(volume_batch_s1_mix, p=0.5)
            img_s1_mix = torch.from_numpy(np.array(volume_batch_s1_mix)).float()  

                

            if random.random() < 0.8:
                volume_batch_s2_mix = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(volume_batch_s2_mix)
            volume_batch_s2_mix = blur(volume_batch_s2_mix, p=0.5)
            img_s2_mix = torch.from_numpy(np.array(volume_batch_s2_mix)).float()  


            
            img_s1[cutmix_box1.unsqueeze(0).unsqueeze(0).expand(img_s2.shape) == 1] = \
            img_s1_mix[cutmix_box1.unsqueeze(0).unsqueeze(0).expand(img_s2.shape) == 1] 
            img_s2_mix[cutmix_box2.unsqueeze(0).unsqueeze(0).expand(img_s2.shape) == 1] 


            outputs1, outputs1_fp = model1(volume_batch, need_fp=True)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_fp_soft1 = torch.softmax(outputs1_fp, dim=1)

            outputs2, outputs2_fp = model2(volume_batch, need_fp=True)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            outputs_fp_soft2 = torch.softmax(outputs2_fp, dim=1)
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)
            


            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(
                outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            

            with torch.no_grad():
                model1.eval()
                model2.eval()

                outputs_mix1 = model1(volume_batch_u)
                outputs_soft_mix1 = torch.softmax(outputs_mix1, dim=1)
                pseudo_outputs_mix1 = torch.argmax(
                    outputs_soft_mix1[args.labeled_bs:].detach(), dim=1, keepdim=False) 

                outputs_mix2 = model2(volume_batch_u)
                outputs_soft_mix2 = torch.softmax(outputs_mix2, dim=1)
                pseudo_outputs_mix2 = torch.argmax(
                    outputs_soft_mix2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            

            
            conf_outputs1 = outputs_soft1[args.labeled_bs:].detach().max(dim=1)[0]
            conf_outputs2 = outputs_soft2[args.labeled_bs:].detach().max(dim=1)[0]


            conf_outputs_mix1 = outputs_soft_mix1[args.labeled_bs:].detach().max(dim=1)[0]
            conf_outputs_mix2 = outputs_soft_mix2[args.labeled_bs:].detach().max(dim=1)[0]

            model1.train()
            model2.train()


            mask_u_w_cutmixed1_1, mask_u_w_cutmixed1_2, conf_u_w_cutmixed1_1, conf_u_w_cutmixed1_2 = pseudo_outputs1.clone(), pseudo_outputs1.clone(), conf_outputs1.clone(), conf_outputs1.clone()
            mask_u_w_cutmixed2_1, mask_u_w_cutmixed2_2, conf_u_w_cutmixed2_1, conf_u_w_cutmixed2_2 = pseudo_outputs2.clone(), pseudo_outputs2.clone(), conf_outputs2.clone(), conf_outputs2.clone()

            mask_u_w_cutmixed1_1[cutmix_box1.unsqueeze(0).expand(pseudo_outputs1.shape) == 1] = pseudo_outputs_mix1[cutmix_box1.unsqueeze(0).expand(pseudo_outputs1.shape) == 1]
            mask_u_w_cutmixed2_1[cutmix_box1.unsqueeze(0).expand(pseudo_outputs1.shape) == 1] = pseudo_outputs_mix2[cutmix_box1.unsqueeze(0).expand(pseudo_outputs1.shape) == 1]
            conf_u_w_cutmixed1_1[cutmix_box1.unsqueeze(0).expand(pseudo_outputs1.shape) == 1] = conf_outputs_mix1[cutmix_box1.unsqueeze(0).expand(pseudo_outputs1.shape) == 1]
            conf_u_w_cutmixed2_1[cutmix_box1.unsqueeze(0).expand(pseudo_outputs1.shape) == 1] = conf_outputs_mix2[cutmix_box1.unsqueeze(0).expand(pseudo_outputs1.shape) == 1]

            mask_u_w_cutmixed1_2[cutmix_box2.unsqueeze(0).expand(pseudo_outputs1.shape) == 1] = pseudo_outputs_mix1[cutmix_box2.unsqueeze(0).expand(pseudo_outputs1.shape) == 1]
            mask_u_w_cutmixed2_2[cutmix_box2.unsqueeze(0).expand(pseudo_outputs1.shape) == 1] = pseudo_outputs_mix2[cutmix_box2.unsqueeze(0).expand(pseudo_outputs1.shape) == 1]
            conf_u_w_cutmixed1_2[cutmix_box2.unsqueeze(0).expand(pseudo_outputs1.shape) == 1] = conf_outputs_mix1[cutmix_box2.unsqueeze(0).expand(pseudo_outputs1.shape) == 1]
            conf_u_w_cutmixed2_2[cutmix_box2.unsqueeze(0).expand(pseudo_outputs1.shape) == 1] = conf_outputs_mix2[cutmix_box2.unsqueeze(0).expand(pseudo_outputs1.shape) == 1]



            outputs_soft_s1_1, outputs_soft_s1_2 = outputs_soft1[args.labeled_bs:].detach().clone(), outputs_soft1[args.labeled_bs:].detach().clone()
            outputs_soft_s2_1, outputs_soft_s2_2 = outputs_soft2[args.labeled_bs:].detach().clone(), outputs_soft2[args.labeled_bs:].detach().clone()
            outputs_soft1_mix = outputs_soft_mix1[args.labeled_bs:].detach().clone()
            outputs_soft2_mix = outputs_soft_mix2[args.labeled_bs:].detach().clone()
            cutmix_mask1 = cutmix_box1.unsqueeze(0).unsqueeze(0).expand_as(outputs_soft_s1_1) == 1
            cutmix_mask2 = cutmix_box2.unsqueeze(0).unsqueeze(0).expand_as(outputs_soft_s1_1) == 1
            outputs_soft_s1_1[cutmix_mask1] = outputs_soft1_mix[cutmix_mask1] 
            outputs_soft_s1_2[cutmix_mask2] = outputs_soft1_mix[cutmix_mask2] 
            outputs_soft_s2_1[cutmix_mask1] = outputs_soft2_mix[cutmix_mask1]
            outputs_soft_s2_2[cutmix_mask2] = outputs_soft2_mix[cutmix_mask2]

            outputs1_s1 = model1(img_s1.cuda())
            outputs_soft1_s1 = torch.softmax(outputs1_s1, dim=1) 

            outputs2_s1 = model2(img_s1.cuda())
            outputs_soft2_s1 = torch.softmax(outputs2_s1, dim=1)

            all_outputs = [torch.cat([
                outputs_soft2[args.labeled_bs:].detach(),
                outputs_soft1[args.labeled_bs:].detach(),
            ], dim=0)]
            dynamic_threshold_manager.update_statistics(all_outputs)



            dynamic_thresholds = torch.zeros_like(outputs_soft1[args.labeled_bs:].detach())



            for class_id in range(num_classes):
                threshold = dynamic_threshold_manager.get_weight(
                    class_id=class_id
                )

                dynamic_thresholds[:, class_id:class_id + 1, :, :] = threshold
            

            pred_indices_u_s1_1 = outputs_soft_s1_1.detach().argmax(dim=1, keepdim=True)   
            dynamic_thresholds_u_s1_1= torch.gather(dynamic_thresholds.detach(), dim=1, index=pred_indices_u_s1_1)

            pred_indices_u_s2_1 = outputs_soft_s2_1.detach().argmax(dim=1, keepdim=True)
            dynamic_thresholds_u_s2_1= torch.gather(dynamic_thresholds.detach(), dim=1, index=pred_indices_u_s2_1)

            pred_indices_u_s1_2 = outputs_soft_s1_2.detach().argmax(dim=1, keepdim=True)
            dynamic_thresholds_u_s1_2= torch.gather(dynamic_thresholds.detach(), dim=1, index=pred_indices_u_s1_2)

            pred_indices_u_s2_2 = outputs_soft_s2_2.detach().argmax(dim=1, keepdim=True)
            dynamic_thresholds_u_s2_2= torch.gather(dynamic_thresholds.detach(), dim=1, index=pred_indices_u_s2_2)

            pred_indices_fp_1 = outputs_soft1[args.labeled_bs:].detach().argmax(dim=1, keepdim=True)
            dynamic_thresholds_fp1= torch.gather(dynamic_thresholds.detach(), dim=1, index=pred_indices_fp_1)

            pred_indices_fp_2 = outputs_soft2[args.labeled_bs:].detach().argmax(dim=1, keepdim=True)
            dynamic_thresholds_fp2= torch.gather(dynamic_thresholds.detach(), dim=1, index=pred_indices_fp_2)







            pseudo_supervision1_s1 = dice_loss(
                outputs_soft1_s1[args.labeled_bs:], mask_u_w_cutmixed2_1.unsqueeze(1).float(), ignore=(conf_u_w_cutmixed2_1 < dynamic_thresholds_u_s2_1.squeeze(1) ).float())
            pseudo_supervision2_s1 = dice_loss(
                outputs_soft2_s1[args.labeled_bs:], mask_u_w_cutmixed1_1.unsqueeze(1).float(), ignore=(conf_u_w_cutmixed1_1 < dynamic_thresholds_u_s1_1.squeeze(1) ).float())
            
            outputs1_s2 = model1(img_s2.cuda())
            outputs_soft1_s2 = torch.softmax(outputs1_s2, dim=1) 

            outputs2_s2 = model2(img_s2.cuda())
            outputs_soft2_s2 = torch.softmax(outputs2_s2, dim=1)
            
            pseudo_supervision1_s2 = dice_loss(
                outputs_soft1_s2[args.labeled_bs:], mask_u_w_cutmixed2_2.unsqueeze(1).float(), ignore=(conf_u_w_cutmixed2_2 < dynamic_thresholds_u_s2_2.squeeze(1) ).float())
            pseudo_supervision2_s2 = dice_loss(
                outputs_soft2_s2[args.labeled_bs:], mask_u_w_cutmixed1_2.unsqueeze(1).float(), ignore=(conf_u_w_cutmixed1_2 < dynamic_thresholds_u_s1_2.squeeze(1) ).float())
            
            fp_supervision_s1 = dice_loss(
                outputs_fp_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1).float(), ignore=(conf_outputs2 < dynamic_thresholds_fp2.squeeze(1) ).float())
            fp_supervision_s2 = dice_loss(
                outputs_fp_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1).float(), ignore=(conf_outputs1 < dynamic_thresholds_fp1.squeeze(1) ).float())
            
            model1_loss = (loss1 +  ( 0.25 * pseudo_supervision1_s1 + 0.25 * pseudo_supervision1_s2 + 0.5 * fp_supervision_s1)) / 2.0    
            model2_loss = (loss2 +  ( 0.25 * pseudo_supervision2_s1 + 0.25 * pseudo_supervision2_s2 + 0.5 * fp_supervision_s2)) / 2.0

            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1


            dynamic_thresholds_weight = ', '.join([
                f"Class {class_id}: {dynamic_threshold_manager.get_weight(class_id=class_id):.4f}"
                for class_id in range(num_classes) ])



            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info(
                    f'iteration {iter_num} : model1 loss : {model1_loss.item():.4f} '
                    f'model2 loss : {model2_loss.item():.4f} | threshold_weight: {dynamic_thresholds_weight}'
                )
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice',
                                  performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95',
                                  mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    snapshot_code_path = snapshot_path + '/code'
    if args.save_code_snapshot:
        if os.path.exists(snapshot_code_path):
            shutil.rmtree(snapshot_code_path)
        shutil.copytree('.', snapshot_code_path,
                        shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
