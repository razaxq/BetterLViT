# -*- coding: utf-8 -*-
import os
import time

import torch.optim

import Config as config
from utils import *

warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'
    end = time.time()
    time_sum, loss_sum = 0, 0
    tv_sum, focal_sum, bd_sum = 0, 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    dices = []
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks = sampled_batch['image'], sampled_batch['label']
        input_ids = sampled_batch['input_ids']
        attention_mask = sampled_batch['attention_mask']

        images, masks = images.cuda(), masks.cuda()
        input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        preds = model(images, input_ids, attention_mask)
        loss_output = criterion(preds, masks.float())  # Loss

        if isinstance(loss_output, tuple):
            out_loss, l_tv, l_focal, l_bd = loss_output
        else:
            out_loss = loss_output
            l_tv, l_focal, l_bd = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()


        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks,preds)

        batch_time = time.time() - end
        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        tv_sum += len(images) * l_tv
        focal_sum += len(images) * l_focal
        bd_sum += len(images) * l_bd
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice

        if i == len(loader):
            avg_div = (config.batch_size * (i - 1) + len(images))
        else:
            avg_div = (i * config.batch_size)

        average_loss = loss_sum / avg_div
        average_tv = tv_sum / avg_div
        average_focal = focal_sum / avg_div
        average_bd = bd_sum / avg_div
        average_time = time_sum / avg_div
        train_iou_average = iou_sum / avg_div
        train_dice_avg = dice_sum / avg_div

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)
            if isinstance(loss_output, tuple):
                writer.add_scalar(logging_mode + '_' + loss_name + '_tv', l_tv.item(), step)
                writer.add_scalar(logging_mode + '_' + loss_name + '_focal', l_focal.item(), step)
                writer.add_scalar(logging_mode + '_' + loss_name + '_bd', l_bd.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg, train_iou_average, average_tv, average_focal, average_bd
