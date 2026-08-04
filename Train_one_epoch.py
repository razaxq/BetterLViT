# -*- coding: utf-8 -*-
import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
from torchinfo import summary
from sklearn.metrics.pairwise import cosine_similarity
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
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    sample_count = 0
    component_sums = {}
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
        out_loss = criterion(preds, masks.float())  # Loss
        # print(model.training)


        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_dice = float(
                criterion._show_dice(
                    preds.detach().clone(),
                    masks.float(),
                ).item()
            )
            train_iou = float(iou_on_batch(masks, preds))

        batch_time = time.time() - end
        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        batch_size = len(images)
        sample_count += batch_size
        time_sum += len(images) * batch_time
        loss_sum += batch_size * float(out_loss.detach().item())
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice
        for name, value in getattr(
            criterion,
            'last_components',
            {},
        ).items():
            component_sums[name] = (
                component_sums.get(name, 0.0)
                + batch_size * float(value)
            )

        average_loss = loss_sum / sample_count
        average_time = time_sum / sample_count
        train_iou_average = iou_sum / sample_count
        train_dice_avg = dice_sum / sample_count

        end = time.time()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss.item(), loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)
            for name, value in getattr(
                criterion,
                'last_components',
                {},
            ).items():
                writer.add_scalar(
                    f"{logging_mode}_loss_{name}",
                    value,
                    step,
                )

    if lr_scheduler is not None:
        lr_scheduler.step()

    criterion.last_epoch_components = {
        name: value / sample_count
        for name, value in component_sums.items()
    }
    if criterion.last_epoch_components:
        logger.info(
            '   [{}] Loss components: {}'.format(
                logging_mode,
                ', '.join(
                    '{}={:.6f}'.format(name, value)
                    for name, value
                    in criterion.last_epoch_components.items()
                ),
            )
        )

    return average_loss, train_dice_avg, train_iou_average
