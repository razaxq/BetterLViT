# -*- coding: utf-8 -*-
import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
from torchinfo import summary
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
    epoch_start = time.time()
    loss_sum = None
    dice_sum = None
    iou_sum = None
    sample_count = 0
    component_sums = {}
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks = sampled_batch['image'], sampled_batch['label']
        input_ids = sampled_batch['input_ids']
        attention_mask = sampled_batch['attention_mask']

        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)


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
            train_dice = criterion._show_dice(
                preds.detach(),
                masks.float(),
            )
            train_iou = iou_on_batch_gpu(masks, preds.detach())

        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        batch_size = len(images)
        sample_count += batch_size
        detached_loss = out_loss.detach()
        loss_sum = (
            batch_size * detached_loss
            if loss_sum is None
            else loss_sum + batch_size * detached_loss
        )
        iou_sum = (
            batch_size * train_iou
            if iou_sum is None
            else iou_sum + batch_size * train_iou
        )
        dice_sum = (
            batch_size * train_dice
            if dice_sum is None
            else dice_sum + batch_size * train_dice
        )
        for name, value in getattr(
            criterion,
            'last_components',
            {},
        ).items():
            if not torch.is_tensor(value):
                value = detached_loss.new_tensor(value)
            value = value.detach()
            component_sums[name] = (
                batch_size * value
                if name not in component_sums
                else component_sums[name] + batch_size * value
            )

        average_loss = loss_sum / sample_count
        train_iou_average = iou_sum / sample_count
        train_dice_avg = dice_sum / sample_count

        should_print = i % config.print_frequency == 0 or i == len(loader)
        should_write = (
            config.tensorboard
            and (
                i % config.tensorboard_frequency == 0
                or i == len(loader)
            )
        )
        if should_print or should_write:
            torch.cuda.synchronize()
            average_time = (time.time() - epoch_start) / i
            component_names = list(
                getattr(criterion, 'last_components', {}).keys()
            )
            component_values = [
                getattr(criterion, 'last_components', {})[name]
                for name in component_names
            ]
            snapshot = torch.stack([
                detached_loss,
                average_loss,
                train_iou,
                train_iou_average,
                train_dice,
                train_dice_avg,
                *component_values,
            ]).float().cpu().tolist()
            (
                loss_value,
                average_loss_value,
                iou_value,
                average_iou_value,
                dice_value,
                average_dice_value,
                *component_snapshot,
            ) = snapshot
            component_snapshot = dict(
                zip(component_names, component_snapshot)
            )
        if should_print:
            print_summary(epoch + 1, i, len(loader), loss_value, loss_name, average_time,
                          average_loss_value, average_time, iou_value, average_iou_value,
                          dice_value, average_dice_value, 0, 0, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups), logger=logger)

        if should_write:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, loss_value, step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', iou_value, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', dice_value, step)
            for name, value in component_snapshot.items():
                writer.add_scalar(
                    f"{logging_mode}_loss_{name}",
                    value,
                    step,
                )

    if lr_scheduler is not None:
        lr_scheduler.step()

    torch.cuda.synchronize()
    epoch_names = list(component_sums)
    epoch_values = torch.stack([
        loss_sum / sample_count,
        dice_sum / sample_count,
        iou_sum / sample_count,
        *(component_sums[name] / sample_count for name in epoch_names),
    ]).float().cpu().tolist()
    average_loss, train_dice_avg, train_iou_average, *component_values = (
        epoch_values
    )
    criterion.last_epoch_components = dict(
        zip(epoch_names, component_values)
    )
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
