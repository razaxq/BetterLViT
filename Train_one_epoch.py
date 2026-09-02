# -*- coding: utf-8 -*-
import torch.optim
import math
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
    # Windows ROCm 7.2 corrupted long-lived device scalar accumulators during
    # sustained training. Keep epoch aggregates as ordinary host numbers and
    # transfer one compact scalar snapshot per batch.
    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
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
        segmentation_loss = criterion(preds, masks.float())
        router_regularization = segmentation_loss.new_zeros(())
        router_supervision = segmentation_loss.new_zeros(())
        router = None
        if model.training:
            target_model = model.module if hasattr(model, 'module') else model
            router = getattr(target_model, 'tcsr', None)
            if router is not None and hasattr(router, 'regularization_loss'):
                router_regularization = router.regularization_loss()
            if router is not None and hasattr(
                router, 'supervised_localization_loss'
            ):
                localization_warmup = min(1.0, float(epoch + 1) / 5.0)
                router_supervision = router.supervised_localization_loss(
                    masks.float(),
                    weight_scale=localization_warmup,
                )
        out_loss = (
            segmentation_loss
            + router_regularization
            + router_supervision
        )
        with torch.no_grad():
            train_dice = criterion._show_dice(
                preds.detach(),
                masks.float(),
            )
            train_iou = iou_on_batch_gpu(masks, preds.detach())
            component_names = list(
                getattr(criterion, 'last_components', {}).keys()
            )
            component_values = [
                getattr(criterion, 'last_components', {})[name]
                for name in component_names
            ]
            if model.training and router_regularization.detach().item() != 0.0:
                component_names.append('tcsr_regularization')
                component_values.append(router_regularization.detach())
            if model.training and router is not None and hasattr(
                router, 'localization_components'
            ):
                for name, value in router.localization_components().items():
                    component_names.append(name)
                    component_values.append(value)
            snapshot = torch.stack([
                out_loss.detach(),
                train_iou,
                train_dice,
                *component_values,
            ]).float().cpu().tolist()
            (
                loss_value,
                iou_value,
                dice_value,
                *component_values,
            ) = snapshot
            component_snapshot = dict(
                zip(component_names, component_values)
            )

        if not math.isfinite(loss_value) or not 0.0 <= loss_value <= 100.0:
            raise FloatingPointError(
                'Invalid loss at epoch {} batch {}: {}'.format(
                    epoch + 1,
                    i,
                    loss_value,
                )
            )

        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        batch_size = len(images)
        sample_count += batch_size
        loss_sum += batch_size * loss_value
        iou_sum += batch_size * iou_value
        dice_sum += batch_size * dice_value
        for name, value in component_snapshot.items():
            component_sums[name] = (
                component_sums.get(name, 0.0)
                + batch_size * value
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
            average_loss_value = average_loss
            average_iou_value = train_iou_average
            average_dice_value = train_dice_avg
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
    average_loss = loss_sum / sample_count
    train_dice_avg = dice_sum / sample_count
    train_iou_average = iou_sum / sample_count
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
