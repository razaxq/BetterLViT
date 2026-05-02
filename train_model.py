# -*- coding: utf-8 -*-
import logging
import numpy as np
import os
import requests
import random
import time
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import Config as config
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
from Train_one_epoch import train_one_epoch
from nets.BetterLViT import BetterLViT
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, read_text


def bark_notify(body, title="训练通知"):
    """极简版：只发送标题和文字内容"""
    bark_key = "uAnJRvt7pxbzE9KK6bCVva"
    url = f"https://api.day.app/{bark_key}/{title}/{body}"
    try:
        # 短 timeout: 网络不可达时直接放过，避免训练脚本被 Bark 阻塞
        requests.get(url, timeout=3)
    except Exception as e:
        print(f"推送失败: {e}")

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
    elif config.task_name == 'Covid19':
        text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf, image_size=config.img_size)


    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
                             
    lr = config.learning_rate
    logger.info(model_type)

    if model_type in ('LViT', 'BetterLViT'):
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        # 'LViT' = frozen CXR-BERT baseline, 'BetterLViT' = LoRA-tuned CXR-BERT
        use_lora = config.text_use_lora and (model_type == 'BetterLViT')
        model = BetterLViT(
            config_vit,
            n_channels=config.n_channels,
            n_classes=config.n_labels,
            text_encoder_name=config.text_encoder_name,
            text_seq_len=config.text_max_len,
            use_lora=use_lora,
            lora_r=config.text_lora_r,
            lora_alpha=config.text_lora_alpha,
            lora_dropout=config.text_lora_dropout,
        )

    elif model_type == 'LViT_pretrain':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = BetterLViT(
            config_vit,
            n_channels=config.n_channels,
            n_classes=config.n_labels,
            text_encoder_name=config.text_encoder_name,
            text_seq_len=config.text_max_len,
            use_lora=config.text_use_lora,
            lora_r=config.text_lora_r,
            lora_alpha=config.text_lora_alpha,
            lora_dropout=config.text_lora_dropout,
        )
        pretrained_UNet_model_path = "MoNuSeg/LViT/Test_session_05.23_10h55/models/best_model-LViT.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)
        logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')
    # thop is incompatible with PEFT-wrapped modules (double-registration
    # leaves stale CPU hooks that break training after .cuda()). Report
    # parameter counts directly — FLOPs aren't needed for the LoRA setup.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params: {} ({:.2f}M)'.format(total, total / 1e6))
    print('trainable params: {} ({:.2f}M, {:.2%})'.format(
        trainable, trainable / 1e6, trainable / max(total, 1)))
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    epoch_history = []
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # Capture LR used for this epoch (scheduler steps inside the val call, so
        # snapshotting before train gives the actual learning rate this epoch ran on)
        epoch_lr = min(g["lr"] for g in optimizer.param_groups)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_loss, train_dice, train_iou = train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None,
                                                            model_type, logger)  # sup

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice, val_iou = train_one_epoch(val_loader, model, criterion,
                                                          optimizer, writer, epoch, lr_scheduler, model_type, logger)
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 5:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
                bark_notify(f"当前最高 Dice 刷新为: {max_dice:.4f}！", title="nb 兄弟")
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_dice': float(train_dice),
            'train_iou': float(train_iou),
            'val_loss': float(val_loss),
            'val_dice': float(val_dice),
            'val_iou': float(val_iou),
            'lr': float(epoch_lr),
        })
        logger.info('--- Epoch History (1..{}) ---'.format(epoch + 1))
        logger.info('{:>5} | {:>10} | {:>10} | {:>9} | {:>10} | {:>10} | {:>9} | {:>10} | {:>4}'.format(
            'Epoch', 'TrainLoss', 'TrainDice', 'TrainIoU', 'ValLoss', 'ValDice', 'ValIoU', 'LR', 'Best'))
        for h in epoch_history:
            marker = '*' if h['epoch'] == best_epoch else ''
            logger.info('{:>5d} | {:>10.4f} | {:>10.4f} | {:>9.4f} | {:>10.4f} | {:>10.4f} | {:>9.4f} | {:>10.2e} | {:>4}'.format(
                h['epoch'], h['train_loss'], h['train_dice'], h['train_iou'],
                h['val_loss'], h['val_dice'], h['val_iou'], h['lr'], marker))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    print("[boot] entered __main__, sending Bark start notification...", flush=True)
    bark_notify("模型开始训练了，请耐心等待！", title="🚀 训练开始")
    print("[boot] Bark call returned, continuing setup...", flush=True)
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
    bark_notify("训练完成！服务器即将自动关机 💤", title="✅ 训练结束")
    print("正在执行关机程序...")
    os.system("shutdown")
