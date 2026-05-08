# -*- coding: utf-8 -*-
"""BetterLViT inference / visualization entry script.

Set `betterlvit.config.test_session` (via Config edit) to the session folder
containing the checkpoint to evaluate. Run from project root after
`pip install -e .` or via `python -m scripts.test`.
"""
# CUDA_VISIBLE_DEVICES must be set before any torch import.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from betterlvit import config as config
from betterlvit.data.dataset import ValGenerator, ImageToImage2D
from betterlvit.io import read_text
from betterlvit.models.better_lvit import BetterLViT

warnings.filterwarnings("ignore")


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save, (448, 448))
        predict_save = cv2.resize(predict_save, (2000, 2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path, predict_save * 255)
    else:
        cv2.imwrite(save_path, predict_save * 255)
    return dice_pred, iou_pred


def vis_and_save_heatmap(model, input_img, input_ids, attention_mask, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda(), input_ids.cuda(), attention_mask.cuda())
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs,
                                                  save_path=vis_save_path + '_predict' + model_type + '.jpg')
    return dice_pred_tmp, iou_tmp


if __name__ == '__main__':
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--session', type=str, default=None,
                     help='Override config.test_session for one-off invocations.')
    _args = _ap.parse_args()
    if _args.session:
        config.test_session = _args.session
    test_session = config.test_session

    model_type = config.model_name
    if config.task_name == "MoNuSeg":
        test_num = 14
        model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "Covid19":
        test_num = 2113
        model_path = "./Covid19/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type in ('LViT', 'BetterLViT'):
        config_vit = config.get_CTranS_config()
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


    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
       print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
       model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + 'Test_text.xlsx')
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            # print(names)
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            test_input_ids = sampled_batch['input_ids']
            test_attention_mask = sampled_batch['attention_mask']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path + str(names) + "_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, test_input_ids, test_attention_mask,
                                                           None, lab,
                                                           vis_path + str(names),
                                                           dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)
