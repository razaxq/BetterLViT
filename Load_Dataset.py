# -*- coding: utf-8 -*-
import os
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import AutoTokenizer

import Config as config

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def to_long_tensor(pic):
    img = torch.from_numpy(np.array(pic, np.uint8))
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def _build_tokenizer():
    return AutoTokenizer.from_pretrained(config.text_encoder_name, trust_remote_code=True)


def _tokenize(tokenizer, text, max_len):
    encoded = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)


def _to_chw_float(image_uint8_tensor):
    """ToTensorV2 returns uint8 [C, H, W]; convert to float in [0, 1]."""
    return image_uint8_tensor.float() / 255.0


class RandomGenerator(object):
    """Mild-augmentation training transform built on albumentations.

    Geometric ops are applied jointly to image and mask so the spatial
    correspondence is preserved (mask uses nearest-neighbour interpolation
    where supported). Intensity / noise / blur ops only touch the image.

    Strengths and probabilities are roughly half of the original strong
    pipeline. The strong version capped val Dice ~0.806 with train Dice
    stuck at ~0.79 — clear underfitting from over-aggressive augmentation.
    This milder budget keeps the regularisation benefit while letting the
    model actually fit the training distribution.
    """

    def __init__(self, output_size):
        h, w = output_size[0], output_size[1]
        self.transform = A.Compose([
            # ----- Geometric (sync to mask) -----
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(
                rotate=(-15, 15),
                scale=(0.95, 1.05),
                translate_percent=(0.0, 0.02),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fit_output=False,
                p=0.5,
            ),
            A.ElasticTransform(
                alpha=60,
                sigma=6,
                interpolation=cv2.INTER_LINEAR,
                p=0.2,
            ),
            # ----- Intensity (image only) -----
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5,
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            # ----- Noise / blur (image only) -----
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            # ----- Final size lock + tensor conversion -----
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ])

    def __call__(self, sample):
        image = np.ascontiguousarray(sample['image'].astype(np.uint8))
        mask = sample['label'].astype(np.uint8)
        # mask is (H, W, 1) from correct_dims; albumentations wants (H, W) for binary mask
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        out = self.transform(image=image, mask=mask)
        image_t = _to_chw_float(out['image'])
        mask_t = out['mask'].long()
        return {'image': image_t, 'label': mask_t,
                'input_ids': sample['input_ids'],
                'attention_mask': sample['attention_mask']}


class ValGenerator(object):
    """Validation/test transform: only enforce target size + tensor convert."""

    def __init__(self, output_size):
        h, w = output_size[0], output_size[1]
        self.transform = A.Compose([
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ])

    def __call__(self, sample):
        image = np.ascontiguousarray(sample['image'].astype(np.uint8))
        mask = sample['label'].astype(np.uint8)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        out = self.transform(image=image, mask=mask)
        image_t = _to_chw_float(out['image'])
        mask_t = out['mask'].long()
        return {'image': image_t, 'label': mask_t,
                'input_ids': sample['input_ids'],
                'attention_mask': sample['attention_mask']}


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.tokenizer = _build_tokenizer()
        self.text_max_len = config.text_max_len

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        input_ids, attention_mask = _tokenize(self.tokenizer, text, self.text_max_len)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'label': mask, 'input_ids': input_ids, 'attention_mask': attention_mask}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.tokenizer = _build_tokenizer()
        self.text_max_len = config.text_max_len

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        # image_filename = self.images_list[idx]  # MoNuSeg
        # mask_filename = image_filename[: -3] + "png"  # MoNuSeg
        mask_filename = self.mask_list[idx]  # Covid19
        image_filename = mask_filename.replace('mask_', '')  # Covid19
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        text = self.rowtext[mask_filename]
        input_ids, attention_mask = _tokenize(self.tokenizer, text, self.text_max_len)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask,
                  'input_ids': input_ids, 'attention_mask': attention_mask}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename
