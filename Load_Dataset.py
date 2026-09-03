# -*- coding: utf-8 -*-
import os
import random
from typing import Callable

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from transformers import AutoTokenizer

import Config as config
from race_semantics import make_zone_basis, parse_report_slots

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        zone_basis = np.asarray(sample['race_zone_basis'], dtype=np.float32)
        x, y = image.shape[1], image.shape[0]
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(0, 1))
            label = np.rot90(label, k, axes=(0, 1))
            zone_basis = np.rot90(zone_basis, k, axes=(1, 2))
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            zone_basis = np.flip(zone_basis, axis=axis + 1).copy()
        elif random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            label = ndimage.rotate(label, angle, order=0, reshape=False)
            zone_basis = ndimage.rotate(
                zone_basis, angle, axes=(1, 2), order=0, reshape=False
            )

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            zone_basis = zoom(
                zone_basis,
                (1, self.output_size[0] / y, self.output_size[1] / x),
                order=0,
            )
        image, label = F.to_pil_image(image.astype(np.uint8)), F.to_pil_image(label.astype(np.uint8))
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        out = {'image': image, 'label': label,
               'input_ids': sample['input_ids'],
               'attention_mask': sample['attention_mask'],
               'race_slot_targets': sample['race_slot_targets'],
               'race_zone_basis': torch.from_numpy(
                   np.ascontiguousarray(zone_basis)
               ).float()}
        return out


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        out = {'image': image, 'label': label,
               'input_ids': sample['input_ids'],
               'attention_mask': sample['attention_mask'],
               'race_slot_targets': sample['race_slot_targets'],
               'race_zone_basis': torch.as_tensor(
                   sample['race_zone_basis'], dtype=torch.float32
               )}
        return out


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


def _tokenize_all(tokenizer, texts, max_len):
    """Tokenize immutable dataset text once instead of once per sample/epoch."""
    encoded = tokenizer(
        list(texts),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    return encoded['input_ids'], encoded['attention_mask']


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = sorted(os.listdir(self.output_path))
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.text_max_len = config.text_max_len
        tokenizer = _build_tokenizer()
        self.input_ids, self.attention_masks = _tokenize_all(
            tokenizer,
            (self.rowtext[name] for name in self.mask_list),
            self.text_max_len,
        )
        self.race_slot_targets = torch.stack([
            parse_report_slots(self.rowtext[name]) for name in self.mask_list
        ])
        self.race_zone_basis = make_zone_basis(image_size, image_size)

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {
            'label': mask,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'race_slot_targets': self.race_slot_targets[idx],
            'race_zone_basis': self.race_zone_basis,
        }

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = sorted(os.listdir(self.input_path))
        self.mask_list = sorted(os.listdir(self.output_path))
        self.rowtext = row_text
        expected_images = [name.replace('mask_', '') for name in self.mask_list]
        if self.images_list != expected_images:
            missing_images = sorted(set(expected_images) - set(self.images_list))
            missing_masks = sorted(set(self.images_list) - set(expected_images))
            raise RuntimeError(
                'Image/mask pairing mismatch: missing_images={}, missing_masks={}'
                .format(missing_images[:10], missing_masks[:10])
            )
        missing_text = [name for name in self.mask_list if name not in self.rowtext]
        if missing_text:
            raise RuntimeError(
                'Missing text rows for {} masks, first entries: {}'.format(
                    len(missing_text), missing_text[:10]
                )
            )
        self.one_hot_mask = one_hot_mask
        self.task_name = task_name
        self.text_max_len = config.text_max_len
        tokenizer = _build_tokenizer()
        self.input_ids, self.attention_masks = _tokenize_all(
            tokenizer,
            (self.rowtext[name] for name in self.mask_list),
            self.text_max_len,
        )
        self.race_slot_targets = torch.stack([
            parse_report_slots(self.rowtext[name]) for name in self.mask_list
        ])
        self.race_zone_basis = make_zone_basis(image_size, image_size)

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.images_list)

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
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask,
                  'input_ids': input_ids, 'attention_mask': attention_mask,
                  'race_slot_targets': self.race_slot_targets[idx],
                  'race_zone_basis': self.race_zone_basis}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename
