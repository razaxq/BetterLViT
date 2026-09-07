# -*- coding: utf-8 -*-
import os
import random
from pathlib import Path
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
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        out = {'image': image, 'label': label,
               'input_ids': sample['input_ids'],
               'attention_mask': sample['attention_mask']}
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
               'attention_mask': sample['attention_mask']}
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


def _regular_file_names(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError('Dataset directory not found: {}'.format(
            directory
        ))
    return tuple(sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_file()
    ))


def _build_sample_records(dataset_path, row_text, sample_records=None):
    """Build a stable, fully paired image/mask/text index."""
    if sample_records is None:
        dataset_path = Path(dataset_path).resolve()
        input_path = dataset_path / 'img'
        output_path = dataset_path / 'labelcol'
        image_names = _regular_file_names(input_path)
        mask_names = _regular_file_names(output_path)
        if len(set(image_names)) != len(image_names):
            raise RuntimeError('Duplicate image filenames detected')
        if len(set(mask_names)) != len(mask_names):
            raise RuntimeError('Duplicate mask filenames detected')
        expected_images = tuple(
            mask_name[len('mask_'):]
            if mask_name.startswith('mask_')
            else ''
            for mask_name in mask_names
        )
        if not all(expected_images) or set(expected_images) != set(image_names):
            raise RuntimeError(
                'Image/mask pairing is not one-to-one in {}'.format(
                    dataset_path
                )
            )
        records = [
            {
                'image_name': image_name,
                'mask_name': mask_name,
                'text_key': mask_name,
                'image_path': str(input_path / image_name),
                'mask_path': str(output_path / mask_name),
            }
            for mask_name, image_name in zip(mask_names, expected_images)
        ]
    else:
        records = [dict(record) for record in sample_records]
        required = {
            'image_name', 'mask_name', 'text_key', 'image_path', 'mask_path'
        }
        for record in records:
            if not required.issubset(record):
                raise RuntimeError(
                    'Manifest sample record is missing required fields'
                )
        records.sort(key=lambda record: record['image_name'])

    image_names = [str(record['image_name']) for record in records]
    mask_names = [str(record['mask_name']) for record in records]
    if len(set(image_names)) != len(image_names):
        raise RuntimeError('Dataset sample index has duplicate image names')
    if len(set(mask_names)) != len(mask_names):
        raise RuntimeError('Dataset sample index has duplicate mask names')
    for record in records:
        image_name = str(record['image_name'])
        mask_name = str(record['mask_name'])
        text_key = str(record['text_key'])
        if mask_name != 'mask_' + image_name or text_key != mask_name:
            raise RuntimeError(
                'Invalid image/mask/text pairing for {}'.format(image_name)
            )
        if text_key not in row_text:
            raise KeyError('Missing text annotation for {}'.format(text_key))
        if not Path(record['image_path']).is_file():
            raise FileNotFoundError(record['image_path'])
        if not Path(record['mask_path']).is_file():
            raise FileNotFoundError(record['mask_path'])
    return tuple(records)


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = list(_regular_file_names(self.output_path))
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
        (
            self.neutral_input_ids,
            self.neutral_attention_mask,
        ) = _tokenize(
            tokenizer,
            config.text_modality_dropout_prompt,
            self.text_max_len,
        )

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

        sample = {'label': mask, 'input_ids': input_ids, 'attention_mask': attention_mask}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224, sample_records=None) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.sample_records = _build_sample_records(
            dataset_path,
            row_text,
            sample_records=sample_records,
        )
        self.images_list = [
            record['image_name'] for record in self.sample_records
        ]
        self.mask_list = [
            record['mask_name'] for record in self.sample_records
        ]
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.text_max_len = config.text_max_len
        tokenizer = _build_tokenizer()
        self.input_ids, self.attention_masks = _tokenize_all(
            tokenizer,
            (
                self.rowtext[record['text_key']]
                for record in self.sample_records
            ),
            self.text_max_len,
        )
        (
            self.neutral_input_ids,
            self.neutral_attention_mask,
        ) = _tokenize(
            tokenizer,
            config.text_modality_dropout_prompt,
            self.text_max_len,
        )

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
        record = self.sample_records[idx]
        mask_filename = record['mask_name']
        image_filename = record['image_name']
        image = cv2.imread(record['image_path'])
        if image is None:
            raise RuntimeError(
                'Failed to decode image: {}'.format(record['image_path'])
            )
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(record['mask_path'], 0)
        if mask is None:
            raise RuntimeError(
                'Failed to decode mask: {}'.format(record['mask_path'])
            )
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
                  'input_ids': input_ids, 'attention_mask': attention_mask}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename
