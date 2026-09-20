"""Regression checks for host-synchronization runtime optimizations."""

import os
import sys

import numpy as np
import torch


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from Load_Dataset import _tokenize, _tokenize_all  # noqa: E402
from utils import (  # noqa: E402
    WeightedDiceFocal,
    iou_on_batch,
    iou_on_batch_gpu,
)


class DeterministicTokenizer:
    """Small tokenizer stand-in used to compare scalar and batched paths."""

    def __call__(
        self,
        texts,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        del padding, truncation, return_tensors
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        masks = []
        for text in texts:
            values = [ord(character) % 97 + 1 for character in text]
            values = values[:max_length]
            attention = [1] * len(values)
            padding_length = max_length - len(values)
            rows.append(values + [0] * padding_length)
            masks.append(attention + [0] * padding_length)
        return {
            'input_ids': torch.tensor(rows, dtype=torch.long),
            'attention_mask': torch.tensor(masks, dtype=torch.long),
        }


def check_iou_equivalence():
    generator = torch.Generator().manual_seed(1219)
    probabilities = torch.rand((16, 1, 32, 32), generator=generator)
    masks = torch.randint(
        0,
        2,
        (16, 1, 32, 32),
        generator=generator,
    )
    probabilities[0].zero_()
    masks[0].zero_()
    probabilities[1].fill_(1.0)
    masks[1].zero_()

    legacy = float(iou_on_batch(masks.clone(), probabilities.clone()))
    vectorized = float(
        iou_on_batch_gpu(masks, probabilities).cpu().item()
    )
    np.testing.assert_allclose(vectorized, legacy, rtol=0.0, atol=1e-7)


def check_tokenization_equivalence():
    tokenizer = DeterministicTokenizer()
    texts = ['normal', 'bilateral opacity', '', 'left base']
    batched_ids, batched_masks = _tokenize_all(tokenizer, texts, 12)
    for index, text in enumerate(texts):
        scalar_ids, scalar_masks = _tokenize(tokenizer, text, 12)
        torch.testing.assert_close(batched_ids[index], scalar_ids)
        torch.testing.assert_close(batched_masks[index], scalar_masks)


def check_loss_components_stay_on_device():
    generator = torch.Generator().manual_seed(1219)
    inputs = torch.rand(
        (4, 1, 16, 16),
        generator=generator,
        requires_grad=True,
    )
    targets = torch.randint(
        0,
        2,
        (4, 1, 16, 16),
        generator=generator,
    ).float()
    criterion = WeightedDiceFocal()
    loss = criterion(inputs, targets)
    loss.backward()
    assert inputs.grad is not None
    for name, value in criterion.last_components.items():
        assert torch.is_tensor(value), name
        assert value.ndim == 0, name
        assert not value.requires_grad, name


def main():
    check_iou_equivalence()
    check_tokenization_equivalence()
    check_loss_components_stay_on_device()
    print('Runtime optimization regression checks passed.')


if __name__ == '__main__':
    main()
