"""Deterministic learnability, baseline parity and real Train-batch preflight."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('BETTERLVIT_EXPERIMENT', 'p11_dual_grain')
import numpy as np
import torch
from torch.utils.data import DataLoader
import Config as config
from nets.dual_grain import DualGrainGuide
from nets.LViT import LViT


def seed_all():
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


def require_gradient(parameter, name):
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert parameter.grad.abs().max() > 0, name


def unit_checks():
    seed_all()
    module = DualGrainGuide(64, 8).cuda()
    visual = torch.randn(2, 64, 224, 224, device='cuda')
    coarse = torch.randn(2, 196, 64, device='cuda')
    guide = torch.randn_like(visual)
    target = torch.randn_like(visual)
    assert torch.equal(module(visual, coarse, guide), guide)
    optimizer = torch.optim.Adam(module.parameters(), lr=3e-4)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = module(visual, coarse, guide)
        (output - target).square().mean().backward()
        require_gradient(module.output_proj.weight, 'output projection')
        if step == 1:
            for name, parameter in module.named_parameters():
                require_gradient(parameter, name)
        optimizer.step()
    module.eval()
    assert torch.equal(module(visual, coarse, guide), module(visual, coarse, guide))
    residual = module(visual, coarse, guide) - guide
    # Fine neighbors inside one original 16x16 patch must be distinguishable.
    assert (residual[:, :, 0, 0] - residual[:, :, 0, 8]).abs().max() > 0
    assert not torch.equal(module(visual, coarse, guide), module(visual, coarse + torch.randn_like(coarse), guide))
    return {'identity_at_initialization': True, 'all_gradients_nonzero_after_second_update': True,
            'subpatch_variation': True, 'coarse_context_used': True}


def initialization_parity():
    cfg = config.get_CTranS_config()
    cfg.dual_grain_enabled = False
    torch.manual_seed(config.seed)
    baseline = LViT(cfg, text_seq_len=config.text_max_len)
    baseline_rng = torch.get_rng_state().clone()
    cfg.dual_grain_enabled = True
    torch.manual_seed(config.seed)
    candidate = LViT(cfg, text_seq_len=config.text_max_len)
    assert torch.equal(baseline_rng, torch.get_rng_state())
    candidate_state = candidate.state_dict()
    assert all(torch.equal(value, candidate_state[name]) for name, value in baseline.state_dict().items())
    return {'common_initial_weights_identical': True, 'initialization_rng_identical': True}


def real_batch():
    from Load_Dataset import ImageToImage2D, RandomGenerator
    from nets.BetterLViT import BetterLViT
    from utils import read_text, WeightedDiceFocal
    seed_all()
    dataset = ImageToImage2D(config.train_dataset, config.task_name,
        read_text(os.path.join(config.task_dataset, 'Train_Val_text.xlsx')),
        RandomGenerator([224, 224]), image_size=224)
    assert len(dataset) == 5716
    batch, names = next(iter(DataLoader(dataset, batch_size=16, num_workers=0)))
    seed_all()
    model = BetterLViT(config.get_CTranS_config(), n_channels=3, n_classes=1,
        text_encoder_name=config.text_encoder_name, text_seq_len=32, use_lora=False).cuda()
    batch = {key: value.cuda() for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        model.dual_grain_enabled = False
        control = model(batch['image'], batch['input_ids'], batch['attention_mask'])
        model.dual_grain_enabled = True
        candidate = model(batch['image'], batch['input_ids'], batch['attention_mask'])
        assert torch.equal(control, candidate)
    del control, candidate
    model.train()
    objective = WeightedDiceFocal()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate, weight_decay=config.weight_decay)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['image'], batch['input_ids'], batch['attention_mask'])
        loss = objective(prediction, batch['label'])
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        for index, module in enumerate(model.dual_grain):
            require_gradient(module.output_proj.weight, 'stage {} output'.format(index))
            if step:
                for name, parameter in module.named_parameters():
                    require_gradient(parameter, '{} {}'.format(index, name))
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    return {'batch_size': 16, 'train_samples': len(dataset), 'losses': losses,
        'two_steps_seconds': time.monotonic() - start,
        'output_sha256': hashlib.sha256(prediction.detach().cpu().numpy().tobytes()).hexdigest(),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'added_parameters': sum(p.numel() for p in model.dual_grain.parameters()),
        'test_split_accessed': False, 'formal_training_performed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and config.dual_grain_enabled
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'unit': unit_checks(), 'initialization': initialization_parity()}
    gc.collect()
    torch.cuda.empty_cache()
    result['real_batch'] = real_batch()
    result['status'] = 'ok'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
