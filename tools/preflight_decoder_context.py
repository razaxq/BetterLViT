"""Five actual Train-batch steps with the FORMAL grouped Adam optimizer.

The optional baseline root runs this harness against untouched historical R2.
No checkpoints, Val/Test images, or full training sessions are produced.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--baseline-root', type=Path)
parser.add_argument('--disable-adapter', action='store_true')
parser.add_argument('--observe', action='store_true')
args = parser.parse_args()
root = args.baseline_root or Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader
import Config as config
from Load_Dataset import ImageToImage2D, RandomGenerator
from nets.BetterLViT import BetterLViT
from train_model import build_optimizer_parameter_groups
from utils import read_text, WeightedDiceFocal


def digest(values):
    h = hashlib.sha256()
    for name, value in values:
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = ImageToImage2D(config.train_dataset, config.task_name,
        read_text(os.path.join(config.task_dataset, 'Train_Val_text.xlsx')),
        RandomGenerator([224,224]), image_size=224)
    batch, names = next(iter(DataLoader(dataset, batch_size=16, num_workers=0)))
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = BetterLViT(config.get_CTranS_config(), use_lora=False,
        text_encoder_name=config.text_encoder_name, text_seq_len=config.text_max_len).cuda().train()
    if args.disable_adapter:
        model.decoder_context = None
    adapter = getattr(model, 'decoder_context', None)
    if adapter is not None:
        adapter.capture_stats = args.observe
    initial_base = digest((n,p) for n,p in model.state_dict().items() if not n.startswith('decoder_context.'))
    initial_adapter = digest(adapter.state_dict().items()) if adapter is not None else None
    initial_rng = dict(cpu=digest([('cpu',torch.get_rng_state())]),
                       cuda=digest([('cuda',torch.cuda.get_rng_state())]))
    image, ids, mask, target = (batch[k].cuda() for k in ('image','input_ids','attention_mask','label'))
    shapes = {}
    residuals = []
    def hook(module, inputs, output):
        shapes.update(feature=list(inputs[0].shape), text=list(inputs[1].shape))
        residuals.append(float((output.detach()-inputs[0].detach()).square().mean().sqrt()))
    if adapter is not None:
        adapter.register_forward_hook(hook)
    groups, _, _ = build_optimizer_parameter_groups(model, config.weight_decay)
    optimizer = torch.optim.Adam(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    objective = WeightedDiceFocal()
    losses, predictions, durations, gradients = [], [], [], []
    for step in range(5):
        torch.cuda.synchronize()
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        pred = model(image, ids, mask)
        loss = objective(pred, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.text_encoder.parameters())
        grads = {n:float(p.grad.abs().max()) for n,p in model.named_parameters()
                 if p.grad is not None and (n.startswith('decoder_context.') or n.startswith('text_module2.'))}
        if adapter is not None:
            assert grads['decoder_context.output.weight'] > 0
            if step:
                assert all(grads['decoder_context.'+n+'.weight'] > 0 for n in ('query','key','value'))
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.time()-started)
        predictions.append(hashlib.sha256(pred.detach().cpu().numpy().tobytes()).hexdigest())
        losses.append(loss.item())
        gradients.append(grads)
    if adapter is not None:
        assert shapes == dict(feature=[16,128,56,56], text=[16,32,128])
        assert residuals[0] == 0 and all(r > 0 for r in residuals[1:])
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    print(json.dumps(dict(status='ok', source_git_commit=sha, seed=config.seed,
        profile=config.experiment_name, adapter_disabled=args.disable_adapter,
        observation_enabled=args.observe,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_image_sha256=hashlib.sha256(image.detach().cpu().numpy().tobytes()).hexdigest(),
        input_text_sha256=digest([('ids',ids),('mask',mask)]),
        initial_base_sha256=initial_base, initial_adapter_sha256=initial_adapter,
        initial_rng=initial_rng, output_sha256_each_step=predictions, loss_each_step=losses,
        seconds_each_step=durations, steady_seconds_per_batch=sum(durations[2:])/3,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), shapes=shapes,
        adapter_parameters=sum(p.numel() for p in adapter.parameters()) if adapter is not None else 0,
        residual_rms_each_step=residuals, gradient_absmax_each_step=gradients,
        temporary_optimizer_steps=5, formal_training_performed=False,
        test_split_accessed=False, val_split_accessed=False, grouped_adam=True)))


if __name__ == '__main__':
    main()
