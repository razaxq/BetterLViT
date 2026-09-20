"""Deterministic whole-model preflight using real augmented Train data only."""
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader
import Config as config
from Load_Dataset import ImageToImage2D, RandomGenerator
from nets.BetterLViT import BetterLViT
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
    initial_base = digest((n,p) for n,p in model.state_dict().items() if not n.startswith('visual_prior.'))
    image, ids, mask, target = (batch[k].cuda() for k in ('image','input_ids','attention_mask','label'))
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-4)
    objective = WeightedDiceFocal()
    losses, predictions, durations = [], [], []
    for step in range(5):
        torch.cuda.synchronize()
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        pred = model(image, ids, mask)
        loss = objective(pred, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if model.visual_prior is not None:
            assert not model.visual_prior.encoder.training
            assert all(p.grad is None for p in model.visual_prior.encoder.parameters())
            assert model.visual_prior.project[-1].weight.grad.abs().max() > 0
            if step:
                assert model.visual_prior.project[0].weight.grad.abs().max() > 0
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.time() - started)
        predictions.append(hashlib.sha256(pred.detach().cpu().numpy().tobytes()).hexdigest())
        losses.append(loss.item())
    print(json.dumps({'status':'ok','experiment':config.experiment_name,'test_split_accessed':False,
        'temporary_optimizer_steps':5,'formal_training_performed':False,'batch_size':16,
        'initial_base_sha256':initial_base,'first_output_sha256':predictions[0],
        'output_sha256_each_step':predictions,'loss_each_step':losses,
        'seconds_each_step':durations,'steady_seconds_per_batch':sum(durations[2:])/3,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'visual_prior':model.visual_prior.provenance if model.visual_prior is not None else None}))


if __name__ == '__main__':
    main()
