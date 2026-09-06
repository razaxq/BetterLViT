"""Real training batch CUDA forward/backward, without optimizer or Test access."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('BETTERLVIT_EXPERIMENT', 'p9_race_pe')
import numpy as np
import torch
from torch.utils.data import DataLoader
import Config as config
from Load_Dataset import ImageToImage2D, RandomGenerator
from nets.BetterLViT import BetterLViT
from utils import read_text, WeightedDiceFocal, RACEObjective
from race_pe_objective import RACEPEObjective


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
        RandomGenerator([224, 224]), image_size=224)
    batch, names = next(iter(DataLoader(dataset, batch_size=16, num_workers=0)))
    # Model seed independent of architecture-specific data preparation.
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = BetterLViT(config.get_CTranS_config(), n_channels=3, n_classes=1,
        text_encoder_name=config.text_encoder_name, text_seq_len=config.text_max_len,
        use_lora=False).cuda().train()
    b = {key: value.cuda() for key, value in batch.items()}
    pred = model(b['image'], b['input_ids'], b['attention_mask'],
        return_aux=config.race_enabled, race_slot_targets=b['race_slot_targets'],
        race_zone_basis=b['race_zone_basis'])
    if config.race_pe_enabled:
        objective = RACEPEObjective(pixel_only=config.race_pe_pixel_only)
    elif config.race_enabled:
        objective = RACEObjective(aux_weight=.05)
    else:
        objective = WeightedDiceFocal()
    loss = objective(pred, b['label'])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if config.race_pe_enabled:
        for route in model.race.routes:
            assert route.evidence[-1].weight.grad.abs().max() > 0
            if not config.race_pe_pixel_only:
                assert route.presence.weight.grad.abs().max() > 0
            if config.race_pe_route_enabled:
                assert route.strength_logit.grad.abs().max() > 0
    final = pred['final'] if isinstance(pred, dict) else pred
    print(json.dumps({'status': 'ok', 'experiment': config.experiment_name,
        'batch_size': 16, 'training_performed': False, 'test_split_accessed': False,
        'loss': loss.item(), 'output_sha256': hashlib.sha256(
            final.detach().cpu().numpy().tobytes()).hexdigest(),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}))


if __name__ == '__main__':
    main()
