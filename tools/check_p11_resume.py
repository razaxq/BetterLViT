"""Exercise the real resume loader and one real training batch without saving weights."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('BETTERLVIT_EXPERIMENT', 'p11_dual_grain')
import numpy as np
import torch
import train_model as training


class PreflightComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    training.config.resume_path = str(args.checkpoint)
    training.config.epochs = 150
    training.logger = logging.getLogger('preflight')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    result = {}

    def first_batch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
        assert epoch == 80 and model.training
        assert loader.batch_size == 16 and loader.num_workers == 4
        assert loader.drop_last and not loader.persistent_workers
        caller = sys._getframe(1).f_locals
        assert caller['lr_scheduler'].state_dict() == checkpoint['lr_scheduler']
        assert caller['epoch_history'] == checkpoint['epoch_history']
        assert caller['best_epoch'] == 80
        assert random.getstate() == checkpoint['rng_state']['python']
        assert np.array_equal(np.random.get_state()[1], checkpoint['rng_state']['numpy']['state'].numpy())
        assert torch.equal(torch.get_rng_state(), checkpoint['rng_state']['torch_cpu'])
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), checkpoint['rng_state']['torch_cuda']))
        assert torch.equal(loader.generator.get_state(), checkpoint['rng_state']['train_generator'])
        assert torch.equal(caller['val_generator'].get_state(), checkpoint['rng_state']['val_generator'])
        state = optimizer.state_dict()
        assert state['param_groups'] == checkpoint['optimizer']['param_groups']
        for key, values in state['state'].items():
            for name, value in values.items():
                expected = checkpoint['optimizer']['state'][key][name]
                assert torch.equal(value.cpu(), expected) if torch.is_tensor(value) else value == expected
        assert all(torch.equal(value.cpu(), checkpoint['state_dict'][name]) for name, value in model.state_dict().items())
        iterator = iter(loader)
        batch, names = next(iterator)
        del iterator
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['image'].cuda(), batch['input_ids'].cuda(), batch['attention_mask'].cuda())
        loss = criterion(prediction, batch['label'].float().cuda())
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().numpy().tobytes())
        result.update(status='ok', resumed_epoch=81, history_rows=80,
            model_optimizer_scheduler_rng_restored=True, batch_size=16,
            names=list(names), loss=loss.item(),
            prediction_sha256=hashlib.sha256(prediction.detach().cpu().numpy().tobytes()).hexdigest(),
            updated_trainable_weights_sha256=digest.hexdigest(),
            test_split_accessed=False, checkpoint_written=False)
        raise PreflightComplete()

    training.train_one_epoch = first_batch
    try:
        training.main_loop(model_type='BetterLViT', tensorboard=False)
    except PreflightComplete:
        pass
    assert result['status'] == 'ok'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
