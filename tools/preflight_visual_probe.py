"""Real Train batch contract tests and bounded timing, before the diagnostic run."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_visual_prior import FixedImages, deterministic, head, predict, write_json
from nets.frozen_visual import FrozenVisualEncoder, FrozenVisualPrior
from utils import WeightedDiceFocal
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    deterministic(1219)
    data = FixedImages(args.data, 'Train_Folder')
    image, label, _ = next(iter(DataLoader(data, batch_size=16, num_workers=0)))
    image, label = image.cuda(), label.cuda()
    results = {}
    for kind in ('cxformer', 'dinov2'):
        encoder = FrozenVisualEncoder(args.models, kind).cuda()
        encoder.train()
        assert not encoder.training and not encoder.backbone.training
        with torch.no_grad():
            feature = encoder(image)
            assert torch.equal(feature, encoder(image))
            torch.cuda.synchronize()
            started = time.time()
            for _ in range(20):
                encoder(image)
            torch.cuda.synchronize()
            seconds = (time.time() - started) / 20
        assert not feature.requires_grad and feature.shape == (16, 384, 16, 16)
        results[kind] = {'feature_sha256': hashlib.sha256(feature.cpu().numpy().tobytes()).hexdigest(),
            'seconds_per_forward_batch16': seconds, 'provenance': encoder.provenance}
        del encoder
    readout = head().cuda()
    opt = torch.optim.AdamW(readout.parameters(), lr=3e-4)
    objective = WeightedDiceFocal()
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = objective(predict(readout, feature), label)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    seconds = (time.time() - started) / 40
    assert torch.isfinite(loss)
    del readout, opt
    before = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    adapter = FrozenVisualPrior(args.models).cuda().train()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(before_cuda, torch.cuda.get_rng_state())
    x3 = torch.randn(16, 256, 56, 56, device='cuda', requires_grad=True)
    assert torch.equal(adapter(x3, image), x3)
    opt = torch.optim.AdamW(adapter.project.parameters(), lr=3e-4)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        adapter(x3, image).square().mean().backward()
        opt.step()
    assert adapter.project[0].weight.grad.abs().max() > 0
    assert all(p.grad is None for p in adapter.encoder.parameters())
    estimated = sum(r['seconds_per_forward_batch16'] for r in results.values()) * 447 + 2 * 20 * 357 * seconds
    results.update(status='ok', test_split_accessed=False, timestamp_unix=time.time(),
        seconds_per_probe_step_batch16=seconds, device_only_estimated_total_seconds=estimated,
        planned_completion_check_delay_seconds=int(estimated * 1.8 + 360),
        adapter_identity_rng_freeze_and_two_step_gradients='passed',
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(args.output, results)
    print(json.dumps(results))


if __name__ == '__main__':
    main()
