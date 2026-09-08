"""Fixed-view Train/Val readout probe. Never constructs or reads a Test path."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from nets.frozen_visual import FrozenVisualEncoder, sha256
from utils import WeightedDiceFocal


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class FixedImages(Dataset):
    def __init__(self, root, split):
        if split not in ('Train_Folder', 'Val_Folder'):
            raise ValueError('This probe permits only Train and Val')
        self.folder = root / split
        self.names = sorted(p.name for p in (self.folder / 'labelcol').iterdir() if p.is_file())
        image_names = sorted(p.name for p in (self.folder / 'img').iterdir() if p.is_file())
        assert image_names == [n.replace('mask_', '') for n in self.names]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = cv2.imread(str(self.folder / 'img' / name.replace('mask_', '')))
        mask = cv2.imread(str(self.folder / 'labelcol' / name), 0)
        if image is None or mask is None:
            raise ValueError('Unreadable Train/Val sample: ' + name)
        # Identical resizing and mask binarization to the locked C4 loader.
        image = cv2.resize(image, (224, 224))
        mask = cv2.resize(mask, (224, 224)) > 0
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255, torch.from_numpy(mask), index


class CachedFeatures(Dataset):
    def __init__(self, feature_path, masks_path):
        self.features = np.load(feature_path, mmap_mode='r')
        self.masks = np.load(masks_path, mmap_mode='r')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index].copy()), torch.from_numpy(self.masks[index].copy()), index


def extract(encoder, dataset, cache, kind, split, batch):
    feature_path = cache / f'{kind}_{split}.npy'
    mask_path = cache / f'{kind}_{split}_masks.npy'
    if feature_path.exists() or mask_path.exists():
        raise RuntimeError('Refusing to overwrite an existing feature cache')
    features = np.lib.format.open_memmap(feature_path, mode='w+', dtype='float32', shape=(len(dataset), 384, 16, 16))
    masks = np.lib.format.open_memmap(mask_path, mode='w+', dtype='uint8', shape=(len(dataset), 224, 224))
    loader = DataLoader(dataset, batch_size=batch, num_workers=4, shuffle=False)
    non_gray = 0
    started = time.time()
    with torch.no_grad():
        for image, label, indices in loader:
            non_gray += int(((image[:, 0] != image[:, 1]) | (image[:, 1] != image[:, 2])).flatten(1).any(1).sum())
            values = encoder(image.cuda()).cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite frozen features')
            features[indices.numpy()] = values
            masks[indices.numpy()] = label.numpy()
    features.flush()
    masks.flush()
    del features, masks
    return feature_path, mask_path, {'seconds': time.time() - started, 'non_gray_images': non_gray,
        'samples': len(dataset), 'name_sha256': hashlib.sha256('\n'.join(dataset.names).encode()).hexdigest(),
        'feature_sha256': sha256(feature_path), 'mask_sha256': sha256(mask_path)}


def head():
    return nn.Sequential(nn.Conv2d(384, 64, 1), nn.GELU(), nn.Conv2d(64, 1, 1))


def predict(readout, feature):
    return torch.sigmoid(F.interpolate(readout(feature), size=(224, 224), mode='nearest'))


def evaluate(readout, loader, names):
    readout.eval()
    records = []
    with torch.no_grad():
        for feature, label, indices in loader:
            pred = predict(readout, feature.cuda())[:, 0] > .5
            target = label.cuda() > 0
            tp = (pred & target).sum((1, 2), dtype=torch.float64)
            pp = pred.sum((1, 2), dtype=torch.float64)
            gp = target.sum((1, 2), dtype=torch.float64)
            iou = torch.where(pp + gp - tp > 0, tp / (pp + gp - tp).clamp_min(1), 0.)
            dice = torch.where(pp + gp > 0, 2 * tp / (pp + gp).clamp_min(1), 0.)
            for j, index in enumerate(indices.tolist()):
                records.append({'image': names[index].replace('mask_', ''), 'iou': iou[j].item(),
                    'dice': dice[j].item(), 'tp': tp[j].item(), 'predicted_area': pp[j].item(),
                    'gt_area': gp[j].item()})
    return {'macro_iou': float(np.mean([r['iou'] for r in records])),
            'macro_dice': float(np.mean([r['dice'] for r in records])), 'records': records}


def run(args):
    deterministic(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.mkdir(parents=True, exist_ok=False)
    metadata = {'started_unix': time.time(), 'stage': 'extracting', 'training_performed': False,
        'test_split_accessed': False, 'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
        'protocol': 'fixed224_noHE_FP32_eager_no_augmentation_macro_at_0.5',
        'metric_contract': 'prediction>0.5; undefined/empty IoU and Dice=0, matching C4 export',
        'head': '384->64->1 GELU, native16 grid then nearest224',
        'optimizer': 'AdamW lr=0.0003 weight_decay=0.0001 constant; no scheduler', 'encoders': {}}
    write_json(args.output / 'status.json', metadata)
    datasets = {split: FixedImages(args.data, split) for split in ('Train_Folder', 'Val_Folder')}
    assert len(datasets['Train_Folder']) == 5716 and len(datasets['Val_Folder']) == 1429
    for kind in ('cxformer', 'dinov2'):
        encoder = FrozenVisualEncoder(args.models, kind).cuda().eval()
        cached = {}
        audits = {}
        for split, dataset in datasets.items():
            feature, mask, audit = extract(encoder, dataset, args.cache, kind, split, args.batch_size)
            cached[split] = CachedFeatures(feature, mask)
            audits[split] = audit
        provenance = encoder.provenance
        del encoder
        torch.cuda.empty_cache()
        deterministic(args.seed)
        readout = head().cuda()
        initial_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in readout.parameters())).hexdigest()
        optimizer = torch.optim.AdamW(readout.parameters(), lr=3e-4, weight_decay=1e-4)
        objective = WeightedDiceFocal()
        generator = torch.Generator().manual_seed(args.seed + 1001)
        train = DataLoader(cached['Train_Folder'], batch_size=args.batch_size, shuffle=True,
            num_workers=4, generator=generator, drop_last=True, persistent_workers=True, pin_memory=True)
        val = DataLoader(cached['Val_Folder'], batch_size=args.batch_size, shuffle=False,
            num_workers=2, persistent_workers=True, pin_memory=True)
        best, history = -1., []
        started = time.time()
        metadata.update(stage='training_' + kind, training_performed=True)
        write_json(args.output / 'status.json', metadata)
        for epoch in range(1, args.epochs + 1):
            readout.train()
            losses = []
            for feature, label, _ in train:
                optimizer.zero_grad(set_to_none=True)
                loss = objective(predict(readout, feature.cuda(non_blocking=True)), label.cuda(non_blocking=True))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite probe loss')
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            metrics = evaluate(readout, val, datasets['Val_Folder'].names)
            history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                'macro_iou': metrics['macro_iou'], 'macro_dice': metrics['macro_dice'],
                'elapsed_seconds': time.time() - started})
            if metrics['macro_iou'] > best:
                best = metrics['macro_iou']
                metrics.update(best_epoch=epoch, kind=kind, source_git_commit=metadata['source_git_commit'],
                    test_split_accessed=False, split='validation', threshold=.5, encoder=provenance)
                write_json(args.output / (kind + '_validation.json'), metrics)
                torch.save({'state_dict': readout.state_dict(), 'epoch': epoch, 'metadata': metrics},
                           args.output / (kind + '_readout.pt'))
            print(json.dumps({'kind': kind, **history[-1]}), flush=True)
            write_json(args.output / (kind + '_history.json'), history)
        metadata['encoders'][kind] = {'extraction': audits, 'initial_head_sha256': initial_hash,
            'trainable_parameters': sum(p.numel() for p in readout.parameters()), 'training_seconds': time.time() - started,
            'completed_unix': time.time(), 'encoder': provenance}
        write_json(args.output / 'status.json', metadata)
        del train, val, readout, optimizer, cached
        torch.cuda.empty_cache()
    metadata.update(stage='complete', completed_unix=time.time())
    write_json(args.output / 'status.json', metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1219)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.output.exists():
            write_json(args.output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'unix': time.time()})
        raise


if __name__ == '__main__':
    main()
