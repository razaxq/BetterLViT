# -*- coding: utf-8 -*-
"""Run-directory resolution: git-commit-hash-keyed folders + auto-resume.

Replaces the legacy `Test_session_MM.DD_HHhMM` timestamp scheme with a
deterministic naming based on (git short hash, config hash). When the same
commit + same config is re-run, the existing folder is reused and the previous
`last_model-<model>.pth.tar` becomes the auto-resume source — so that
crash/Ctrl-C recovery doesn't require manually editing `config.resume_path`.
"""
import hashlib
import inspect
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]


# Path-derived / non-training-behavior fields excluded from the config hash.
# Including these would break reuse:
#   - path fields: derived from save_path itself (chicken-and-egg)
#   - resume_path / resume_max_dice: changing them would force a new folder,
#     defeating the point of automatic resume
#   - enable_bark / shutdown_after_train: side-effect switches that do not
#     affect training output; toggling them between runs should still reuse
#     the same folder
_EXCLUDE = {
    'session_name', 'save_path', 'model_path', 'tensorboard_folder',
    'logger_path', 'visualize_path', 'test_session',
    'resume_path', 'resume_max_dice',
    'enable_bark', 'shutdown_after_train',
}


@dataclass
class RunInfo:
    session_name: str
    save_path: str
    config_hash: str
    auto_resume_path: Optional[str]


def _git_short_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.SubprocessError, OSError):
        return None


def _stable_config_hash(config_globals: dict, module_name: str) -> str:
    items = {}
    for k, v in config_globals.items():
        if k.startswith('_'):
            continue
        if k in _EXCLUDE:
            continue
        if inspect.ismodule(v):
            continue
        if callable(v):
            # Only call no-arg functions defined in *this* module. Imported
            # callables (e.g. ml_collections.ConfigDict) and parameterised
            # functions are skipped. This catches get_CTranS_config so the
            # CTrans structural hyper-parameters (KV_size, num_heads,
            # num_layers, expand_ratio, patch_sizes, base_channel, dropouts)
            # do participate in the hash.
            if getattr(v, '__module__', None) != module_name:
                continue
            try:
                sig = inspect.signature(v)
                if any(p.default is inspect.Parameter.empty
                       and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                      inspect.Parameter.KEYWORD_ONLY)
                       for p in sig.parameters.values()):
                    continue
                called = v()
            except Exception:
                continue
            if hasattr(called, 'to_dict'):
                items[k] = called.to_dict()
            else:
                try:
                    items[k] = dict(called)
                except (TypeError, ValueError):
                    items[k] = repr(called)
            continue
        items[k] = v

    blob = json.dumps(items, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:8]


def _last_model_ckpt(folder: str, model_name: str) -> Optional[str]:
    ckpt = os.path.join(folder, 'models', 'last_model-{}.pth.tar'.format(model_name))
    return ckpt if os.path.isfile(ckpt) else None


def resolve_run_dir(task_name: str, model_name: str,
                    config_globals: dict) -> RunInfo:
    """Decide the run folder for this invocation.

    Atomically claims the folder via O_CREAT|O_EXCL on `config_hash.txt` so
    two processes starting at the same second can't both wrongly perceive
    the folder as 'unowned'.
    """
    cfg_hash = _stable_config_hash(
        config_globals,
        config_globals.get('__name__', 'betterlvit.config'),
    )
    git_id = _git_short_hash()

    if git_id is None:
        session = 'Test_session_' + time.strftime('%m.%d_%Hh%M')
        folder = '{}/{}/{}/'.format(task_name, model_name, session)
        os.makedirs(folder, exist_ok=True)
        return RunInfo(session, folder, cfg_hash, None)

    base_dir = '{}/{}'.format(task_name, model_name)
    candidates = [git_id, '{}_{}'.format(git_id, cfg_hash)]

    for name in candidates:
        folder = '{}/{}/'.format(base_dir, name)
        os.makedirs(folder, exist_ok=True)
        hash_file = os.path.join(folder, 'config_hash.txt')

        try:
            with open(hash_file, 'x') as f:
                f.write(cfg_hash)
            return RunInfo(name, folder, cfg_hash, None)
        except FileExistsError:
            pass

        with open(hash_file) as f:
            existing = f.read().strip()
        if existing == cfg_hash:
            return RunInfo(name, folder, cfg_hash,
                           _last_model_ckpt(folder, model_name))

    fallback = '{}_{}_{}'.format(git_id, cfg_hash, int(time.time()))
    folder = '{}/{}/'.format(base_dir, fallback)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'config_hash.txt'), 'w') as f:
        f.write(cfg_hash)
    return RunInfo(fallback, folder, cfg_hash, None)
