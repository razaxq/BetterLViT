# -*- coding: utf-8 -*-
"""Offline precomputation of frozen text embeddings for LViT.

Background
----------
The original LViT pipeline produced per-token BERT embeddings inside the
dataloader via the unmaintained mxnet ``bert-embedding`` package, which does
not run on modern GPUs (e.g. RTX 5090 / CUDA 12). This script reproduces the
*same* behaviour with the *same* weights (``bert-base-uncased``) through
HuggingFace ``transformers``, but computes the embeddings **once, offline**,
and caches them to disk. Training/testing then just reads the cache, so BERT
never enters the training loop and the architecture/recipe are unchanged.

For each text spreadsheet it writes a sibling cache file
``<name>.emb<max_len>.pt`` holding ``{image_key: float32 ndarray (max_len, 768)}``,
keyed exactly like :func:`utils.read_text` (by the ``Image`` column), so the
dataloader lookup ``self.rowtext[mask_filename]`` is unchanged.

Usage
-----
    python tools/precompute_text_emb.py \
        --xlsx datasets/Covid19/Train_Folder/Train_Val_text.xlsx \
        --xlsx datasets/Covid19/Test_Folder/Test_text.xlsx

``--max-len`` defaults to 10 (the cap the original ``ImageToImage2D`` applied
to QaTa-COV19). ``--add-special-tokens`` is OFF by default to mirror the legacy
``bert_embedding`` package, which returned per-word embeddings without
[CLS]/[SEP]; keep it a knob in case the reproduction needs A/B-ing.
"""
import argparse
import os
import sys

import torch
from transformers import BertModel, BertTokenizer

# Reuse the exact text preprocessing (incl. the ' EOF XXX' padding) of the
# original pipeline so keys and inputs match read_text() byte-for-byte.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xlsx', action='append', required=True,
                    help='Path to a *_text.xlsx file. Repeat for multiple splits.')
    ap.add_argument('--model', default='bert-base-uncased',
                    help='HuggingFace model id (default: bert-base-uncased, '
                         'the same weights the original LViT used).')
    ap.add_argument('--max-len', type=int, default=10,
                    help='Fixed token length per sample (default: 10, the '
                         'original QaTa-COV19 cap). Output is zero-padded/truncated '
                         'to exactly this length so a batch collates cleanly.')
    ap.add_argument('--add-special-tokens', action='store_true',
                    help='Prepend [CLS]/append [SEP]. OFF by default to mirror '
                         'the legacy mxnet bert_embedding behaviour.')
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    print(f'Loading {args.model} on {args.device} (frozen) ...')
    tokenizer = BertTokenizer.from_pretrained(args.model)
    bert = BertModel.from_pretrained(args.model).eval().to(args.device)
    for p in bert.parameters():
        p.requires_grad_(False)
    hidden = bert.config.hidden_size  # 768 for bert-base

    for xlsx in args.xlsx:
        text = read_text(xlsx)  # {image_key: padded description string}
        cache = {}
        for key, desc in text.items():
            enc = tokenizer(desc,
                            add_special_tokens=args.add_special_tokens,
                            truncation=True,
                            max_length=args.max_len,
                            return_tensors='pt').to(args.device)
            with torch.no_grad():
                hs = bert(**enc).last_hidden_state[0]  # (L, hidden)
            emb = torch.zeros(args.max_len, hidden, dtype=torch.float32)
            n = min(hs.shape[0], args.max_len)
            emb[:n] = hs[:n].float().cpu()
            cache[key] = emb.numpy()

        base = os.path.splitext(xlsx)[0]
        out = f'{base}.emb{args.max_len}.pt'
        torch.save(cache, out)
        print(f'  wrote {out}: {len(cache)} entries, shape ({args.max_len}, {hidden})')

    print('Done.')


if __name__ == '__main__':
    main()
