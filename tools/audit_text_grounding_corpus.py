"""Read-only Train text audit; no image pixels, masks, validation, or Test inference.

Dependencies: transformers==4.44.2, openpyxl==3.1.5. CXRBertTokenizer in the
audited snapshot is an unmodified subclass of BertTokenizer. This script uses
the local vocab and explicit options, without downloading or executing HF code.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess

from openpyxl import load_workbook
from transformers import BertTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(text):
    """Strict literal location parser. Unknown words cause abstention, not repair."""
    clauses = text.lower().strip().rstrip('.').split(',')
    if len(clauses) != 3:
        return None
    slots = {}
    for clause in clauses[-1].strip().split(' and '):
        m = re.fullmatch(r'((?:upper|middle|lower|all)(?:\s+(?:upper|middle|lower|all))*)\s+(left|right)\s+lung', clause.strip())
        if not m or m[2] in slots:
            return None
        words = set(m[1].split())
        slots[m[2]] = tuple(sorted({'upper', 'middle', 'lower'} if 'all' in words else words))
    return tuple(sorted(slots.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-folder', type=Path, required=True)
    ap.add_argument('--tokenizer-snapshot', type=Path, required=True)
    ap.add_argument('--source-repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.train_folder.name != 'Train_Folder':
        raise ValueError('This audit accepts the existing Train_Folder only.')
    workbook = args.train_folder / 'Train_Val_text.xlsx'
    # Enumerate training membership only; never load image/mask pixels.
    names = sorted(p.name for p in (args.train_folder / 'labelcol').iterdir() if p.is_file())
    wanted = set(names)
    rows = {}
    for row in load_workbook(workbook, read_only=True, data_only=True).active.values:
        if row[0] in wanted:
            if row[0] in rows:
                raise ValueError('Duplicate training workbook row')
            rows[row[0]] = str(row[1])
    if set(rows) != wanted:
        raise ValueError('Training membership does not match workbook')
    texts = [rows[n] for n in names]
    vocab = args.tokenizer_snapshot / 'vocab.txt'
    tok = BertTokenizer(vocab_file=str(vocab), do_lower_case=True, do_basic_tokenize=True)
    full = [tok.encode(t, add_special_tokens=True, truncation=False) for t in texts]
    short = tok(texts, max_length=32, padding='max_length', truncation=True)['input_ids']
    groups = defaultdict(set)
    for t, ids in zip(texts, short):
        groups[tuple(ids)].add(t)
    parsed = [binding(t) for t in texts]
    bags = defaultdict(set)
    for t, b in zip(texts, parsed):
        if b is not None:
            bags[tuple(sorted(re.findall(r'[a-z]+', t.lower())))].add(b)
    ambiguous_bags = {k for k, v in bags.items() if len(v) > 1}
    freq = Counter(texts)
    valid = [b for b in parsed if b is not None]
    asym = [b for b in valid if len(b) == 2 and dict(b)['left'] != dict(b)['right']]
    records = []
    for filename in ['Config.py', 'Load_Dataset.py', 'nets/BetterLViT.py', 'nets/LViT.py', 'nets/Vit.py', 'nets/eppa.py']:
        records.append({'path': filename, 'sha256': digest(args.source_repo / filename)})
    lengths = sorted(map(len, full))
    result = {
        'kind': 'train_text_static_audit_not_segmentation_evaluation',
        'train_count': len(names),
        'train_membership_sha256': hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest(),
        'train_name_text_sha256': hashlib.sha256(json.dumps([(n, rows[n]) for n in names], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
        'workbook_sha256': digest(workbook), 'vocab_sha256': digest(vocab),
        'tokenizer_snapshot': args.tokenizer_snapshot.name,
        'source_commit': subprocess.check_output(['git', '-C', str(args.source_repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_files': records,
        'dependencies': {p: importlib.metadata.version(p) for p in ['transformers', 'tokenizers', 'openpyxl']},
        'unique_raw_texts': len(freq),
        'unique_visible_sequences_32': len(groups),
        'raw_template_collision_groups_after_tokenization': sum(len(v) > 1 for v in groups.values()),
        'wordpiece_length_including_cls_sep': {'min': min(lengths), 'median': lengths[len(lengths)//2], 'max': max(lengths)},
        'truncated_at_32_samples': sum(len(x) > 32 for x in full),
        'truncated_at_32_templates': len({t for t, x in zip(texts, full) if len(x) > 32}),
        'samples_with_unk_tokens': sum(tok.unk_token_id in ids for ids in full),
        'strict_location_parser_coverage': len(valid),
        'strict_location_parser_abstentions': len(texts)-len(valid),
        'strict_asymmetric_bilateral_samples': len(asym),
        'strict_unilateral_samples': sum(len(b) == 1 for b in valid),
        'same_word_multiset_different_binding_groups': len(ambiguous_bags),
        'samples_in_same_word_multiset_different_binding_groups': sum(b is not None and tuple(sorted(re.findall(r'[a-z]+', t.lower()))) in ambiguous_bags for t, b in zip(texts, parsed)),
        'two_random_distinct_train_samples_identical_raw_text_probability': sum(c*(c-1) for c in freq.values())/(len(texts)*(len(texts)-1)),
        'top5_template_sample_count': sum(n for _, n in freq.most_common(5)),
        'top5_templates': [{'text': t, 'count': n} for t, n in freq.most_common(5)],
        'limitations': ['Local migration copy; hashes recorded, not a live remote dataset verification.', 'Workbook is shared Train/Val; only rows in Train membership are analyzed.', 'No mask content loaded; parser fields are language annotations, not disease absence labels.', 'Same-word-multiset groups are evidence of available binding distinctions, not evidence the trained model confuses them.', 'With zero observed truncations, raw-template collisions here arise during tokenization/normalization, not from the 32-token limit.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in {'source_files','top5_templates','limitations'}}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
