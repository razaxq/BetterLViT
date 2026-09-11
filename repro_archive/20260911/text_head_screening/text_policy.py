"""Frozen Train-derived language controls; no pixel labels or test data."""
from collections import defaultdict
import hashlib
import re

VARIANTS = ('canonical', 'semantic_match', 'relation_swap', 'drop_location', 'generic')
SCOPES = ('main', 'eppa', 'both')
COUNTS = ('one','two','three','four','five','six','seven','eight','nine','ten')
LEVELS = ('upper','middle','lower')


def parse(text):
    clean = ' '.join(text.lower().strip().split()).rstrip('.')
    m = re.fullmatch(r'(bilateral|unilateral) pulmonary infection, ('+'|'.join(COUNTS)+r') infected areas?, (.+)', clean)
    if not m:
        return None
    slots = {}
    for phrase in m[3].split(' and '):
        match = re.fullmatch(r'((?:upper|middle|lower|all)(?: (?:upper|middle|lower|all))*) (left|right) lung', phrase)
        if not match or match[2] in slots:
            return None
        words = match[1].split()
        if len(words) != len(set(words)) or ('all' in words and len(words) != 1):
            return None
        slots[match[2]] = tuple(x for x in LEVELS if x in words or 'all' in words)
    if (m[1]=='unilateral') != (len(slots)==1):
        return None
    return (m[1], m[2], tuple(sorted(slots.items())))


def render(key, include_location=True):
    scope, count, slots = key
    prefix = scope.capitalize()+' pulmonary infection, '+count+' infected '+('area' if count=='one' else 'areas')
    if not include_location:
        return prefix+'.'
    return prefix+', '+' and '.join(' '.join(levels)+' '+side+' lung' for side,levels in slots)+'.'


def group(key):
    if key is None:return 'unparsed'
    slots = dict(key[2])
    if len(slots)==1:return 'unilateral'
    return 'symmetric_bilateral' if slots['left']==slots['right'] else 'asymmetric_bilateral'


class TextPolicy:
    def __init__(self, train_texts):
        self.pool = defaultdict(set)
        for text in train_texts:
            key = parse(text)
            if key is not None:self.pool[key].add(text)

    def variants(self, name, text):
        key = parse(text)
        out = dict(canonical=text, semantic_match=text, relation_swap=text,
                   drop_location=text, generic='Pulmonary infection.')
        if key is None:return out
        out['canonical'] = render(key)
        out['drop_location'] = render(key, include_location=False)
        alternatives = sorted(x for x in self.pool.get(key,()) if x != text)
        if alternatives:
            index = int(hashlib.sha256(('text-a-v1:'+name).encode()).hexdigest(),16)%len(alternatives)
            out['semantic_match'] = alternatives[index]
        if group(key) != 'symmetric_bilateral':
            swap = tuple(sorted(('right' if side=='left' else 'left', levels) for side,levels in key[2]))
            out['relation_swap'] = render((key[0],key[1],swap))
            assert parse(out['relation_swap']) != key
        assert parse(out['canonical']) == key and parse(out['semantic_match']) == key
        return out
