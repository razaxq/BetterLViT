"""Train-only, label-blind grouping and fixed relation eligibility."""
from collections import Counter
import hashlib
import re
from text_policy import TextPolicy, parse, group


def records(mask_names, texts):
    policy = TextPolicy(texts[n] for n in mask_names)
    frequency = Counter(texts[n] for n in mask_names)
    rows = []
    for index, mask in enumerate(mask_names):
        name = mask.replace('mask_', '')
        match = re.search(r'sub-S\d+', name)
        patient = match.group(0) if match else name
        holdout = int(hashlib.sha256(('text-b-v1:'+patient).encode()).hexdigest(),16)%5 == 0
        text = texts[mask]; variants = policy.variants(mask,text)
        kind = group(parse(text)); eligible = kind in ('unilateral','asymmetric_bilateral')
        rows.append(dict(index=index,name=name,mask_name=mask,group_id=patient,
            explicit_patient_id=match is not None,partition='holdout' if holdout else 'fit',
            eligible=eligible,semantic_group=kind,source_text=text,canonical=variants['canonical'],
            reference=variants['relation_swap'] if eligible else variants['canonical'],
            template_frequency=frequency[text]))
    fit = {r['group_id'] for r in rows if r['partition']=='fit'}
    hold = {r['group_id'] for r in rows if r['partition']=='holdout'}
    assert not fit & hold
    return rows


def counts(rows):
    return dict(samples=len(rows),partitions=dict(Counter(r['partition'] for r in rows)),
        eligible_partitions=dict(Counter(r['partition'] for r in rows if r['eligible'])),
        explicit_patient_id_samples=sum(r['explicit_patient_id'] for r in rows),
        groups=len(set(r['group_id'] for r in rows)),
        semantic_groups=dict(Counter(r['semantic_group'] for r in rows)))
