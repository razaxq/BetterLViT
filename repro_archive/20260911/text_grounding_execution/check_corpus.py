"""Validate the fixed intervention policy using training text only, without torch."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from openpyxl import load_workbook
from transformers import BertTokenizer
from analysis import digest,write_json
from text_policy import TextPolicy,VARIANTS,parse,group


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-folder',type=Path,required=True)
    p.add_argument('--tokenizer-snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((Path(__file__).parent/'manifest.json').read_text())
    assert a.train_folder.name=='Train_Folder'
    names=sorted(p.name for p in (a.train_folder/'labelcol').iterdir() if p.is_file())
    workbook=a.train_folder/'Train_Val_text.xlsx'
    assert digest(workbook)==manifest['train_workbook_sha256']
    membership=hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest()
    assert membership==manifest['train_membership_sha256']
    wanted=set(names);texts={}
    for row in load_workbook(workbook,read_only=True,data_only=True).active.values:
        if row[0] in wanted:
            assert row[0] not in texts
            texts[row[0]]=str(row[1])
    assert set(texts)==wanted and len(texts)==5716
    vocab=a.tokenizer_snapshot/'vocab.txt'
    assert digest(vocab)=='e025f479c3a7ce6e248971bfe1b409708db0439e9488d30e254f81f27f75d678'
    tok=BertTokenizer(vocab_file=str(vocab),do_lower_case=True,do_basic_tokenize=True)
    policy=TextPolicy(texts.values());counts=Counter();lengths=Counter();records=[]
    for name in names:
        text=texts[name];key=parse(text);g=group(key);counts['group/'+g]+=1
        original=tok.encode(text,truncation=False);variants=policy.variants(name,text)
        for variant in VARIANTS:
            value=variants[variant];ids=tok.encode(value,truncation=False)
            assert len(ids)<=32 and tok.unk_token_id not in ids,(name,variant)
            lengths[variant]=max(lengths[variant],len(ids))
            counts['tokens_changed/'+variant]+=ids!=original
            counts['raw_changed/'+variant]+=value!=text
            if key is not None and variant in ('canonical','semantic_match'):assert parse(value)==key
            if key is not None and variant=='relation_swap':
                assert (parse(value)!=key)==(g in ('unilateral','asymmetric_bilateral'))
        records.append((name,variants))
    result=dict(kind='train_only_intervention_policy_check',passed=True,samples=len(names),
        train_membership_sha256=membership,workbook_sha256=digest(workbook),vocab_sha256=digest(vocab),
        policy_sha256=digest(Path(__file__).parent/'text_policy.py'),semantic_train_groups=len(policy.pool),
        counts=dict(counts),maximum_wordpiece_lengths=dict(lengths),no_truncation=True,no_unknown_tokens=True,
        generated_variants_sha256=hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        mask_pixels_read=False,validation_images_read=False,test_split_accessed=False,
        note='Strict complete-sentence parser; abstentions may exceed the earlier location-only corpus audit.')
    write_json(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
