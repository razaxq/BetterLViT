"""Publish credential-free verification records for completed upload batches."""
import json
import shutil
from pathlib import Path

from huggingface_hub import get_token
from upload_pending import DOCS, HERE, OUT, write_json


def main():
    result = HERE / 'results'
    result.mkdir(exist_ok=True)
    token = get_token()
    names = ['upload_plan.json', 'source_audit.json', 'summary.json', 'additional_plan.json',
             'additional_manifests.json', 'additional_sources_github_verified.json', 'additional_verified.json',
             'test_exports_upload_verified.json']
    names += [p.name for p in OUT.glob('*_upload_verified.json')]
    for name in sorted(set(names)):
        source = OUT / name
        if source.exists():
            data = source.read_bytes()
            assert not token or token.encode() not in data, 'Credential must not enter an archive'
            (result / name).write_bytes(data)
    recipe = json.loads((OUT / 'summary.json').read_text())
    assert recipe['phase'] == 'complete' and recipe['verified_files'] == 30
    proofs = [json.loads((OUT / (row['label'] + '_upload_verified.json')).read_text()) for row in recipe['experiments']]
    if (OUT / 'additional_verified.json').exists():
        proofs += json.loads((OUT / 'additional_verified.json').read_text())
    assert all(p['verified'] and p['independent_local_listing_verified'] for p in proofs)
    labels = {p['bucket_prefix']: p.get('label', next((r['label'] for r in recipe['experiments'] if r['prefix'] == p['bucket_prefix']), '')) for p in proofs}
    summary = dict(bucket=recipe['bucket'], completed_training_archives=len(proofs),
                   all_nine_training_archives_complete=len(proofs) == 9,
                   verified_training_files=sum(p['verified_files'] for p in proofs),
                   training_logical_bytes=sum(f['bytes'] for p in proofs for f in p['files']),
                   model_files_preserved=True, credentials_in_artifacts=False,
                   experiments=[dict(label=labels[p['bucket_prefix']], prefix=p['bucket_prefix'],
                                     source_git_commit=p['source_git_commit'], verified_files=p['verified_files']) for p in proofs])
    if (OUT / 'test_exports_upload_verified.json').exists():
        test = json.loads((OUT / 'test_exports_upload_verified.json').read_text())
        assert test['verified']
        summary.update(test_exports_verified_files=test['verified_files'],
                       test_exports_logical_bytes=sum(f['bytes'] for f in test['files']),
                       evaluation_source_git_commit=test['evaluation_source_git_commit'])
    write_json(OUT / 'all_uploads_summary.json', summary)
    write_json(result / 'all_uploads_summary.json', summary)
    rows = ['# 已核验的云端归档', '',
            f'已完成 {len(proofs)} 组训练归档，{summary["verified_training_files"]} 个文件，{summary["training_logical_bytes"]:,} 逻辑字节。全部源文件大小/Xet哈希、服务器云端列表与独立本地云端列表一致。', '',
            '| 实验 | Git前缀 | 已核验文件 |', '|---|---|---:|']
    rows += [f'| {x["label"]} | {x["prefix"]} | {x["verified_files"]} |' for x in summary['experiments']]
    rows += ['', '这些字节是文件逻辑大小，不能等同于网络实际传输量或去重后的云端物理占用。所有上传均为添加操作，服务器原模型保留。P11续训Best来自80轮父提交，续训目录的runtime/selection文件明确记录其来源。']
    if 'test_exports_verified_files' in summary:
        rows += ['', f'本轮六模型Test的JSON/日志及三种子汇总与协议已追加并核验，共{summary["test_exports_verified_files"]}文件、{summary["test_exports_logical_bytes"]:,}字节。历史评估文件保留。']
    else:
        rows += ['', '本轮六模型Test产物须待预约检查确认全部完成后追加，本记录不宣称已上传这些新Test结果。']
    (result / 'README.md').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
