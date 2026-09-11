# Text grounding research evidence

Status: research, static Train audit, and a CPU mathematical prototype only. No new model was trained, no Test inference was performed, and the active RS3 run was not polled or modified.

- [Chinese research report and primary sources](REPORT.md)
- [Train corpus audit](train_text_audit.json)
- [Synthetic numerical checks and counterexamples](mass_projection_checks.json)
- [Source access/novelty notes](source_review.json)

The main-path code audit uses frozen source `f79842331e4e5e41526ed66da31ec174ea4a61b2`; file hashes are in the corpus audit. The audit corpus is the local migration copy and is not claimed to have been freshly verified against the running remote server. `Train_Val_text.xlsx` is filtered using only `Train_Folder/labelcol` filenames. Mask/image pixels and Val/Test membership are not used by the script.

## Reproduce

Validated with Python 3.12.14, transformers 4.44.2, tokenizers 0.19.1, openpyxl 3.1.5 and numpy 2.5.3. Use an isolated environment. Supply existing local dataset/tokenizer paths; the script downloads no model and runs no remote model code.

```powershell
python -m pip install transformers==4.44.2 tokenizers==0.19.1 openpyxl==3.1.5 numpy==2.5.3
python tools/audit_text_grounding_corpus.py --train-folder '<dataset>/Train_Folder' --tokenizer-snapshot '<local-CXR-BERT-snapshot>' --source-repo '<frozen-source-checkout>' --output repro_archive/20260911/text_grounding_research/train_text_audit.json
python tools/check_text_mass_redistribution.py --output repro_archive/20260911/text_grounding_research/mass_projection_checks.json
```

Expected corpus: 5716 records, 273 raw templates, 240 tokenized sequences, max WordPiece length including CLS/SEP 24, zero truncations. Source/membership/workbook/vocab hashes must also match before calling it a reproduction. Strict parser abstention is intentional; no spelling/clinical label repair is performed.

Expected numerical check: finite-difference error below 1e-7, identity/common-mode errors below 1e-12, mass error below 1e-10 in the tested float64 case. These checks do not establish learned text localization, CUDA/autograd integration, precision preservation, or real-data IoU improvement.

The suggested T1–T4 labels in the report are research-arm names, not launched experiments or frozen manifests. Any later formal training requires its own complete source SHA, experiment tag, runtime provenance and registration.
