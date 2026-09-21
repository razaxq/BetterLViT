# FSDR and RACE defence presentation

Updated 22 September 2026 from the user's latest edited reference deck. The
delivered presentation has 21 slides, English slide text and bilingual speaker
notes. It uses the descriptive title **FSDR and RACE**, with editable relationship
diagrams for placement in LViT, decoder fusion, FSDR, RACE and adaptive filtering.

## Files

- `FSDR_RACE_20260922.pptx`: delivered presentation.
- `FSDR_RACE_Defense_Speaker_Notes_20260922.md`: companion notes.
- `update.mjs`: reproducible authoring source using `@oai/artifact-tool`.
- `source_20260915.pptx`: unchanged user-edited input, retained for reproduction.
  Its historical wording is input material, not the updated presentation.
- `evidence/`: supplementary single-cycle RACE records used by backup slides.
- `manifest.json`: source, evidence and delivered-file hashes.

## Evidence boundaries

All internal ablations shown use the single-cycle cosine recipe. The primary
study has four configurations and three matched seeds, 80 epochs, frozen
CXR-BERT, no LoRA, validation macro IoU checkpoint selection, and fixed Test
probability `> 0.5`. Main values are mean ± sample SD over all seeds. The combined
model gives Test IoU `76.2262%` and Dice `84.7112%`, with `+0.7487` IoU percentage
points over the matched PLAM control. Complete RACE includes routing and its
auxiliary training. The older routing/binding controls remain explicitly limited
to seed 1219.

External comparison recipes differ. DD-CMD is `0.0732` IoU percentage points above
our mean in these evaluations. MMI-UNet's author-checkpoint training list remains
unverified. The split caveats and prior Test access remain visible in the deck.
No models were trained or re-evaluated for this presentation update.

Primary and external source records live under the repository's `docs/results/`.
The notes' original `outputs/p8_*` references map to the three evidence files
archived here. Source runtime commits and checkpoint hashes remain unchanged.

## Rebuilding

Use the Codex bundled Node runtime with `@oai/artifact-tool` **2.8.59**, the
Presentations skill **26.904.11930**, and its bundled Python runtime. The builder
imports the original deck and writes a separate candidate, validated output,
notes, and rendered slide previews. It requires the repository's result JSON.

Set `PRESENTATIONS_SKILL` to the installed skill directory and
`CODEX_DEPENDENCIES` to the bundled dependencies directory. Make its Node modules
available to this folder, for example through a local `node_modules` junction.
Run the skill's operation marker once immediately before authoring, then run
`node update.mjs` with the bundled Node executable. The script's default inputs
are relative to this directory; output defaults to ignored `.build-output/`.
For repeated builds, set a fresh `BUILD_VERSION` and output directory because
the finalizer refuses to overwrite a prior final file or validation receipt.

Optional overrides: `SOURCE_PPTX`, `EVIDENCE_ROOT`, `OUTPUT_ROOT`, `BUILD_VERSION`.
PowerPoint export metadata can vary, so regenerated ZIP hashes need not equal
the manifest's delivery hash. Review all rendered slides after editing.
