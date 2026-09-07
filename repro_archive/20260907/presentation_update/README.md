# BetterLViT presentation update, 7 September 2026

This source updates the user's existing 28-slide English presentation in place on an imported copy. It preserves the black/white theme, Cambria headings, Calibri body text, slide dimensions, author information and main/backup division. The source presentation and delivered presentation remain local because the supplied cover contains personal student details.

## Evidence and interpretation

- `build/evidence.json` is a compact snapshot of the formal experiment tracker, completed C4/P9/C8/P10 Test exports, recomputed C4 validation area quartiles and P11 launch/preflight/midpoint records. Per-image records are not duplicated here.
- Formal 150-epoch and exploratory 80-epoch runs are presented separately. Headline metrics are per-image Test macro Dice/IoU at threshold 0.5.
- EPPA comparisons A2/A0 and A4/A1 give observed IoU differences of +1.462 and +1.2783 percentage points. They are single-seed observations, not proof of multi-seed stability.
- C8/C4 Test IoU confidence limits include zero. P10/C8 does not establish an additional routing benefit. Auxiliary supervision is not claimed as a second innovation.
- P11 is pending as of the existing epoch-40 inspection at 19:27 Australia/Sydney. This task did not inspect the live training job or alter the scheduled final inspection.
- Source links are embedded in slide speaker notes. Source and model provenance are in the backup slides.

## Reproduction

Use the bundled Codex Presentations runtime (26.904.11930) and `@oai/artifact-tool`. The builder does not use python-pptx. Copy this directory to a private task workspace before running it. Provide the original `BetterLViTv1.pptx` as the first argument. The SHA-256 of the source used here was `7c9646f76a9918341d16f6ebff6c4b8639d1fd00905b1fdb408ddfebbc570508`.

1. Point `build/node_modules` to the bundled Node modules and set `RUNTIME_NODE_MODULES` to that path.
2. Set `PRESENTATION_SKILL_DIR` and `RUNTIME_PYTHON` to the bundled skill and Python paths if they differ from the author's machine.
3. Create an `output` directory next to `build`.
4. Run the bundled Node executable with `build/update_deck.mjs <absolute-path-to-original.pptx>`. `--draft-only` skips final publication but still renders the draft.
5. Review all slide PNGs. The finalizer writes a separate PPTX and receipt and refuses to overwrite prior outputs/receipts. Use a new private workspace for another run.

Chart literals are rounded to eight decimal places for editable Excel-backed snapshots. The unrounded experiment metrics remain in `evidence.json`. Displayed percentages use three decimals in the tables.

## Delivery and validation

Delivered filename: `BetterLViT_20260907_updated.pptx`.

- 28 slides, 13 native tables and 2 native charts with embedded data workbooks.
- Package and layout validation passed with zero findings and zero warnings.
- All slides rendered at 1280 × 720 and were visually inspected. The final file was imported and rendered again. Four connector slides were re-inspected after round-trip routing adjustments.
- Required fonts, editable chart titles and workbook/cache agreement passed structural checks.
- This does not claim a native PowerPoint application opening test.
- Final SHA-256: `80251853b8304a0694e603d493d3e0a6451fbfd6dd249230d8334e0ae8ac7a38`, 125,716 bytes.

The user's original PPTX remains unchanged. No training source or experiment configuration changed during this presentation update.
