# File Manifest

Every file in this release, its original location in the source project, and
what it's for. "Transform" notes any change made before release (redaction,
rename); everything else is copied unmodified.

## scripts/corpus_construction/

| Release file | Original path | Transform | Purpose |
|---|---|---|---|
| `download_news_articles.py` | `scripts/headlines/download_news_articles.py` | none (byte-identical to `code/download_news_articles.py`) | Streams and filters the 1,000-headline sample from *All the News 2.0* |
| `download_acl_corpus.py` | `scripts/Acl/download_acl_corpus_v2.py` | renamed (dropped `_v2` suffix) | Streams, filters, and assigns `prompt_id` for the 1,000-introduction ACL sample. **This is the version that actually produced the released data** — `code/download_acl_corpus_v2.py` is an earlier, stale copy with different filter thresholds (2010–2019 vs. the actual 2015–2020) and was excluded |
| `build_master_headlines.py` | `scripts/headlines/build_master.py` | renamed | Combines the 9 per-checkpoint headline generation outputs + human reference into one master table |
| `build_master_acl.py` | `scripts/Acl/build_master_acl.py` | none | Same, for ACL |

**Excluded as superseded/abandoned:** `code/download_acl_corpus.py` (v1 draft,
superseded by v2), `code/download_acl_intro_final.py` (an earlier-named
prototype, also carrying the stale 2010–2019 filter), `code/filter_acl_intro.py`
(a separate filter step from an earlier two-stage pipeline design, made
redundant once `download_acl_corpus_v2.py` began filtering and assigning
`prompt_id` in one pass).

## scripts/generation/

| Release file | Original path | Transform | Purpose |
|---|---|---|---|
| `generate_headlines.py` | `scripts/headlines/generate.py` | renamed | Production headline generation, all 9 checkpoints, argparse-driven |
| `generate_acl_base.py` | `scripts/Acl/generate_acl.py` | renamed | ACL generation for the 3 base checkpoints (completion-style prompt) |
| `generate_acl_sft_dpo.py` | `code/generate_acl_instruction.py` | renamed | ACL generation for the 6 SFT/DPO checkpoints (instruction-style prompt) |

**Excluded:** `code/generate_headlines.py` (an earlier pilot script with
few-shot examples and sampling enabled — not what produced the final,
greedy-decoded, zero-shot dataset).

## scripts/slurm_jobs/

The actual SLURM job scripts submitted on the HPC cluster, 18 total (9
headline + 3 ACL base + 6 ACL SFT/DPO), one per checkpoint. Every path
containing the original HPC account/username was replaced with the
placeholders `<hpc_account>`/`<hpc_user>` before inclusion — see
`scripts/slurm_jobs/README.md`.

- `headlines/` (9 files) and `acl_base/` (3 files) were copied from
  `scripts/headlines/jobs/*.sh` and
  `scripts/Acl/jobs/gen_acl_{mistral,llama,olmo2}_base.sh` respectively —
  verified accurate (each calls the correct generation script with the
  correct arguments).
- `acl_sft_dpo/` (6 files) — **not** copied from
  `scripts/Acl/jobs/gen_acl_*_{sft,dpo}.sh` in the original project, because
  those 6 files are stale: they call `generate_acl.py` (the base/
  completion-style script), which was already established (via a
  directly-pasted, genuinely-run job script and the thesis author's
  confirmation) to be wrong — the real SFT/DPO runs called
  `generate_acl_instruction.py`. Instead, these 6 files were re-fetched
  directly from the HPC cluster during release preparation
  (`~/thesis/scripts/jobs/acl_instruction_jobs/` there), confirmed to call
  `generate_acl_instruction.py` with the correct `--model_id`/`--family`/
  `--stage` for each of the 6 checkpoints, and included as-is (path-scrubbed
  only). The stale originals in `scripts/Acl/jobs/` were left untouched in
  the source project — this release simply doesn't draw from them.

## scripts/postprocessing/

| Release file | Original path | Transform | Purpose |
|---|---|---|---|
| `headline_quality_check.py` | `code/headline_quality_check.py` | none | Diagnostic only (alignment, word-count, truncation-artifact checks) — makes no changes to the data; headline post-processing needed zero exclusions |
| `acl_quality_check.py` | `code/acl_quality_check.py` (byte-identical to `scripts/Acl/acl_quality_check.py`) | none | Diagnostic only — row counts, null counts, repetition rate, usability rate per checkpoint |
| `clean_acl_outputs.py` | `scripts/Acl/clean_acl_outputs.py` | none | Actual data-modifying step: repeat-onset truncation + sentence-boundary trimming on ACL generation outputs |
| `add_prompt_style.py` | `scripts/Acl/add_prompt_style.py` | none | Labels each ACL row's `prompt_style` (`completion` for base, `instruction` for SFT/DPO) |

## scripts/feature_extraction/

| Release file | Original path | Transform | Purpose |
|---|---|---|---|
| `extract_features_headlines.py` | `scripts/headlines/extract_features.py` (byte-identical to `code/extract_features.py`) | none | Stanza/UD-based extraction of the headline grammatical feature set |
| `extract_features_acl.py` | `scripts/Acl/extract_features_acl.py` (byte-identical to `code/extract_features_acl.py`) | none | Same, for ACL (adds clause-linked passive-voice detection) |

## scripts/statistics/

| Release file | Original path | Transform | Purpose |
|---|---|---|---|
| `stat_analysis_headlines.py` | `code/stat_analysis.py` | renamed + edited | Computes the H1/H2 statistics for headlines. Release-copy edits: (1) `run_pca()` (standalone PCA scatter analysis), its `main()` call, and the `sklearn.decomposition.PCA` import were deleted; (2) `run_factor_analysis()` (Bartlett, KMO, PAF, varimax, factor scores, correlation matrix, and all their CSV/figure outputs) was replaced by `get_boxplot_candidate_features()`, which keeps only the zero-variance feature exclusion that `plot_selected_feature_boxplots()` depends on; (3) the module docstring and one comment were updated to match. **Verified:** the original script and this release version were both run on `data/features/` in the same clean environment, and all 6 CSV outputs were byte-identical. `scripts/headlines/stat_analysis.py` is a preserved pre-diagnostics rollback copy (different `RESULTS_DIR`, missing 5 diagnostic outputs) and was excluded |
| `stat_analysis_acl.py` | `code/stat_analysis_acl.py` | renamed + edited | Same treatment as above (H2 only for ACL; H1 restricted to SFT-vs-DPO). **Verified** the same way: all 6 CSV outputs byte-identical to the original script's output. `scripts/Acl/stat_analysis_acl.py` is the equivalent excluded rollback copy |
| `generate_h3_comparison.py` | `code/generate_h3_comparison.py` | none | Combines both genres' per-family statistics into the H3 cross-genre comparison figures/tables |

## data/human_corpora_manifests/

| Release file | Derived from | Transform |
|---|---|---|
| `headlines_source_manifest.csv` | `Datasets/news_articles_1000.csv` | Kept only `prompt_id` (renamed from `article_id`), `publication`, `date`, `year`, `url`, `word_count`. **Dropped: `title` (human headline text), `article` (full article body), `author`** |
| `acl_source_manifest.csv` | `data/acl_intro_final.csv` | Kept only `prompt_id`, `anthology_id`, `year`, `venue`, `word_count`. **Dropped: `introduction` (full source text)** |

## data/generated_outputs/

| Release file | Derived from | Transform |
|---|---|---|
| `master_headlines_redacted.csv` | `data/master_headlines.csv` | All 10,000 rows kept; `text` field set to null for the 1,000 rows where `source == "human"`. All 9,000 generated-text rows are unchanged |
| `master_acl_redacted.csv` | `data/master_acl.csv` | Same treatment; 1,000 human rows redacted, 9,000 generated rows unchanged |

Per-checkpoint raw generation output files (`outputs/headlines/*.csv`,
`outputs/acl/*.csv` in the original project — 9 files per genre, each mixing
generated text with `human_headline`/`human_continuation`/`input_text`
source-derived columns) were **not** individually re-included: the redacted
master files above are their consolidated, already-redacted equivalent, and
including both would only duplicate the same redaction work nine times over
per genre for no added information.

## data/features/

| Release file | Original path | Transform |
|---|---|---|
| `features_headlines.csv` | `data/features_headlines.csv` | none — verified to contain no raw-text columns |
| `features_acl.csv` | `data/features_acl.csv` | none — verified to contain no raw-text columns |

## data/results/{headlines,acl,cross_genre}/

Originally, every `.csv` and `.png` from
`results/result final/{headlines,acl,cross_genre}/` in the original project
was copied unmodified (22 files each for headlines/acl, 4 for cross_genre).

**EFA and PCA were subsequently removed from the final thesis**, so the 10
EFA/PCA-related files per genre (factor loadings, factor scores, eigenvalues,
EFA diagnostics, scree plots, PCA scatter data — both the `.csv` and `.png`
forms) were removed from this release. `correlation_matrix.csv` and
`correlation_heatmap.png` were removed as a follow-up for the same reason —
that correlation matrix was computed purely as an EFA factorability
diagnostic (feeding Bartlett's test, KMO, and PAF) and is not read by any
H1/H2/H3 computation. 10 files per genre remain in `headlines/` and `acl/`
(4 in `cross_genre/`, unaffected): Cohen's d matrices, Kruskal-Wallis and
Mann-Whitney tables, and the H3 cross-genre comparison — none contain any
source or generated text.

## requirements/

`analysis.txt` and `generation.txt` were newly written for this release (not
copied from the original project, which had no requirements/lockfile at all). `scikit-learn` was removed from `analysis.txt`
after the EFA/PCA removal, since no remaining script imports it.
`generation.txt`'s versions were originally unverified guesses but are now
pinned to the real values retrieved from the HPC `thesis_env` venv via
`pip freeze` — see `docs/LIMITATIONS.md` item 2.

## LICENSE

`LICENSE` (MIT) was newly added for this release. It covers the code only;
the source corpora and model checkpoints remain under their own terms.
