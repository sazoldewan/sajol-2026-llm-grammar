# Known Reproducibility Limitations

This file expands on the summary in the top-level `README.md`.

## 1. Unpinned model revisions — RESOLVED for 7 of 9, ambiguous for 2

Every generation script loads its checkpoint by Hugging Face model ID only
(`AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)`), with no `revision=`
argument pinning a specific commit hash. If any of these repositories are
updated upstream (re-uploaded weights, safetensors conversion, config
changes), re-running generation against the same model ID may not reproduce
the original outputs byte-for-byte, even with identical code and decoding
parameters.

This has since been checked against the actual Hugging Face cache on the HPC
cluster (`$HF_HOME/hub/models--.../snapshots/`, `HF_HOME=$WORK/.cache/huggingface`).
Each `snapshots/` folder is named with the exact commit hash of what was
downloaded. Result:

- **7 of 9 checkpoints had exactly one cached snapshot** — a clean, confirmed
  revision hash for each. These are recorded in the README's checkpoint
  table: Mistral base/SFT/DPO, Llama base/SFT/DPO, and OLMo-2 base.
- **`allenai/OLMo-2-1124-7B-SFT` and `allenai/OLMo-2-1124-7B-DPO` each have
  two cached snapshots**:
  - OLMo-2-1124-7B-SFT: `1de02c0175118a9de5854aec80a1f970e701e928` and
    `606fc5cf326aac28795e7d019948b57ac9434ca2`
  - OLMo-2-1124-7B-DPO: `d6fa9c9f6f5918d0bf2a162261e0620290c9d4ed` and
    `e34ea60adff2e575f4fe7569eaffd1b28509b6fd`

  Two cached snapshots means that repository was fetched at two different
  points in time (e.g. the model was re-downloaded after an upstream update,
  or a second job re-triggered a fresh pull). The cache alone cannot say
  which of the two was in effect during the specific runs that produced this
  release's ACL OLMo-2 SFT/DPO data — both are physically present, and
  `from_pretrained` would only have used one per run. Two ways to narrow
  this down, if it matters for your use of this release:
  1. Check each snapshot folder's modification time (`stat` or
     `ls -la --time=ctime` on both `snapshots/<hash>/` directories) and
     compare against the timestamps in the OLMo-2 SFT/DPO SLURM job logs
     (the `TaskPrologue` line at the top of each `.log` file gives the job's
     start time) — whichever snapshot existed *before* the relevant job ran
     is the candidate; if a snapshot's mtime is *after* every relevant job's
     start time, it can be ruled out.
  2. If the two snapshots are actually identical content under different
     commit hashes (e.g. a no-op re-tag upstream), the ambiguity is harmless
     — this can be checked by diffing the two snapshot directories' file
     hashes.

  Absent that, treat the OLMo-2 SFT/DPO revision as unresolved: two specific
  candidate commits are known, but not which one was used.

## 2. Generation-environment package versions — RESOLVED (verified)

Originally, the HPC SLURM environment (`module load python/3.12-base`,
virtualenv `thesis_env`) that ran all generation jobs had no package versions
captured or exported anywhere in the project. This has since been resolved:
the thesis author ran `python --version` and
`pip freeze | grep -E "torch|transformers|accelerate|datasets|huggingface-hub"`
directly inside `thesis_env` on the HPC cluster and provided the output.
`requirements/generation.txt` now pins the verified real values:

```
Python 3.12.0
torch==2.12.0
transformers==5.11.0
accelerate==1.14.0
datasets==5.0.1
huggingface_hub==1.19.0
```

`accelerate` was not in this release's earlier package list at all — it
isn't imported directly by any generation script, but is required by
`transformers` at runtime for `device_map="auto"`, which every generation
script uses. It's now included and pinned.

Decoding behavior for greedy generation is normally stable across
`transformers` versions, but tokenizer defaults, chat-template handling, and
model-specific generation-config defaults have changed across versions
historically. That's no longer a concern here since the exact version is now
known and pinned — this item is kept only as a record of what was verified
and when, not as an open limitation.

## 3. Headline-ID assignment script not recovered (mapping itself is verified)

`data/human_corpora_manifests/headlines_source_manifest.csv` and every
downstream headline file (features, master, results) key on `prompt_id`
values `HL_0001`–`HL_1000`. Tracing back through every script in the original
project (`grep -rl "HL_\|article_id"` across the full codebase) turns up no
script that creates this column — `scripts/corpus_construction/
download_news_articles.py` writes its output with `index=False` and no ID
column at all. The `article_id`/`HL_####` values must have been added in an
ad hoc, undocumented step (manual rename + sequential numbering) before the
file was uploaded to the HPC cluster as `$WORK/data/news_articles_1000.csv`,
which the generation script then reads assuming a `prompt_id` column already
exists.

The README gives a one-line reconstruction (`HL_{i+1:04d}` over the
downloader's row order) for how this numbering was most likely produced. The
*script* that did it is still not recovered — no code anywhere in the
project performs this assignment — but the *numbering itself* has since been
verified, not just inferred: `gen_llama_base`'s SLURM job log (still held by
the thesis author) shows

```
Loading data from /home/atuin/<hpc_account>/<hpc_user>/data/news_articles_1000.csv
[HL_0001] Human:     France's Sarkozy reveals his 'Passions' but insists no come-back on cards
```

which is an exact match, including punctuation, to row 1
(`article_id=HL_0001`) of `Datasets/news_articles_1000.csv`, the original
local corpus file (not included in this release, since it contains article text).
This confirms the file uploaded to the HPC as `$WORK/data/news_articles_1000.csv`
was in the same row order as the local copy here, with `HL_0001`–`HL_1000`
already assigned exactly as this release's manifest has them. So: the
*command that created the ID column* is still unknown, but the *resulting
`prompt_id` mapping* is confirmed correct against an actual generation run,
not just plausible.

By contrast, the ACL corpus's `prompt_id` assignment (`ACL_0001`–`ACL_1000`)
is fully scripted and traceable: `download_acl_corpus.py` (originally
`download_acl_corpus_v2.py`) assigns it explicitly, in final post-filter row
order, in the same script that does the filtering — no separate step, no gap.

## 4. Hardcoded absolute paths in every script

All scripts in this release still contain their original hardcoded path
constants:
- Local paths of the form `/Users/dewansajol/Study/Uni Bamberg/Thesis/...`
  (corpus construction, feature extraction, statistics scripts).
- HPC paths of the form `$WORK/data/...`, `$WORK/outputs/...` (generation
  scripts, meant to run under SLURM with `WORK` exported).

None of these were genericized for this release — running any script here
requires first editing its path constants (near the top of each file) to
point at wherever you've placed this repository's data. This was a deliberate
choice to keep the released scripts byte-identical to what was actually run
for the thesis, rather than risk silently changing behavior while "cleaning
up" paths.

## 5. Headline sample is a small, stream-order-dependent slice of the source corpus

The 1,000 sampled headlines come from only 7 of the 27 publications in *All
the News 2.0* (Business Insider, Hyperallergic, Reuters, TMZ, Vice, Vice
News, Vox), because the downloader streams the dataset and stops the moment
1,000 filter-passing articles have been collected — it does not sample
across the full publication set. This is not a bug, but it means:
- The headline sample is **not** representative of the full 27-publication,
  2.7-million-article corpus's stylistic range.
- If the upstream dataset (`rjac/all-the-news-2-1-Component-one`) is ever
  reordered, re-partitioned, or re-uploaded, re-running the same filters will
  likely not reproduce the same 1,000 articles, even with the same filter
  thresholds.

## 6. Non-deterministic GPU kernels not addressed

Generation used greedy decoding (`do_sample=False`), which removes sampling
randomness, but no explicit determinism flags (e.g. `torch.use_deterministic_algorithms`,
fixed cuDNN settings) were set in the generation scripts. Minor floating-point
non-determinism from GPU kernel scheduling is possible in principle, though
unlikely to change greedy argmax decisions in practice for most tokens.
