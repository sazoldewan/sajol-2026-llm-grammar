# SLURM Job Scripts

The exact SLURM job scripts used to run generation on the HPC cluster, one
per checkpoint (18 total: 9 headline + 3 ACL base + 6 ACL SFT/DPO).

`<hpc_account>` and `<hpc_user>` are placeholders standing in for the
original HPC account/username in every path (`/home/hpc/<hpc_account>/<hpc_user>/...`,
`/home/atuin/<hpc_account>/<hpc_user>`) — replace both with your own before
submitting any of these with `sbatch`. Nothing else needs editing to resubmit
a job as-is, other than having `~/thesis_env` (see `requirements/generation.txt`)
and `~/thesis/scripts/*.py` (see `../generation/`) in place at those paths.

**Script names.** The job scripts call the generation scripts by the names
they had on the cluster. In this release they were renamed:

| Name called in job scripts | Name in `scripts/generation/` |
|---|---|
| `generate.py` | `generate_headlines.py` |
| `generate_acl.py` | `generate_acl_base.py` |
| `generate_acl_instruction.py` | `generate_acl_sft_dpo.py` |

Either copy the release scripts to `~/thesis/scripts/` under their original
names, or edit the `python ~/thesis/scripts/...` line in each job script.

- `headlines/` — all 9 headline checkpoints, calling `generate.py`.
- `acl_base/` — the 3 ACL base checkpoints, calling `generate_acl.py`.
- `acl_sft_dpo/` — the 6 ACL SFT/DPO checkpoints, calling
  `generate_acl_instruction.py`. These were re-fetched directly from the HPC
  cluster during release preparation — a set of similarly-named job scripts
  had been left in the original project's `scripts/Acl/jobs/` calling the
  wrong script (`generate_acl.py`, the base/completion-style script) for
  these 6 checkpoints; those were never uploaded here. See
  `docs/MANIFEST.md` for the full story.

All 18 use `--gres=gpu:a40:1`; headline jobs request `--time=02:00:00`, ACL
jobs (base and SFT/DPO alike) request `--time=04:00:00`. `HF_TOKEN` is always
read from `~/.cache/huggingface/token`, never hardcoded.
