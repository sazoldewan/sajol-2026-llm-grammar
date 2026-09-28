"""
Add prompt_style column to the 9 ACL generation output CSVs.

Design decision: base checkpoints were generated with a completion-style
prompt (raw text only); SFT and DPO checkpoints were generated with an
instruction-style prompt (explicit register/style instruction). This is a
direct confound for grammatical comparison against base, so H1/H2 are
tested on the headline genre only (uniform completion-style prompt across
all 9 headline checkpoints); the ACL genre contributes to H3 (genre
moderation) via the SFT vs DPO comparison only. ACL base outputs are
retained in the dataset, labelled prompt_style = 'completion', and
reported descriptively rather than included in H1/H2 testing.

  stage == 'base'        -> prompt_style = 'completion'
  stage in {'sft','dpo'} -> prompt_style = 'instruction'

Files are modified in place (9 files in outputs/acl/*_acl.csv).
"""

import glob
import os

import pandas as pd

DATA_DIR = "/Users/dewansajol/Study/Uni Bamberg/Thesis/outputs/acl"


def prompt_style_for_stage(stage):
    return "completion" if stage == "base" else "instruction"


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_acl.csv")))
    print(f"Found {len(files)} files in {DATA_DIR}")

    for f in files:
        name = os.path.basename(f)
        df = pd.read_csv(f)
        df["prompt_style"] = df["stage"].apply(prompt_style_for_stage)
        df.to_csv(f, index=False)
        print(f"{name:24s} stage={df['stage'].iloc[0]:12s} "
              f"prompt_style={df['prompt_style'].iloc[0]}")


if __name__ == "__main__":
    main()
