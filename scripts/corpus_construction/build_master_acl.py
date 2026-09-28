"""
Build master ACL dataset (long format).

Input:
  - 9 generated output CSVs in outputs/acl/*_acl.csv
  - ACL corpus: data/acl_intro_final.csv (for year, venue)

Human reference:
  human_continuation is derived by splitting each paper's introduction at
  token 200 using each model family's own tokenizer, so it is NOT identical
  across all 9 files — only within a family (confirmed: identical within
  llama and olmo2, NOT identical within mistral due to minor decode
  differences; substantially different across families since tokenizers
  differ). A single canonical human reference is used for all 1,000 papers
  rather than one per family, to avoid the human baseline itself shifting
  across model-family comparisons. OLMo-2 was chosen as the canonical
  family (empirically the "middle ground" tokenizer granularity between
  Llama and Mistral, based on continuation-token-count distributions).

Design decision (see project notes): base ACL checkpoints were generated
with a completion-style prompt while SFT/DPO checkpoints were generated
with an instruction-style prompt (prompt_style column, added upstream by
add_prompt_style.py). This is a direct confound for grammatical
comparison against base, so H1/H2 are tested on the headline genre only;
the ACL genre contributes to H3 (genre moderation) via the SFT vs DPO
comparison only. ACL base outputs are retained here and labelled
prompt_style = 'completion' for descriptive reporting, not for H1/H2.

Output: data/master_acl.csv
  Schema: prompt_id, genre, source, model, stage, checkpoint,
          prompt_style, text, is_usable, year, venue
"""

import pandas as pd
import glob
import os

OUTPUT_DIR = '/Users/dewansajol/Study/Uni Bamberg/Thesis/outputs/acl'
DATA_DIR = '/Users/dewansajol/Study/Uni Bamberg/Thesis/data'
CANONICAL_HUMAN_FAMILY = 'olmo2'
CANONICAL_HUMAN_TOKENIZER = 'allenai/OLMo-2-1124-7B'

# ── Load all 9 generated output CSVs ────────────────────────────────────────
files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*_acl.csv')))
model_dfs = {os.path.basename(f).replace('_acl.csv', ''): pd.read_csv(f) for f in files}
print(f"Loaded {len(model_dfs)} files: {sorted(model_dfs.keys())}")

generated_all = pd.concat(model_dfs.values(), ignore_index=True)

# ── Build generated rows ─────────────────────────────────────────────────────
generated_rows = pd.DataFrame({
    'prompt_id':  generated_all['prompt_id'],
    'genre':      'acl',
    'source':     'generated',
    'model':      generated_all['model'],
    'stage':      generated_all['stage'],
    'checkpoint': generated_all['checkpoint'],
    'prompt_style': generated_all['prompt_style'],
    'text':       generated_all['generated_text_clean'],
    'is_usable':  generated_all['is_usable'],
})

# ── Build human rows from canonical OLMo-2 family ───────────────────────────
# Verify identical within olmo2 family before trusting it as canonical
olmo2_base = model_dfs[f'{CANONICAL_HUMAN_FAMILY}_base'].set_index('prompt_id')['human_continuation']
olmo2_sft = model_dfs[f'{CANONICAL_HUMAN_FAMILY}_sft'].set_index('prompt_id')['human_continuation']
olmo2_dpo = model_dfs[f'{CANONICAL_HUMAN_FAMILY}_dpo'].set_index('prompt_id')['human_continuation']
assert olmo2_base.equals(olmo2_sft) and olmo2_base.equals(olmo2_dpo), (
    f"{CANONICAL_HUMAN_FAMILY} family human_continuation is not identical "
    "across base/sft/dpo — aborting, canonical family assumption violated."
)
print(f"Verified: human_continuation identical across "
      f"{CANONICAL_HUMAN_FAMILY}_base/sft/dpo.")

human_ref = model_dfs[f'{CANONICAL_HUMAN_FAMILY}_base'][['prompt_id', 'human_continuation']].drop_duplicates()

human_rows = pd.DataFrame({
    'prompt_id':  human_ref['prompt_id'],
    'genre':      'acl',
    'source':     'human',
    'model':      None,
    'stage':      None,
    'checkpoint': None,
    'prompt_style': None,
    'text':       human_ref['human_continuation'],
    'is_usable':  True,
})

# ── Combine ───────────────────────────────────────────────────────────────────
master = pd.concat([human_rows, generated_rows], ignore_index=True)

# ── Merge with acl_intro_final.csv for year/venue ──────────────────────────
intro = pd.read_csv(os.path.join(DATA_DIR, 'acl_intro_final.csv'))[['prompt_id', 'year', 'venue']]
master = master.merge(intro, on='prompt_id', how='left')

# ── Final schema order ──────────────────────────────────────────────────────
master = master[['prompt_id', 'genre', 'source', 'model', 'stage', 'checkpoint',
                  'prompt_style', 'text', 'is_usable', 'year', 'venue']]

master = master.sort_values(['prompt_id', 'source', 'model', 'stage']).reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, 'master_acl.csv')
master.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# ── Report ───────────────────────────────────────────────────────────────────
print(f"\nCanonical human reference family: {CANONICAL_HUMAN_FAMILY} "
      f"(tokenizer: {CANONICAL_HUMAN_TOKENIZER})")

print(f"\nTotal rows: {len(master)}")

print(f"\nRows per source:")
print(master['source'].value_counts().to_string())

print(f"\nRows per stage (human rows have stage=NaN):")
print(master['stage'].value_counts(dropna=False).to_string())

print(f"\nNull counts per column:")
print(master.isnull().sum().to_string())

n_human = (master['source'] == 'human').sum()
n_generated = (master['source'] == 'generated').sum()
print(f"\nHuman rows = {n_human} (expected 1000): {'OK' if n_human == 1000 else 'MISMATCH'}")
print(f"Generated rows = {n_generated} (expected 9000): {'OK' if n_generated == 9000 else 'MISMATCH'}")
