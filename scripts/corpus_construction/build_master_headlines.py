import pandas as pd
import os

output_dir = '/Users/dewansajol/Study/Uni Bamberg/Thesis/outputs/'
data_dir = '/Users/dewansajol/Study/Uni Bamberg/Thesis/data/'

# ── Load all 9 model output CSVs ───────────────────────────────────────────────
model_dfs = []
for f in sorted(os.listdir(output_dir)):
    if f.endswith('_headlines.csv'):
        df = pd.read_csv(os.path.join(output_dir, f))
        model_dfs.append(df)

generated = pd.concat(model_dfs, ignore_index=True)

# ── Build generated rows ───────────────────────────────────────────────────────
generated_rows = pd.DataFrame({
    'prompt_id': generated['prompt_id'],
    'genre': 'headlines',
    'source': 'generated',
    'model': generated['model'],
    'stage': generated['stage'],
    'checkpoint': generated['checkpoint'],
    'text': generated['generated_headline'],
    'input_tokens': generated['input_tokens'],
    'output_tokens': generated['output_tokens'],
})

# ── Build human rows ───────────────────────────────────────────────────────────
# Take human headlines from any one CSV — they're identical across all 9
human_ref = generated[generated['model'] == 'olmo2'][['prompt_id', 'human_headline']].drop_duplicates()

human_rows = pd.DataFrame({
    'prompt_id': human_ref['prompt_id'],
    'genre': 'headlines',
    'source': 'human',
    'model': None,
    'stage': None,
    'checkpoint': None,
    'text': human_ref['human_headline'],
    'input_tokens': None,
    'output_tokens': None,
})

# ── Combine and sort ───────────────────────────────────────────────────────────
master = pd.concat([human_rows, generated_rows], ignore_index=True)
master = master.sort_values(['prompt_id', 'source', 'model', 'stage']).reset_index(drop=True)

print(f'Total rows: {len(master)}')
print(f'Sources: {master["source"].value_counts().to_dict()}')
print(f'Models: {master["model"].value_counts().to_dict()}')
print(f'Stages: {master["stage"].value_counts().to_dict()}')

# ── Save ──────────────────────────────────────────────────────────────────────
master.to_csv(os.path.join(data_dir, 'master_headlines.csv'), index=False)
print('Saved master_headlines.csv')