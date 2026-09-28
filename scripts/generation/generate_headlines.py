import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate headlines from LLM checkpoints")
parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
parser.add_argument("--family", required=True, help="Model family (olmo2, llama, mistral)")
parser.add_argument("--stage", required=True, help="Training stage (base, sft, dpo)")
args = parser.parse_args()

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID = args.model_id
MODEL_FAMILY = args.family
STAGE = args.stage
MAX_INPUT_TOKENS = 250
MIN_NEW_TOKENS = 8
MAX_NEW_TOKENS = 20
CHECKPOINT_EVERY = 100

DATA_PATH = os.path.expandvars("$WORK/data/news_articles_1000.csv")
OUTPUT_PATH = os.path.expandvars(f"$WORK/outputs/{MODEL_FAMILY}_{STAGE}_headlines.csv")
CHECKPOINT_PATH = os.path.expandvars(f"$WORK/outputs/{MODEL_FAMILY}_{STAGE}_checkpoint.csv")

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} articles")

# ── Resume from checkpoint if exists ──────────────────────────────────────────
if os.path.exists(CHECKPOINT_PATH):
    completed = pd.read_csv(CHECKPOINT_PATH)
    completed_ids = set(completed["prompt_id"].tolist())
    df = df[~df["prompt_id"].isin(completed_ids)]
    results = completed.to_dict("records")
    print(f"Resuming from checkpoint: {len(completed_ids)} done, {len(df)} remaining")
else:
    results = []
    print("Starting fresh generation")

# ── Load model and tokenizer ───────────────────────────────────────────────────
print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"Model loaded on: {next(model.parameters()).device}")

# ── Generation ─────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(OUTPUT_PATH)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

for idx, row in df.iterrows():
    prompt_id = row["prompt_id"]
    article = row["article"]
    human_headline = row["title"]

    article_tokens = tokenizer.encode(article, add_special_tokens=False)
    truncated_tokens = article_tokens[:MAX_INPUT_TOKENS]
    truncated_article = tokenizer.decode(truncated_tokens)
    actual_input_tokens = len(truncated_tokens)

    prompt = f"Article: {truncated_article}\nHeadline:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            min_new_tokens=MIN_NEW_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    generated_text = generated_text.split('\n')[0].strip()
    output_token_count = len(generated_tokens)

    print(f"[{prompt_id}] Human:     {human_headline}")
    print(f"[{prompt_id}] Generated: {generated_text}")

    results.append({
        "prompt_id": prompt_id,
        "model": MODEL_FAMILY,
        "stage": STAGE,
        "checkpoint": MODEL_ID,
        "human_headline": human_headline,
        "generated_headline": generated_text,
        "input_tokens": actual_input_tokens,
        "output_tokens": output_token_count
    })

    if len(results) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        print(f"Checkpoint saved: {len(results)} articles done")

# ── Save final output ──────────────────────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(out_df)} results to {OUTPUT_PATH}")

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("Checkpoint file removed")