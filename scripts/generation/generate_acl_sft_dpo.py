import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate ACL intro continuations")
parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
parser.add_argument("--family",   required=True, help="Model family (olmo2, llama, mistral)")
parser.add_argument("--stage",    required=True, help="Training stage (base, sft, dpo)")
args = parser.parse_args()

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID       = args.model_id
MODEL_FAMILY   = args.family
STAGE          = args.stage
INPUT_TOKENS   = 200        # exact token split point
MIN_NEW_TOKENS = 200        # minimum generation length
MAX_NEW_TOKENS = 500        # maximum generation length
CHECKPOINT_EVERY = 100

DATA_PATH       = os.path.expandvars("$WORK/data/acl_intro_final.csv")
OUTPUT_PATH     = os.path.expandvars(f"$WORK/outputs/{MODEL_FAMILY}_{STAGE}_acl.csv")
CHECKPOINT_PATH = os.path.expandvars(f"$WORK/outputs/{MODEL_FAMILY}_{STAGE}_acl_checkpoint.csv")

# ── Instruction prompt for SFT and DPO models ──────────────────────────────────
# Base models use completion-style (raw text only).
# SFT and DPO models use instruction-style to elicit academic prose.
INSTRUCTION_TEMPLATE = (
    "Continue the following NLP paper introduction in formal academic prose, "
    "staying on the same topic and writing style. "
    "Write only the continuation itself. "
    "Use flowing paragraphs only. "
    "Do not use headings, bullet points, numbered lists, tables, "
    "Markdown, URLs, references, or BibTeX.\n\n"
    "Text:\n{input_text}\n\nContinuation:"
)

def build_prompt(input_text, stage):
    """
    Base models: raw completion prompt (no instruction wrapper).
    SFT/DPO models: instruction-style prompt.
    """
    if stage == "base":
        return input_text
    else:
        return INSTRUCTION_TEMPLATE.format(input_text=input_text)

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} ACL introductions")
print(f"Prompt style: {'completion (base)' if STAGE == 'base' else 'instruction (sft/dpo)'}")

# ── Resume from checkpoint if exists ──────────────────────────────────────────
if os.path.exists(CHECKPOINT_PATH):
    completed     = pd.read_csv(CHECKPOINT_PATH)
    completed_ids = set(completed["prompt_id"].tolist())
    df            = df[~df["prompt_id"].isin(completed_ids)]
    results       = completed.to_dict("records")
    print(f"Resuming: {len(completed_ids)} done, {len(df)} remaining")
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

# ── Output directory ───────────────────────────────────────────────────────────
out_dir = os.path.dirname(OUTPUT_PATH)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

# ── Generation ─────────────────────────────────────────────────────────────────
for idx, row in df.iterrows():
    prompt_id    = row["prompt_id"]
    introduction = row["introduction"]

    # Tokenize full introduction
    all_tokens = tokenizer.encode(introduction, add_special_tokens=False)

    # Split at exactly INPUT_TOKENS
    input_tokens_list   = all_tokens[:INPUT_TOKENS]
    continuation_tokens = all_tokens[INPUT_TOKENS:]

    # Decode input text
    input_text         = tokenizer.decode(input_tokens_list)
    human_continuation = tokenizer.decode(continuation_tokens)

    # Skip if no human continuation left
    if not human_continuation.strip():
        print(f"[{prompt_id}] Skipped — no human continuation after {INPUT_TOKENS} tokens")
        continue

    # Build prompt — instruction-style for SFT/DPO, completion for base
    prompt = build_prompt(input_text, STAGE)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            min_new_tokens=MIN_NEW_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens   = output[0][inputs["input_ids"].shape[1]:]
    generated_text     = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    output_token_count = len(generated_tokens)

    print(f"[{prompt_id}] output_tokens={output_token_count} | human_continuation_tokens={len(continuation_tokens)}")

    results.append({
        "prompt_id":                 prompt_id,
        "model":                     MODEL_FAMILY,
        "stage":                     STAGE,
        "checkpoint":                MODEL_ID,
        "input_text":                input_text,
        "human_continuation":        human_continuation,
        "generated_text":            generated_text,
        "input_tokens":              len(input_tokens_list),
        "output_tokens":             output_token_count,
        "human_continuation_tokens": len(continuation_tokens),
    })

    if len(results) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        print(f"Checkpoint saved: {len(results)} done")

# ── Save final output ──────────────────────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(out_df)} results to {OUTPUT_PATH}")

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("Checkpoint removed")