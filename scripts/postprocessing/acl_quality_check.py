"""
ACL Generation Output Quality Check
====================================
Reads all 9 ACL generation output CSVs and reproduces the statistics
reported in the thesis methods section.

Input: /Users/dewansajol/Study/Uni Bamberg/Thesis/outputs/acl/*_acl.csv
Expected columns: prompt_id, model, stage, checkpoint, input_text,
human_continuation, generated_text, input_tokens, output_tokens,
human_continuation_tokens, generated_text_clean, is_usable

Repetition (is_repetitive) is NOT stored in the CSVs — it is computed
here at runtime from generated_text: a row is repetitive if any 10-word
sequence appears 3+ times.

Reports:
  1. Total rows per file and overall
  2. Null counts per column per file
  3. % of generated_text hitting exactly 500 tokens (overall + per checkpoint)
  4. Repetition rate per checkpoint (count + %)
  5. Usability rate per checkpoint — is_usable == False (count + %), plus overall total
  6. generated_text_clean token distribution per checkpoint (min/max/mean/median)
  7. human_continuation_tokens distribution per checkpoint family (min/max/mean/median)
  8. Summary table
  9. PASS/FAIL verdict (PASS iff all 9 files have 1,000 rows and zero nulls)
"""

import glob
import os
from collections import Counter

import pandas as pd

DATA_DIR = "/Users/dewansajol/Study/Uni Bamberg/Thesis/outputs/acl"
TOKEN_CEILING = 500
EXPECTED_ROWS_PER_FILE = 1000


def is_repetitive(text, n=10, min_repeats=3):
    words = str(text).split()
    if len(words) < n:
        return False
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    return max(counts.values()) >= min_repeats if counts else False


def load_all_files():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_acl.csv")))
    data = {}
    for p in paths:
        name = os.path.basename(p)
        df = pd.read_csv(p)
        df["is_repetitive"] = df["generated_text"].apply(is_repetitive)
        data[name] = df
    return data


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    data = load_all_files()
    print(f"Loaded {len(data)} files from {DATA_DIR}")

    # ── 1. Row counts ────────────────────────────────────────────────────
    section("1. ROW COUNTS PER FILE")
    total_rows = 0
    row_count_ok = True
    for name, df in data.items():
        n = len(df)
        total_rows += n
        ok = n == EXPECTED_ROWS_PER_FILE
        row_count_ok &= ok
        print(f"{name:24s} {n:6d} rows  [{'OK' if ok else 'MISMATCH'}]")
    print(f"\nOverall total: {total_rows} rows (expected {len(data) * EXPECTED_ROWS_PER_FILE})")

    # ── 2. Null counts ───────────────────────────────────────────────────
    section("2. NULL COUNTS PER COLUMN PER FILE")
    any_nulls = False
    for name, df in data.items():
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if len(nulls) > 0:
            any_nulls = True
            print(f"{name}:")
            print(nulls.to_string())
        else:
            print(f"{name}: no nulls")

    # ── 3. % hitting 500-token ceiling ──────────────────────────────────
    section(f"3. % OF generated_text HITTING EXACTLY {TOKEN_CEILING} TOKENS")
    total_at_ceiling = 0
    for name, df in data.items():
        at_ceiling = (df["output_tokens"] == TOKEN_CEILING).sum()
        total_at_ceiling += at_ceiling
        print(f"{name:24s} {at_ceiling:5d} / {len(df):5d}  ({at_ceiling / len(df) * 100:5.1f}%)")
    print(f"\nOverall: {total_at_ceiling} / {total_rows} ({total_at_ceiling / total_rows * 100:.1f}%)")

    # ── 4. Repetition rate per checkpoint ───────────────────────────────
    section("4. REPETITION RATE PER CHECKPOINT (is_repetitive == True)")
    total_repetitive = 0
    for name, df in data.items():
        n_rep = df["is_repetitive"].sum()
        total_repetitive += n_rep
        print(f"{name:24s} {n_rep:5d} / {len(df):5d}  ({n_rep / len(df) * 100:5.1f}%)")
    print(f"\nOverall: {total_repetitive} / {total_rows} ({total_repetitive / total_rows * 100:.1f}%)")

    # ── 5. Usability rate per checkpoint ────────────────────────────────
    section("5. USABILITY RATE PER CHECKPOINT (is_usable == False)")
    total_unusable = 0
    for name, df in data.items():
        n_unusable = (~df["is_usable"]).sum()
        total_unusable += n_unusable
        print(f"{name:24s} {n_unusable:5d} / {len(df):5d}  ({n_unusable / len(df) * 100:5.1f}%)")
    print(f"\nOverall: {total_unusable} / {total_rows} ({total_unusable / total_rows * 100:.1f}%)")

    # ── 6. generated_text_clean token distribution per checkpoint ──────
    section("6. generated_text_clean TOKEN DISTRIBUTION PER CHECKPOINT")
    print(f"{'file':24s} {'min':>6s} {'max':>6s} {'mean':>8s} {'median':>8s}")
    clean_token_stats = {}
    for name, df in data.items():
        orig_wc = df["generated_text"].apply(lambda t: len(str(t).split()))
        clean_wc = df["generated_text_clean"].apply(lambda t: len(str(t).split()))
        tok_per_word = df["output_tokens"] / orig_wc
        est_tokens = (clean_wc * tok_per_word).round()
        clean_token_stats[name] = est_tokens
        print(f"{name:24s} {est_tokens.min():6.0f} {est_tokens.max():6.0f} "
              f"{est_tokens.mean():8.1f} {est_tokens.median():8.1f}")
    print("\n(generated_text_clean token counts are estimated via each row's own "
          "output_tokens / word_count ratio, not exact retokenization.)")

    # ── 7. human_continuation_tokens distribution per checkpoint family ─
    section("7. human_continuation_tokens DISTRIBUTION PER CHECKPOINT FAMILY")
    print(f"{'model':14s} {'min':>6s} {'max':>6s} {'mean':>8s} {'median':>8s}")
    seen_families = set()
    for name, df in data.items():
        model = df["model"].iloc[0]
        if model in seen_families:
            continue
        seen_families.add(model)
        hc = df["human_continuation_tokens"]
        print(f"{model:14s} {hc.min():6d} {hc.max():6d} {hc.mean():8.1f} {hc.median():8.1f}")

    # ── 8. Summary table ────────────────────────────────────────────────
    section("8. SUMMARY TABLE")
    header = (f"{'checkpoint':24s} {'rows':>6s} {'nulls':>6s} {'@500':>6s} "
              f"{'rep':>6s} {'unusable':>9s} {'clean_mean_tok':>15s}")
    print(header)
    print("-" * len(header))
    for name, df in data.items():
        rows = len(df)
        nulls = df.isnull().sum().sum()
        at_ceiling = (df["output_tokens"] == TOKEN_CEILING).sum()
        rep = df["is_repetitive"].sum()
        unusable = (~df["is_usable"]).sum()
        clean_mean = clean_token_stats[name].mean()
        print(f"{name:24s} {rows:6d} {nulls:6d} {at_ceiling:6d} "
              f"{rep:6d} {unusable:9d} {clean_mean:15.1f}")

    # ── 9. PASS/FAIL verdict ────────────────────────────────────────────
    section("9. VERDICT")
    total_nulls = sum(df.isnull().sum().sum() for df in data.values())
    verdict_pass = row_count_ok and (total_nulls == 0) and (len(data) == 9)
    print(f"All 9 files present:      {'YES' if len(data) == 9 else 'NO'}")
    print(f"All files have 1,000 rows: {'YES' if row_count_ok else 'NO'}")
    print(f"Zero nulls across all files: {'YES' if total_nulls == 0 else f'NO ({total_nulls} nulls)'}")
    print()
    print("PASS" if verdict_pass else "FAIL")


if __name__ == "__main__":
    main()
