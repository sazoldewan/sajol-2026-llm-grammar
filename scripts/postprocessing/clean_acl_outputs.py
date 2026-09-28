"""
Clean ACL generation outputs: repeat-onset truncation + sentence trimming.

Two data-quality issues were found in the 9 ACL generation output CSVs:
  1. 78.4% of generated texts hit the MAX_NEW_TOKENS=500 ceiling and were
     cut off mid-sentence.
  2. Base checkpoints showed high rates of degenerate repetition under
     greedy decoding (up to 78.1% for OLMo-2 base), consistent with the
     neural text degeneration problem (Holtzman et al., 2020).

Two-step cleaning procedure, applied to generated_text:
  1. Repeat-onset truncation: a text is flagged as repetitive if any
     10-word sequence appears 3+ times. If flagged, the text is truncated
     at the onset of the first repeated span (i.e. right before the
     repeat begins), preserving the non-degenerate prefix.
  2. Sentence-boundary trim: applied to ALL rows regardless of whether
     repetition was detected. Trims to the last complete sentence
     (ending in '.', '!', or '?'). This also resolves the mid-sentence
     token-ceiling truncations from issue 1. If no sentence boundary
     exists in the (possibly repeat-truncated) text, the text is kept
     as-is rather than emptied.

Output:
  - New column generated_text_clean holds the cleaned text.
  - Original generated_text column is preserved intact.
  - New column is_usable is False if generated_text_clean has fewer than
    30 words (near-empty after cleaning), True otherwise. Usability
    threshold rationale: below 30 words, Stanza has almost nothing to
    parse and feature rates become meaningless.

Files are modified in place (9 files in outputs/acl/*_acl.csv).
"""

import glob
import os
import re
from collections import Counter

import pandas as pd

DATA_DIR = "/Users/dewansajol/Study/Uni Bamberg/Thesis/outputs/acl"
USABILITY_MIN_WORDS = 30


def find_repeat_onset_word_idx(words, n=10, min_repeats=3):
    """Return the word index where a repeated n-gram's 2nd occurrence
    begins, or None if no n-gram repeats min_repeats+ times."""
    if len(words) < n:
        return None
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    best_start = None
    for i, ng in enumerate(ngrams):
        if counts[ng] >= min_repeats:
            best_start = i
            break
    if best_start is None:
        return None
    target = ngrams[best_start]
    occurrence_positions = [i for i, ng in enumerate(ngrams) if ng == target]
    return occurrence_positions[1]


def clean_text(text):
    """Returns (clean_text, was_repeat_truncated, was_sentence_trimmed, hit_fallback)."""
    words = text.split()
    onset_idx = find_repeat_onset_word_idx(words)

    if onset_idx is not None:
        stage1 = ' '.join(words[:onset_idx])
        repeat_truncated = True
    else:
        stage1 = text
        repeat_truncated = False

    matches = list(re.finditer(r'[.!?]', stage1))
    if matches:
        cut = matches[-1].end()
        final = stage1[:cut].strip()
        sentence_trimmed = (final != stage1.strip())
        fallback = False
    else:
        final = stage1.strip()
        sentence_trimmed = False
        fallback = True

    return final, repeat_truncated, sentence_trimmed, fallback


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_acl.csv")))
    print(f"Found {len(files)} files in {DATA_DIR}")

    report_rows = []

    for f in files:
        name = os.path.basename(f)
        df = pd.read_csv(f)

        orig_word_count = df["generated_text"].apply(lambda t: len(str(t).split()))
        tok_per_word = df["output_tokens"] / orig_word_count

        results = df["generated_text"].apply(lambda t: clean_text(str(t)))
        df["generated_text_clean"] = results.apply(lambda r: r[0])
        repeat_trunc_mask = results.apply(lambda r: r[1])
        sentence_trim_mask = results.apply(lambda r: r[2])
        fallback_mask = results.apply(lambda r: r[3])

        clean_word_count = df["generated_text_clean"].apply(lambda t: len(str(t).split()))
        est_tokens_after = (clean_word_count * tok_per_word).round().astype(int)

        df["is_usable"] = clean_word_count >= USABILITY_MIN_WORDS

        df.to_csv(f, index=False)

        any_change_mask = repeat_trunc_mask | sentence_trim_mask
        n_unusable = (~df["is_usable"]).sum()

        report_rows.append({
            "file": name,
            "n_rows": len(df),
            "before_mean_tok": df["output_tokens"].mean(),
            "before_median_tok": df["output_tokens"].median(),
            "after_mean_tok_est": est_tokens_after.mean(),
            "after_median_tok_est": est_tokens_after.median(),
            "n_repeat_truncated": repeat_trunc_mask.sum(),
            "n_sentence_trimmed": sentence_trim_mask.sum(),
            "n_any_truncated": any_change_mask.sum(),
            "n_fallback_no_sentence_boundary": fallback_mask.sum(),
            "n_unusable": n_unusable,
        })

    report_df = pd.DataFrame(report_rows)

    print("\nSaved generated_text_clean + is_usable to all files.\n")
    print(report_df.to_string(index=False))

    total_unusable = report_df["n_unusable"].sum()
    total_rows = report_df["n_rows"].sum()
    print(f"\nTotal unusable rows: {total_unusable} / {total_rows} "
          f"({total_unusable / total_rows * 100:.1f}%)")


if __name__ == "__main__":
    main()
