"""
Headline data quality check.
Run this before feeding master_headlines.csv to Stanza.

Checks:
  - Missing or empty text
  - Alignment (each prompt_id has exactly 1 human + 9 generated rows)
  - Word count distribution per group (target: 8-16 words)
  - Truncation artifacts in generated headlines
    (dangling prepositions/conjunctions, trailing comma, mid-word apostrophe cut)

Usage:
    python headline_quality_check.py
"""

import csv
import re
from collections import Counter, defaultdict

MASTER_FILE = "/Users/dewansajol/Study/Uni Bamberg/Thesis/data/master_headlines.csv"

WORD_MIN = 8
WORD_MAX = 16

EXPECTED_GROUPS = [
    ("mistral", "base"), ("mistral", "sft"), ("mistral", "dpo"),
    ("llama",   "base"), ("llama",   "sft"), ("llama",   "dpo"),
    ("olmo2",   "base"), ("olmo2",   "sft"), ("olmo2",   "dpo"),
]

# Words that signal the headline was cut before completing a thought
DANGLERS = {
    'the', 'a', 'an', 'of', 'in', 'on', 'to', 'for', 'with', 'from', 'by',
    'at', 'as', 'and', 'or', 'but', 'that', 'which', 'when', 'where', 'while',
    'after', 'before', 'during', 'over', 'under', 'about', 'against', 'between',
    'into', 'through', 'than', 'if', 'although', 'because', 'since', 'until',
    'including', 'amid', 'following', 'per', 'via', 'its', 'their', 'his',
    'her', 'our', 'this', 'these', 'those', 'not', 'up', 'out',
    'also', 'just', 'still', 'already', 'now', 'be', 'been', 'is', 'are',
    'was', 'were', 'will', 'would', 'could', 'should', 'has', 'have', 'had',
}


def load_master(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def word_count(text):
    return len(text.strip().split())


def is_truncated(text):
    h = text.strip()
    if not h:
        return True, "empty"
    if h.endswith(','):
        return True, "trailing comma"
    if h.endswith(" '"):
        return True, "mid-word apostrophe cut"
    if re.search(r"[a-zA-Z][ntdr]'$", h):
        return True, "mid-word apostrophe cut"
    last_word = h.split()[-1].lower().rstrip("',;:")
    if last_word in DANGLERS:
        return True, f"dangling '{last_word}'"
    return False, "ok"


def check_alignment(rows):
    """Each prompt_id should appear exactly once as human and once per model/stage."""
    by_pid = defaultdict(list)
    for row in rows:
        by_pid[row['prompt_id']].append(row)

    issues = []
    for pid, pid_rows in sorted(by_pid.items()):
        human_rows = [r for r in pid_rows if r['source'] == 'human']
        if len(human_rows) != 1:
            issues.append(f"{pid}: {len(human_rows)} human rows (expected 1)")
        for model, stage in EXPECTED_GROUPS:
            matches = [r for r in pid_rows if r['model'] == model and r['stage'] == stage]
            if len(matches) != 1:
                issues.append(f"{pid}: {len(matches)} rows for {model}/{stage} (expected 1)")
    return issues


def analyze_group(rows, label):
    total = len(rows)
    empty = [r for r in rows if not r['text'].strip()]
    wcs = [word_count(r['text']) for r in rows if r['text'].strip()]

    short    = sum(1 for w in wcs if w < WORD_MIN)
    in_range = sum(1 for w in wcs if WORD_MIN <= w <= WORD_MAX)
    long_    = sum(1 for w in wcs if w > WORD_MAX)
    avg_wc   = sum(wcs) / len(wcs) if wcs else 0

    trunc = [(r, reason) for r in rows
             if r['text'].strip() and is_truncated(r['text'])[0]
             for _, reason in [is_truncated(r['text'])]]

    return {
        'label': label,
        'total': total,
        'empty': empty,
        'short': short,
        'in_range': in_range,
        'long': long_,
        'avg_wc': avg_wc,
        'truncated': trunc,
    }


def print_summary_table(group_results):
    col = f"{'Group':<22} {'N':>5} {'Empty':>6} {'<8w':>5} {'8-16w':>7} {'>16w':>6} {'Trunc':>7} {'Trunc%':>7} {'AvgW':>6}"
    print(col)
    print("-" * len(col))
    for r in group_results:
        n_trunc = len(r['truncated'])
        trunc_pct = n_trunc / r['total'] * 100 if r['total'] else 0
        print(
            f"{r['label']:<22} {r['total']:>5} {len(r['empty']):>6} "
            f"{r['short']:>5} {r['in_range']:>7} {r['long']:>6} "
            f"{n_trunc:>7} {trunc_pct:>6.1f}% {r['avg_wc']:>6.1f}"
        )


def main():
    rows = load_master(MASTER_FILE)
    print(f"Loaded {len(rows)} rows from master file.\n")

    # --- 1. Alignment check ---
    print("=" * 65)
    print("1. ALIGNMENT CHECK")
    print("=" * 65)
    issues = check_alignment(rows)
    if issues:
        print(f"  {len(issues)} alignment issue(s) found:")
        for issue in issues[:20]:
            print(f"    {issue}")
        if len(issues) > 20:
            print(f"    ... and {len(issues) - 20} more")
    else:
        print("  OK — every prompt_id has exactly 1 human + 9 generated rows.")

    # --- 2. Per-group quality summary ---
    print("\n" + "=" * 65)
    print("2. QUALITY SUMMARY BY GROUP")
    print("=" * 65)
    print(f"   Target word range: {WORD_MIN}–{WORD_MAX} words\n")

    human_rows = [r for r in rows if r['source'] == 'human']
    group_results = [analyze_group(human_rows, "human")]

    for model, stage in EXPECTED_GROUPS:
        grp = [r for r in rows if r['model'] == model and r['stage'] == stage]
        group_results.append(analyze_group(grp, f"{model}_{stage}"))

    print_summary_table(group_results)

    # --- 3. Truncation examples for generated groups ---
    print("\n" + "=" * 65)
    print("3. TRUNCATION EXAMPLES (generated only, up to 5 per group)")
    print("=" * 65)
    for r in group_results[1:]:  # skip human
        if not r['truncated']:
            continue
        type_counts = Counter(reason for _, reason in r['truncated'])
        print(f"\n  {r['label']} — {len(r['truncated'])} truncated "
              f"({', '.join(f'{c} {t}' for t, c in type_counts.most_common())})")
        for row, reason in r['truncated'][:5]:
            print(f"    {row['prompt_id']} [{reason}]: {row['text']}")

    # --- 4. Overall pass/fail ---
    print("\n" + "=" * 65)
    print("4. OVERALL PASS / FAIL")
    print("=" * 65)
    total_empty = sum(len(r['empty']) for r in group_results)
    total_trunc = sum(len(r['truncated']) for r in group_results[1:])
    total_generated = sum(r['total'] for r in group_results[1:])

    print(f"  Empty rows (all groups):        {total_empty}")
    print(f"  Truncated generated headlines:  {total_trunc} / {total_generated} "
          f"({total_trunc / total_generated * 100:.1f}%)")
    print(f"  Alignment issues:               {len(issues)}")

    if total_empty == 0 and len(issues) == 0:
        print("\n  PASS — data is clean. Safe to feed to Stanza.")
    else:
        print("\n  REVIEW — fix the issues above before parsing.")


if __name__ == "__main__":
    main()
