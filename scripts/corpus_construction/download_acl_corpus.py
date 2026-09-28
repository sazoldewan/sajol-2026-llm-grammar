"""
ACL Introduction Corpus — Single-Pass Collection Script
=========================================================
Streams KRLabsOrg/acl-anthology-md (fulltext config), applies ALL filters
in one pass, and saves acl_intro_final.csv directly.

Requirements:
  - English papers only (non-English venues excluded)
  - Year: 2015-2020
  - Introduction section extracted via regex
  - Minimum 300 words
  - No markdown tables in introduction
  - No image placeholders
  - No figure/table captions
  - No math-heavy content (LaTeX > 10%)
  - No code blocks
  - No copyright/footer markers
  - No non-ASCII content > 1% of tokens
  - No combining diacritical marks (garbled font encoding)
  - No GLYPH artefacts

Output schema:
  prompt_id, anthology_id, year, venue, introduction, word_count
"""

import os
import re
import unicodedata
import pandas as pd
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.expandvars("$WORK/data/acl_intro_final.csv")

YEAR_MIN = 2015
YEAR_MAX = 2020

# Excluded non-English venues
NON_ENGLISH_VENUES = {
    'jeptalnrecital', 'clicit', 'ccl', 'clib', 'rocling',
    'ijclclp', 'tal', 'icon'  # tal=French, icon=Indian NLP (mixed languages)
}

MIN_WORDS = 300
MATH_TOKEN_THRESHOLD = 0.10  # exclude if LaTeX tokens > 10% of words
NON_ASCII_THRESHOLD  = 0.01  # exclude if non-ASCII tokens > 1% of words
PROGRESS_EVERY       = 5000

# Copyright/footer markers that bleed in from PDF conversion
COPYRIGHT_MARKERS = [
    '©', 'creative commons', 'association for machine translation',
    'all rights reserved', 'this work is licensed',
    'european association', 'the authors. this article',
]

# ── Parse anthology_id ──────────────────────────────────────────────────────

def parse_anthology_id(anthology_id):
    """
    Extract year and venue from anthology_id.
    Format: YEAR.VENUE-type.NUM  e.g. 2020.acl-main.123
    Returns (year, venue) or (None, None) if unparseable.
    """
    if not isinstance(anthology_id, str):
        return None, None
    parts = anthology_id.split('.')
    if len(parts) < 2:
        return None, None
    try:
        year = int(parts[0])
    except ValueError:
        return None, None
    venue = parts[1].split('-')[0].lower()
    return year, venue

# ── Introduction extraction ─────────────────────────────────────────────────

def extract_introduction(text):
    """
    Extract introduction section from markdown.
    Finds Introduction heading and extracts until next section heading.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    intro_pattern = re.compile(
        r'^#{1,3}\s*(?:\d+[\.\s]+)?introduction\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    section_pattern = re.compile(
        r'^#{1,3}\s+\S',
        re.MULTILINE
    )

    match = intro_pattern.search(text)
    if not match:
        return None

    intro_start = match.end()
    next_section = section_pattern.search(text, intro_start)
    intro_text = text[intro_start:next_section.start()] if next_section else text[intro_start:]

    return intro_text.strip() or None

# ── Cleaning ────────────────────────────────────────────────────────────────

def clean_introduction(text):
    """
    Strip known artefacts from introduction text:
    - Image placeholders
    - Figure captions
    - Table captions
    - Collapse multiple blank lines
    - Collapse multiple spaces (horizontal only)
    """
    # Image placeholders
    text = text.replace('<!-- image -->', '')
    text = text.replace('<!-- formula-not-decoded -->', '')

    # Figure and table captions
    text = re.sub(r'\bFigure\s+\d+[:\.].*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bTable\s+\d+[:\.].*?(?=\n|$)',  '', text, flags=re.IGNORECASE)

    # Collapse multiple spaces (not newlines)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# ── Quality filters ─────────────────────────────────────────────────────────

def has_markdown_table(text):
    """True if 3+ consecutive lines start with |"""
    lines = text.split('\n')
    count = 0
    for line in lines:
        if line.strip().startswith('|'):
            count += 1
            if count >= 3:
                return True
        else:
            count = 0
    return False

def has_image(text):
    """True if text contains markdown image syntax or HTML img tags"""
    return bool(re.search(r'!\[|<img', text, re.IGNORECASE))

def has_code_blocks(text):
    """True if text contains fenced code blocks"""
    return bool(re.search(r'```|~~~', text))

def is_math_heavy(text):
    """True if LaTeX tokens exceed threshold"""
    words = text.split()
    if not words:
        return True
    math_tokens = sum(1 for w in words if re.match(r'\\\w+', w) or '$' in w)
    return (math_tokens / len(words)) > MATH_TOKEN_THRESHOLD

def has_copyright_markers(text):
    """True if text contains copyright/footer markers"""
    text_lower = text.lower()
    return any(marker in text_lower for marker in COPYRIGHT_MARKERS)

def has_glyph_artefacts(text):
    """True if text contains GLYPH encoding artefacts"""
    return 'GLYPH' in text

def has_combining_characters(text):
    """True if text contains combining diacritical marks (garbled font encoding)"""
    return any(unicodedata.category(c) == 'Mn' for c in text)

def non_ascii_fraction(text):
    """Fraction of tokens containing non-ASCII characters"""
    words = text.split()
    if not words:
        return 1.0
    non_ascii = sum(1 for w in words if any(ord(c) > 127 for c in w))
    return non_ascii / len(words)

def has_character_spacing(text):
    """
    True if text contains character-spaced words (PDF OCR artefact).
    Detects 4+ consecutive single-letter tokens separated by spaces.
    e.g. 'N o w a d a y s'
    """
    tokens = text.split()
    consecutive = 0
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            consecutive += 1
            if consecutive >= 4:
                return True
        else:
            consecutive = 0
    return False

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading ACL Anthology dataset (streaming)...")
    dataset = load_dataset(
        "KRLabsOrg/acl-anthology-md",
        "fulltext",
        split="train",
        streaming=True,
    )

    results  = []
    streamed = 0
    skipped  = {
        'year':            0,
        'venue':           0,
        'no_intro':        0,
        'too_short':       0,
        'has_table':       0,
        'has_image':       0,
        'has_code':        0,
        'math_heavy':      0,
        'copyright':       0,
        'glyph':           0,
        'combining_chars': 0,
        'non_ascii':       0,
        'char_spacing':    0,
    }

    for paper in dataset:
        streamed += 1

        if streamed % PROGRESS_EVERY == 0:
            print(f"  Streamed {streamed:,} — collected {len(results):,} so far...")

        # ── Parse year and venue ────────────────────────────────────────────
        anthology_id = paper.get('anthology_id', '')
        year, venue  = parse_anthology_id(anthology_id)

        if year is None or not (YEAR_MIN <= year <= YEAR_MAX):
            skipped['year'] += 1
            continue

        if venue in NON_ENGLISH_VENUES:
            skipped['venue'] += 1
            continue

        # ── Get markdown text ───────────────────────────────────────────────
        text = paper.get('markdown', '')
        if not text:
            skipped['no_intro'] += 1
            continue

        # ── Extract introduction ────────────────────────────────────────────
        intro = extract_introduction(text)
        if not intro:
            skipped['no_intro'] += 1
            continue

        # ── Clean introduction ──────────────────────────────────────────────
        intro = clean_introduction(intro)
        if not intro:
            skipped['no_intro'] += 1
            continue

        # ── Apply all quality filters ───────────────────────────────────────
        if has_markdown_table(intro):
            skipped['has_table'] += 1
            continue

        if has_image(intro):
            skipped['has_image'] += 1
            continue

        if has_code_blocks(intro):
            skipped['has_code'] += 1
            continue

        if is_math_heavy(intro):
            skipped['math_heavy'] += 1
            continue

        if has_copyright_markers(intro):
            skipped['copyright'] += 1
            continue

        if has_glyph_artefacts(intro):
            skipped['glyph'] += 1
            continue

        if has_combining_characters(intro):
            skipped['combining_chars'] += 1
            continue

        if non_ascii_fraction(intro) > NON_ASCII_THRESHOLD:
            skipped['non_ascii'] += 1
            continue

        if has_character_spacing(intro):
            skipped['char_spacing'] += 1
            continue

        # ── Check minimum word count ────────────────────────────────────────
        word_count = len(intro.split())
        if word_count < MIN_WORDS:
            skipped['too_short'] += 1
            continue

        # ── Collect ──────────────────────────────────────────────────────────
        results.append({
            'anthology_id': anthology_id,
            'year':         year,
            'venue':        venue,
            'introduction': intro,
            'word_count':   word_count,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nStreaming complete.")
    print(f"  Total streamed:  {streamed:,}")
    print(f"  Total collected: {len(results):,}")
    print(f"\nSkipped breakdown:")
    for reason, count in sorted(skipped.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {reason:<20}: {count:,}")

    if not results:
        print("ERROR: No papers collected. Check filters.")
        return

    # ── Assign prompt_ids and save ────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.insert(0, 'prompt_id', [f"ACL_{i+1:04d}" for i in range(len(df))])

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\nWord count stats:")
    print(f"  Min:    {df['word_count'].min()}")
    print(f"  Max:    {df['word_count'].max()}")
    print(f"  Mean:   {df['word_count'].mean():.0f}")
    print(f"  Median: {df['word_count'].median():.0f}")

    print(f"\nYear distribution:")
    print(df['year'].value_counts().sort_index().to_string())

    print(f"\nTop 15 venues:")
    print(df['venue'].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()