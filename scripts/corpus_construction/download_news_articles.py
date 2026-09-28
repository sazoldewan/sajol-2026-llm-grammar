import re
import pandas as pd
from datasets import load_dataset

dataset = load_dataset(
    "rjac/all-the-news-2-1-Component-one",
    split="train",
    streaming=True
)

# ── UI artifacts from web scraping ────────────────────────────────────────────
UI_PATTERNS = [
    r"Chevron icon\s*\n.*?navigation options\.",
    r"Close icon",
    r"Facebook icon",
    r"Twitter icon",
]

# ── Boilerplate line patterns to strip ───────────────────────────────────────
BOILERPLATE_LINE_PATTERNS = [
    r"(?i)^read the original article on .+",
    r"(?i)^follow .{0,40} on (facebook|twitter).*",
    r"(?i)^copyright \d{4}.*",
    r"(?i)^sign up for.{0,60}(newsletter|updates|emails).*",
    r"(?i)^subscribe to (our|the) .{0,40}(newsletter|updates).*",
    r"(?i)^this article originally appeared.*",
    r"(?i)^.{0,60}has affiliate partnership.*",
    r"(?i)^reprinted (with permission )?from.*",
    r"(?i)^this story originally (appeared|ran).*",
    r"(?i)^get the latest .{0,40}(news|updates) (in your inbox|delivered).*",
]

def clean_article(text):
    text = text.replace("\xa0", " ")

    for pattern in UI_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if any(re.match(p, stripped) for p in BOILERPLATE_LINE_PATTERNS):
            continue
        cleaned_lines.append(stripped)
    text = "\n".join(cleaned_lines)

    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def is_listicle(article):
    if re.search(r"(?i)has affiliate partnership", article):
        return True
    if re.search(r"(?i)insider picks team writes about", article):
        return True
    lines = article.split("\n")
    listicle_headers = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and len(stripped.split()) < 6:
            prev_blank = (i == 0) or not lines[i - 1].strip()
            next_blank = (i == len(lines) - 1) or not lines[i + 1].strip()
            if prev_blank and next_blank:
                listicle_headers += 1
    return listicle_headers >= 4

# ── Stream and collect ───────────────────────────────────────────────────────
papers = []
checked = 0

for record in dataset:
    checked += 1

    year = record.get("year")
    title = (record.get("title") or "").strip()
    raw_article = (record.get("article") or "").strip()

    # Year filter: 2021 or older
    if not year or int(year) > 2021:
        continue

    # Non-empty title and article
    if not title or not raw_article:
        continue

    # Title: 8–14 words
    if not (8 <= len(title.split()) <= 14):
        continue

    # Clean article first, then apply length and quality filters
    article = clean_article(raw_article)

    # Article length: 300–500 words
    word_count = len(article.split())
    if not (300 <= word_count <= 500):
        continue

    # Drop listicles and product lists
    if is_listicle(article):
        continue

    papers.append({
        "publication": record.get("publication"),
        "date": record.get("date"),
        "year": int(year),
        "title": title,
        "article": article,
        "author": record.get("author"),
        "url": record.get("url"),
        "word_count": word_count,
    })

    if len(papers) == 1000:
        break

df = pd.DataFrame(papers)
df.to_csv("news_articles_1000.csv", index=False)

print("Done!")
print(f"Checked:    {checked} records")
print(f"Saved:      {len(df)} articles")
print(f"Word count: min={df['word_count'].min()}, max={df['word_count'].max()}, mean={df['word_count'].mean():.0f}")
