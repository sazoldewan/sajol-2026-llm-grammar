"""
H3 Genre-Comparison Figures — Headlines vs. ACL Introductions
MA Thesis: Dewan Sajol, University of Bamberg

Combines the already-computed per-genre statistics (headlines' full
base/SFT/DPO analysis, ACL's SFT/DPO-only analysis) into two cross-genre
comparison figures for H3 (genre moderates the magnitude of training-stage
effects).

IMPORTANT: the ACL primary analysis contains only SFT and DPO checkpoints
(base ACL outputs were excluded — see stat_analysis_acl.py docstring, since
base used completion-style prompting and is not comparable to SFT/DPO's
instruction-style prompting). No ACL base values are created or imputed
anywhere in this script; the ACL panel/bars simply have fewer categories
than the headline ones.

Input:
  Headlines: results/avg_abs_d_by_checkpoint.csv
             results/h1_stage_effects_cohens_d.csv
  ACL:       results/acl/avg_abs_d_by_checkpoint_acl.csv
             results/acl/h1_stage_effects_cohens_d_acl.csv

Output (results/result final/cross_genre/):
  h3_genre_comparison.png / .csv
  h3_sft_dpo_stage_effect_by_genre.png / .csv

This script only creates these four new files — it does not modify or
regenerate any existing per-genre result file.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = "/Users/dewansajol/Study/Uni Bamberg/Thesis/results/result final/headlines/"
RESULTS_ACL_DIR = "/Users/dewansajol/Study/Uni Bamberg/Thesis/results/result final/acl/"
OUTPUT_DIR = "/Users/dewansajol/Study/Uni Bamberg/Thesis/results/result final/cross_genre/"

HEADLINE_AVG_D_FILE = os.path.join(RESULTS_DIR, "avg_abs_d_by_checkpoint.csv")
HEADLINE_H1_FILE    = os.path.join(RESULTS_DIR, "h1_stage_effects_cohens_d.csv")
ACL_AVG_D_FILE       = os.path.join(RESULTS_ACL_DIR, "avg_abs_d_by_checkpoint_acl.csv")
ACL_H1_FILE          = os.path.join(RESULTS_ACL_DIR, "h1_stage_effects_cohens_d_acl.csv")

OUT_FIG1 = os.path.join(OUTPUT_DIR, "h3_genre_comparison.png")
OUT_CSV1 = os.path.join(OUTPUT_DIR, "h3_genre_comparison_data.csv")
OUT_FIG2 = os.path.join(OUTPUT_DIR, "h3_sft_dpo_stage_effect_by_genre.png")
OUT_CSV2 = os.path.join(OUTPUT_DIR, "h3_sft_dpo_stage_effect_by_genre_data.csv")

STAGE_COLORS = {'base': '#2196F3', 'sft': '#FF9800', 'dpo': '#4CAF50'}
STAGE_ORDER  = ['base', 'sft', 'dpo']
FAMILY_ORDER = ['Mistral', 'Llama', 'OLMo2']

# Display-only labels for the compact two-line x-tick format
# (e.g. "Mistral\nBASE") — purely cosmetic, not used in any data processing.
FAMILY_DISPLAY = {'mistral': 'Mistral', 'llama': 'Llama', 'olmo2': 'OLMo2'}
STAGE_DISPLAY  = {'base': 'BASE', 'sft': 'SFT', 'dpo': 'DPO'}


def stage_of_checkpoint(checkpoint):
    return checkpoint.rsplit('_', 1)[-1]


def family_of_checkpoint(checkpoint):
    return checkpoint.rsplit('_', 1)[0]


def checkpoint_display_label(checkpoint):
    family = FAMILY_DISPLAY.get(family_of_checkpoint(checkpoint), family_of_checkpoint(checkpoint))
    stage = STAGE_DISPLAY.get(stage_of_checkpoint(checkpoint), stage_of_checkpoint(checkpoint))
    return f"{family}\n{stage}"


# ── 1. h3_genre_comparison.png ──────────────────────────────────────────────────

def build_genre_comparison():
    print("── h3_genre_comparison ─────────────────────────────────────────")

    hl = pd.read_csv(HEADLINE_AVG_D_FILE)
    acl = pd.read_csv(ACL_AVG_D_FILE)

    hl['genre'] = 'News Headlines'
    acl['genre'] = 'ACL Introductions'

    combined = pd.concat([hl, acl], ignore_index=True)
    combined['stage'] = combined['checkpoint'].apply(stage_of_checkpoint)
    combined['family'] = combined['checkpoint'].apply(family_of_checkpoint)
    combined = combined.rename(columns={'mean_abs_cohens_d': 'mean_abs_d'})

    n_headline = len(hl)
    n_acl = len(acl)

    # Sort each panel by family (Mistral, Llama, OLMo2) then stage (base, sft, dpo)
    # 'family' holds lowercase checkpoint-derived names (e.g. 'mistral'), while
    # FAMILY_ORDER holds display-cased names (e.g. 'Mistral') — map through
    # FAMILY_DISPLAY first so the two actually match.
    combined['_family_rank'] = combined['family'].apply(
        lambda f: FAMILY_ORDER.index(FAMILY_DISPLAY.get(f, f))
        if FAMILY_DISPLAY.get(f, f) in FAMILY_ORDER else len(FAMILY_ORDER))
    combined['_stage_rank'] = combined['stage'].apply(
        lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else len(STAGE_ORDER))
    combined = combined.sort_values(['genre', '_family_rank', '_stage_rank']).reset_index(drop=True)

    # Save plotted data (required columns only)
    out_df = combined[['genre', 'checkpoint', 'stage', 'mean_abs_d']]
    out_df.to_csv(OUT_CSV1, index=False)
    print(f"  Saved: {OUT_CSV1}")

    y_max = combined['mean_abs_d'].max() * 1.1

    # sharey=False (default): each panel keeps its own visible tick labels.
    # The identical scale across both panels is enforced explicitly below via
    # the same y_max on both axes, rather than relying on sharey (which hides
    # the second panel's tick labels and makes it look unscaled).
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))

    for ax, genre in zip(axes, ['News Headlines', 'ACL Introductions']):
        sub = combined[combined['genre'] == genre]
        colors = [STAGE_COLORS[s] for s in sub['stage']]

        # Visually group bars by family: bars within a family sit flush next
        # to each other, with a wider gap inserted between families. Only one
        # tick (the family name) is placed at the centre of each group — this
        # is purely a bar-position/tick-label change, the underlying
        # mean_abs_d values, their order, and the colours are untouched.
        bar_width = 0.8
        inter_family_gap = 0.6
        positions = []
        x_cursor = 0.0
        prev_family = None
        for fam in sub['family']:
            if prev_family is not None and fam != prev_family:
                x_cursor += inter_family_gap
            positions.append(x_cursor)
            x_cursor += 1.0
            prev_family = fam

        ax.bar(positions, sub['mean_abs_d'], width=bar_width, color=colors)
        ax.set_title(genre, fontsize=11, fontweight='bold')
        ax.set_xlabel('Model family', fontsize=10)

        # Centre one tick per family group over that family's bars.
        family_order_seen = list(dict.fromkeys(sub['family']))
        tick_pos, tick_labels = [], []
        for fam in family_order_seen:
            fam_positions = [p for p, f in zip(positions, sub['family']) if f == fam]
            tick_pos.append(sum(fam_positions) / len(fam_positions))
            tick_labels.append(FAMILY_DISPLAY.get(fam, fam))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=9)
        ax.set_xlim(positions[0] - bar_width, positions[-1] + bar_width)

        ax.set_ylim(0, y_max)
        ax.tick_params(axis='y', labelleft=True, labelsize=9)

    axes[0].set_ylabel("Mean |Cohen's d| vs human", fontsize=10)
    axes[1].set_ylabel("Mean |Cohen's d| vs human", fontsize=10)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=STAGE_COLORS[s], label=s.upper()) for s in STAGE_ORDER]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle('Mean grammatical distance from human baseline across genres',
                 fontsize=12, y=1.06)

    fig.tight_layout()
    fig.savefig(OUT_FIG1, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUT_FIG1}")

    return n_headline, n_acl, hl['mean_abs_cohens_d'].mean(), acl['mean_abs_cohens_d'].mean()


# ── 2. h3_sft_dpo_stage_effect_by_genre.png ─────────────────────────────────────

def build_sft_dpo_stage_effect():
    print("\n── h3_sft_dpo_stage_effect_by_genre ──────────────────────────")

    hl_h1 = pd.read_csv(HEADLINE_H1_FILE, index_col=0)
    acl_h1 = pd.read_csv(ACL_H1_FILE, index_col=0)

    records = []
    for genre, h1_df in [('News Headlines', hl_h1), ('ACL Introductions', acl_h1)]:
        for family in FAMILY_ORDER:
            col = f'{family}: sft vs dpo'
            if col not in h1_df.columns:
                continue
            mean_abs_d = h1_df[col].abs().mean()
            records.append({
                'genre': genre,
                'family': family,
                'comparison': 'sft vs dpo',
                'mean_abs_d': mean_abs_d,
            })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_CSV2, index=False)
    print(f"  Saved: {OUT_CSV2}")

    hl_mean = out_df[out_df['genre'] == 'News Headlines']['mean_abs_d'].mean()
    acl_mean = out_df[out_df['genre'] == 'ACL Introductions']['mean_abs_d'].mean()

    # Grouped bar chart: x = family, bars = genre
    genres = ['News Headlines', 'ACL Introductions']
    genre_colors = {'News Headlines': '#2196F3', 'ACL Introductions': '#FF9800'}
    x = range(len(FAMILY_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for i, genre in enumerate(genres):
        vals = [
            out_df[(out_df['family'] == fam) & (out_df['genre'] == genre)]['mean_abs_d'].values
            for fam in FAMILY_ORDER
        ]
        vals = [v[0] if len(v) else 0 for v in vals]
        offset = (i - 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width=width,
               label=genre, color=genre_colors[genre])

    ax.set_xticks(list(x))
    ax.set_xticklabels(FAMILY_ORDER, fontsize=10)
    ax.set_xlabel('Model family', fontsize=10)
    ax.set_ylabel("Mean |Cohen's d| (SFT vs DPO)", fontsize=10)
    ax.set_title('SFT--DPO grammatical stage effect across genres', fontsize=12)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_FIG2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUT_FIG2}")

    return hl_mean, acl_mean


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_headline, n_acl, hl_avg_d, acl_avg_d = build_genre_comparison()
    hl_sft_dpo, acl_sft_dpo = build_sft_dpo_stage_effect()

    print("\n── Summary ────────────────────────────────────────────────────")
    print(f"  Headline checkpoints plotted:        {n_headline}")
    print(f"  ACL checkpoints plotted:              {n_acl}")
    print(f"  Mean human-distance (headlines):      {hl_avg_d:.4f}")
    print(f"  Mean human-distance (ACL):             {acl_avg_d:.4f}")
    print(f"  Mean SFT-vs-DPO effect (headlines):   {hl_sft_dpo:.4f}")
    print(f"  Mean SFT-vs-DPO effect (ACL):          {acl_sft_dpo:.4f}")
    print(f"\n  Output files:")
    for p in [OUT_FIG1, OUT_CSV1, OUT_FIG2, OUT_CSV2]:
        print(f"    {p}")


if __name__ == "__main__":
    main()
