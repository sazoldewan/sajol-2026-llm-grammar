"""
Statistical Analysis — Headline Grammar Features
MSc Thesis: Dewan Sajol, University of Bamberg

Analyses
--------
1. Cohen's d        — effect size per feature per checkpoint vs human baseline (H2)
2. Kruskal-Wallis   — non-parametric omnibus test per feature
   + Mann-Whitney U — pairwise follow-up with Bonferroni correction
                      H1: stage effects (base vs sft vs dpo within each family)
                      H2: each checkpoint vs human

Additional diagnostic outputs
------------------------------
- selected_feature_boxplots.png    — boxplots of the 8 features with the
                                      largest mean |Cohen's d| vs human
- avg_abs_d_by_checkpoint.png/.csv — mean |Cohen's d| per checkpoint vs human
- h1_stage_effects_heatmap.png/.csv — Cohen's d for base/SFT/DPO pairwise
                                      comparisons within each model family

h3_genre_comparison.png is NOT produced here — it requires both genres'
statistics together and will come from a separate combined script once
ACL's stat_analysis run exists.

RELEASE-COPY NOTE: Exploratory factor analysis (EFA) and PCA are not part of
the final thesis and have been removed from this release copy, including
their outputs (pca_scatter.png, factor loadings/scores, eigenvalues, scree plots,
correlation matrix). The only part of the former run_factor_analysis() that
another analysis depends on — the zero-variance feature exclusion used by
plot_selected_feature_boxplots() — is kept in get_boxplot_candidate_features().
All remaining CSV outputs were verified to be byte-identical to those of the
original (pre-removal) script run in the same environment; see docs/MANIFEST.md.

Input:   /Users/dewansajol/Study/Uni Bamberg/Thesis/data/features_headlines.csv
Outputs: /Users/dewansajol/Study/Uni Bamberg/Thesis/results/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_FILE = "/Users/dewansajol/Study/Uni Bamberg/Thesis/data/features_headlines.csv"
RESULTS_DIR   = "/Users/dewansajol/Study/Uni Bamberg/Thesis/results/result final/headlines/"

# ── Feature names ──────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'g1_noun_rate', 'g1_propn_rate', 'g1_verb_rate', 'g1_adj_rate',
    'g1_adv_rate', 'g1_pron_rate', 'g1_adp_rate', 'g1_nominalization_rate',
    'g2_nsubj_rate', 'g2_amod_rate', 'g2_nmod_rate', 'g2_compound_rate',
    'g2_ccomp_rate', 'g2_xcomp_rate', 'g2_advcl_rate', 'g2_acl_rate',
    'g2_acl_relcl_rate',
    'g3_agentless_passive_rate', 'g3_by_passive_rate',
    'g3_progressive_rate', 'g3_part_clause_rate',
    'g4_present_tense_rate', 'g4_past_tense_rate', 'g4_perfect_aspect_rate',
    'g5_necessity_modal_rate', 'g5_possibility_modal_rate', 'g5_predictive_modal_rate',
    'g6_mean_sent_len', 'g6_mean_dep_dist', 'g6_mean_tree_depth',
    'g6_finite_sub_rate', 'g6_nonfinite_sub_rate', 'g6_noun_verb_ratio',
    'g7_first_person_rate', 'g7_second_person_rate', 'g7_exist_there_rate',
]

# g6_noun_verb_ratio excluded: 14.6% NaN (verbless headlines)
EXCLUDE_FEATURES = ['g6_noun_verb_ratio']
ALPHA            = 0.05

GROUP_ORDER = [
    'human',
    'mistral_base', 'mistral_sft', 'mistral_dpo',
    'llama_base',   'llama_sft',   'llama_dpo',
    'olmo2_base',   'olmo2_sft',   'olmo2_dpo',
]

MODEL_FAMILIES = {
    'Mistral': ['mistral_base', 'mistral_sft', 'mistral_dpo'],
    'Llama':   ['llama_base',   'llama_sft',   'llama_dpo'],
    'OLMo2':   ['olmo2_base',   'olmo2_sft',   'olmo2_dpo'],
}

STAGE_COLORS = {'human': '#9E9E9E', 'base': '#2196F3', 'sft': '#FF9800', 'dpo': '#4CAF50'}


def stage_of_group(group):
    """'mistral_base' -> 'base'; 'human' -> 'human'."""
    return 'human' if group == 'human' else group.rsplit('_', 1)[-1]


# ── Helpers ────────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Cohen's d with pooled SD (positive = a > b)."""
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


def to_csv(df, name):
    path = os.path.join(RESULTS_DIR, name)
    df.to_csv(path)
    print(f"  Saved: {name}")


# ── 0. Load data ───────────────────────────────────────────────────────────────

def load_data():
    print("Loading features...")
    df = pd.read_csv(FEATURES_FILE)
    df['group'] = df.apply(
        lambda r: 'human' if r['source'] == 'human'
                  else f"{r['model']}_{r['stage']}",
        axis=1
    )
    feat_cols = [c for c in FEATURE_NAMES if c not in EXCLUDE_FEATURES]
    print(f"  Rows: {len(df):,}  |  Feature columns: {len(feat_cols)}")
    print(f"  Groups: {sorted(df['group'].unique())}")
    return df, feat_cols


# ── 1. Feature list for selected_feature_boxplots.png ─────────────────────────
# RELEASE-COPY NOTE: this used to be run_factor_analysis(), which also ran the
# full EFA pipeline (Bartlett's test, KMO, Principal Axis Factoring, varimax
# rotation, factor scores, and their CSV/figure outputs) plus a standalone PCA
# scatter analysis. EFA and PCA are not part of the final thesis, so all of
# that has been removed from this release copy. The only part of the original
# function that anything else in this script depends on is the zero-variance
# feature exclusion below: plot_selected_feature_boxplots() uses the resulting
# fa_feat_cols list to restrict its top-8-by-|Cohen's d| feature selection.
# That dependency is preserved here; everything else was deleted.

def get_boxplot_candidate_features(df, feat_cols):
    print("\n── Feature list (zero-variance exclusion) ─────────────────────")

    fa_df = df[['group', 'prompt_id'] + feat_cols].dropna()
    X_raw = fa_df[feat_cols].values
    print(f"  Complete cases: {len(fa_df):,} / {len(df):,}")

    # Drop any zero-variance features. Constant columns are degenerate for
    # any variance-based feature ranking. Features are detected dynamically
    # from the current feature file.
    stds = X_raw.std(axis=0)
    valid_mask = stds > 1e-10
    n_excluded = int((~valid_mask).sum())
    excl = [feat_cols[i] for i in range(len(feat_cols)) if not valid_mask[i]]
    if n_excluded:
        print(f"  Excluding {n_excluded} zero-variance feature(s): {excl}")
    fa_feat_cols = [feat_cols[i] for i in range(len(feat_cols)) if valid_mask[i]]

    return fa_feat_cols


# ── 2. Cohen's d (H2: each checkpoint vs human) ───────────────────────────────

def run_cohens_d(df, feat_cols):
    print("\n── Cohen's d (checkpoints vs human) ─────────────────────────")

    human = df[df['group'] == 'human']
    model_groups = [g for g in GROUP_ORDER if g != 'human']

    records = {}
    for g in model_groups:
        grp = df[df['group'] == g]
        records[g] = {
            feat: cohens_d(grp[feat].dropna().values, human[feat].dropna().values)
            for feat in feat_cols
        }

    d_matrix = pd.DataFrame(records, index=feat_cols)
    to_csv(d_matrix, 'cohens_d_matrix.csv')

    vmax = max(d_matrix.abs().max().max(), 0.5)
    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(
        d_matrix,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        linecolor='#eeeeee',
        annot=True,
        fmt='.2f',
        annot_kws={'size': 6},
        xticklabels=[g.replace('_', '\n') for g in model_groups],
        yticklabels=feat_cols,
        cbar_kws={'label': "Cohen's d  (positive = model > human)"},
    )
    ax.set_title("Cohen's d — Model Checkpoints vs Human Baseline", fontsize=13, pad=12)
    ax.set_xlabel('Model checkpoint', fontsize=10)
    ax.set_ylabel('Feature', fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=7)
    fig.tight_layout()
    save(fig, 'cohens_d_heatmap.png')

    small_plus  = d_matrix[d_matrix.abs().gt(0.2).any(axis=1)]
    medium_plus = d_matrix[d_matrix.abs().gt(0.5).any(axis=1)]
    print(f"  Features with |d| > 0.2 (small)  in ≥1 group: {len(small_plus)} / {len(feat_cols)}")
    print(f"  Features with |d| > 0.5 (medium) in ≥1 group: {len(medium_plus)} / {len(feat_cols)}")
    # Print top-5 effect sizes for quick inspection
    flat = [(feat, g, d_matrix.loc[feat, g]) for feat in d_matrix.index for g in d_matrix.columns]
    flat.sort(key=lambda x: abs(x[2]), reverse=True)
    print("  Top-5 |d| (model vs human):")
    for feat, g, d in flat[:5]:
        print(f"    {g:<15} {feat:<35} d={d:+.4f}")
    return d_matrix


# ── 2b. Selected-feature boxplots ─────────────────────────────────────────────

def plot_selected_feature_boxplots(df, d_matrix, fa_feat_cols):
    print("\n── Selected feature boxplots ─────────────────────────────────")

    # Restrict candidates to the final analyzed feature list (post-exclusions,
    # i.e. after the zero-variance exclusion in get_boxplot_candidate_features()).
    candidate_feats = [f for f in d_matrix.index if f in fa_feat_cols]
    mean_abs_d = d_matrix.loc[candidate_feats].abs().mean(axis=1).sort_values(ascending=False)
    selected = mean_abs_d.head(8).index.tolist()
    print(f"  Selected 8 features (mean |Cohen's d| vs human): {selected}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for i, feat in enumerate(selected):
        ax = axes[i]
        data = [df[df['group'] == g][feat].dropna().values for g in GROUP_ORDER]
        colors = [STAGE_COLORS[stage_of_group(g)] for g in GROUP_ORDER]

        bp = ax.boxplot(data, patch_artist=True,
                         medianprops={'color': 'black', 'linewidth': 1.2})
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks(range(1, len(GROUP_ORDER) + 1))
        ax.set_xticklabels(GROUP_ORDER, rotation=90, fontsize=7)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_ylabel('Value', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)

    legend_handles = [
        Patch(facecolor=STAGE_COLORS[s], label='Human' if s == 'human' else s.upper())
        for s in ['human', 'base', 'sft', 'dpo']
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 1.04))

    fig.suptitle("Top 8 Features by Mean |Cohen's d| vs Human", fontsize=13, y=1.08)
    fig.tight_layout()
    save(fig, 'selected_feature_boxplots.png')


# ── 2c. Mean |Cohen's d| by checkpoint ─────────────────────────────────────────

def plot_avg_abs_d_by_checkpoint(d_matrix):
    print("\n── Mean |Cohen's d| by checkpoint ────────────────────────────")

    model_groups = [g for g in GROUP_ORDER if g != 'human']
    mean_abs = d_matrix[model_groups].abs().mean(axis=0).sort_values()  # closest -> furthest

    out = mean_abs.rename('mean_abs_cohens_d').to_frame()
    out.index.name = 'checkpoint'
    to_csv(out, 'avg_abs_d_by_checkpoint.csv')

    colors = [STAGE_COLORS[stage_of_group(g)] for g in mean_abs.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(mean_abs.index, mean_abs.values, color=colors)
    ax.set_ylabel("Mean |Cohen's d| vs human", fontsize=10)
    ax.set_title("Mean Absolute Cohen's d by Checkpoint\n(ordered closest → furthest from human)",
                 fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    legend_handles = [Patch(facecolor=STAGE_COLORS[s], label=s.upper()) for s in ['base', 'sft', 'dpo']]
    ax.legend(handles=legend_handles, fontsize=9)

    fig.tight_layout()
    save(fig, 'avg_abs_d_by_checkpoint.png')


# ── 2d. H1 stage-effects heatmap (Cohen's d, within each family) ──────────────

def run_h1_stage_effects_heatmap(df, feat_cols):
    print("\n── H1 stage-effects heatmap (Cohen's d) ──────────────────────")

    groups_data = {g: df[df['group'] == g] for g in GROUP_ORDER}
    comparisons = [('base', 'sft'), ('base', 'dpo'), ('sft', 'dpo')]

    # Cohen's d direction: positive = first group in the comparison has a
    # higher feature value than the second group (matches cohens_d(a, b)'s
    # own convention, positive = a > b).
    records = {}
    for family, fgroups in MODEL_FAMILIES.items():
        stage_to_group = {g.rsplit('_', 1)[-1]: g for g in fgroups}
        for s1, s2 in comparisons:
            col_name = f'{family}: {s1} vs {s2}'
            a_df, b_df = groups_data[stage_to_group[s1]], groups_data[stage_to_group[s2]]
            records[col_name] = {
                feat: cohens_d(a_df[feat].dropna().values, b_df[feat].dropna().values)
                for feat in feat_cols
            }

    h1_matrix = pd.DataFrame(records, index=feat_cols)
    to_csv(h1_matrix, 'h1_stage_effects_cohens_d.csv')

    vmax = max(h1_matrix.abs().max().max(), 0.5)
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(
        h1_matrix,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        linecolor='#eeeeee',
        annot=True,
        fmt='.2f',
        annot_kws={'size': 6},
        cbar_kws={'label': "Cohen's d (positive = first stage > second stage)"},
    )
    ax.set_title("H1: Stage-Effect Cohen's d Within Each Model Family", fontsize=13, pad=12)
    ax.set_xlabel('Comparison', fontsize=10)
    ax.set_ylabel('Feature', fontsize=10)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=7)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    fig.tight_layout()
    save(fig, 'h1_stage_effects_heatmap.png')

    return h1_matrix


# ── 4. Kruskal-Wallis + Mann-Whitney ──────────────────────────────────────────

def run_statistics(df, feat_cols):
    print("\n── Kruskal-Wallis + Mann-Whitney U ───────────────────────────")

    groups_data = {g: df[df['group'] == g] for g in GROUP_ORDER}

    # Omnibus Kruskal-Wallis
    kw_records = []
    sig_features = []
    for feat in feat_cols:
        samples = [groups_data[g][feat].dropna().values for g in GROUP_ORDER
                   if len(groups_data[g][feat].dropna()) >= 3]
        if len(samples) < 2:
            continue
        try:
            stat, p = kruskal(*samples)
            kw_records.append({'feature': feat, 'H': stat, 'p_value': p,
                                'significant': p < ALPHA})
            if p < ALPHA:
                sig_features.append(feat)
        except Exception:
            pass

    kw_df = pd.DataFrame(kw_records).set_index('feature')
    to_csv(kw_df, 'kruskal_wallis_results.csv')
    print(f"  Kruskal-Wallis significant: {len(sig_features)} / {len(feat_cols)} features (α={ALPHA})")

    # H1: pairwise stage effects within each model family
    h1_records = []
    for family, fgroups in MODEL_FAMILIES.items():
        pairs = list(combinations(fgroups, 2))
        n_tests = len(pairs) * len(sig_features)
        bonf = ALPHA / n_tests if n_tests > 0 else ALPHA
        for feat in sig_features:
            for g1, g2 in pairs:
                a = groups_data[g1][feat].dropna().values
                b = groups_data[g2][feat].dropna().values
                if len(a) < 3 or len(b) < 3:
                    continue
                stat, p = mannwhitneyu(a, b, alternative='two-sided')
                h1_records.append({
                    'family': family, 'group_1': g1, 'group_2': g2, 'feature': feat,
                    'U': stat, 'p_value': p,
                    'p_bonferroni': min(p * n_tests, 1.0),
                    'significant': p < bonf,
                    'cohens_d': cohens_d(a, b),
                })

    h1_df = pd.DataFrame(h1_records)
    if not h1_df.empty:
        to_csv(h1_df.set_index(['family', 'group_1', 'group_2', 'feature']),
               'mannwhitney_h1_stage_effects.csv')
        print(f"  H1 significant (Bonferroni): {h1_df['significant'].sum()} / {len(h1_df)}")

    # H2: each checkpoint vs human
    human_data = groups_data['human']
    model_groups = [g for g in GROUP_ORDER if g != 'human']
    n_tests_h2 = len(model_groups) * len(sig_features)
    bonf_h2 = ALPHA / n_tests_h2 if n_tests_h2 > 0 else ALPHA

    h2_records = []
    for feat in sig_features:
        hum = human_data[feat].dropna().values
        for g in model_groups:
            mod = groups_data[g][feat].dropna().values
            if len(hum) < 3 or len(mod) < 3:
                continue
            stat, p = mannwhitneyu(hum, mod, alternative='two-sided')
            h2_records.append({
                'checkpoint': g, 'feature': feat,
                'U': stat, 'p_value': p,
                'p_bonferroni': min(p * n_tests_h2, 1.0),
                'significant': p < bonf_h2,
                'cohens_d_vs_human': cohens_d(mod, hum),
            })

    h2_df = pd.DataFrame(h2_records)
    if not h2_df.empty:
        to_csv(h2_df.set_index(['checkpoint', 'feature']),
               'mannwhitney_h2_vs_human.csv')
        print(f"  H2 significant (Bonferroni): {h2_df['significant'].sum()} / {len(h2_df)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results → {RESULTS_DIR}\n")

    df, feat_cols = load_data()
    fa_feat_cols = get_boxplot_candidate_features(df, feat_cols)
    d_matrix = run_cohens_d(df, feat_cols)
    plot_selected_feature_boxplots(df, d_matrix, fa_feat_cols)
    plot_avg_abs_d_by_checkpoint(d_matrix)
    run_h1_stage_effects_heatmap(df, feat_cols)
    run_statistics(df, feat_cols)

    # NOTE: h3_genre_comparison.png (cross-genre comparison of training-stage
    # effects, headlines vs. ACL introductions) is intentionally NOT produced
    # here — it needs both genres' statistics together, and will be generated
    # by a separate combined script once ACL's stat_analysis run exists.

    print("\n── Complete ───────────────────────────────────────────────────")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        print(f"  {f:<50} {size:>10,} bytes")


if __name__ == "__main__":
    main()
