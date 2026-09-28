"""
Grammatical Feature Extraction — 36 Features
MA Thesis: Dewan Sajol
University of Bamberg

Extracts 36 grammatical features from master_headlines.csv using Stanza.
Input:  master_headlines.csv  (10,000 rows)
Output: features_headlines.csv (10,000 rows, 36 feature columns)

Feature groups:
  G1 — POS-Based Features           (8 features)
  G2 — Dependency Relation Features  (9 features, g2_nsubj_pass_rate dropped)
  G3 — Voice and Aspect Features     (4 features)
  G4 — Tense Features                (3 features)
  G5 — Modal Verb Features           (3 features)
  G6 — Structural Complexity         (6 features)
  G7 — Stance and Interaction        (3 features)

All rate features are normalized per 1,000 tokens.

Notes:
  - g6_noun_verb_ratio returns None for verbless headlines (e.g. "Senate Vote on Tax Bill").
    These are retained as None in the output and treated as a separate category in analysis,
    as verbless constructions are themselves a diagnostic compression feature of the headline genre.
  - Passive counting (G3) operates at sentence level: one passive event per sentence,
    regardless of how many passive constructions appear in that sentence. For headlines
    (typically single sentences) this is not a limitation. Revisit for ACL introductions.
  - acl:relcl (finite relative clauses) is intentionally excluded from g6_nonfinite_sub_rate
    because relative clauses are finite, not non-finite. They are captured separately
    in g2_acl_relcl_rate only.
  - Per-1,000 token normalization produces high variance for short texts (8-10 words).
    This is mathematically correct and standard in corpus linguistics, but should be
    noted as a limitation when reporting results for the headline genre.
"""

import pandas as pd
import stanza
import os
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
MASTER_FILE      = "/Users/dewansajol/Study/Uni Bamberg/Thesis/data/master_headlines.csv"
OUTPUT_FILE      = "/Users/dewansajol/Study/Uni Bamberg/Thesis/data/features_headlines.csv"
CHECKPOINT       = "/Users/dewansajol/Study/Uni Bamberg/Thesis/data/features_headlines_checkpoint.csv"
CHECKPOINT_EVERY = 500

# ── Stanza pipeline ────────────────────────────────────────────────────────────
print("Loading Stanza pipeline...")
nlp = stanza.Pipeline(
    'en',
    processors='tokenize,pos,lemma,depparse',
    verbose=False,
    use_gpu=False
)
print("Stanza ready.\n")

# ── Vocabulary sets ────────────────────────────────────────────────────────────

# Nominalization suffixes
NOMINALIZATION_SUFFIXES = ('tion', 'ment', 'ness', 'ity')

# Modal verb lemmas
NECESSITY_MODALS    = {'must', 'should', 'ought', 'need'}
POSSIBILITY_MODALS  = {'can', 'may', 'could', 'might'}
PREDICTIVE_MODALS   = {'will', 'would', 'shall'}

# First and second person pronouns (lemma-based; Stanza preserves surface lemmas
# for possessives e.g. my -> my, so these are included explicitly)
FIRST_PERSON_PRONOUNS  = {'i', 'we', 'me', 'us', 'my', 'our', 'myself', 'ourselves'}
SECOND_PERSON_PRONOUNS = {'you', 'your', 'yourself', 'yourselves'}


# ── Feature name list ──────────────────────────────────────────────────────────
FEATURE_NAMES = [
    # G1 — POS-Based (8)
    'g1_noun_rate', 'g1_propn_rate', 'g1_verb_rate', 'g1_adj_rate',
    'g1_adv_rate', 'g1_pron_rate', 'g1_adp_rate', 'g1_nominalization_rate',
    # G2 — Dependency Relations (9)
    'g2_nsubj_rate', 'g2_amod_rate', 'g2_nmod_rate', 'g2_compound_rate',
    'g2_ccomp_rate', 'g2_xcomp_rate', 'g2_advcl_rate', 'g2_acl_rate',
    'g2_acl_relcl_rate',
    # G3 — Voice and Aspect (4)
    'g3_agentless_passive_rate', 'g3_by_passive_rate',
    'g3_progressive_rate', 'g3_part_clause_rate',
    # G4 — Tense (3)
    'g4_present_tense_rate', 'g4_past_tense_rate', 'g4_perfect_aspect_rate',
    # G5 — Modal Verbs (3)
    'g5_necessity_modal_rate', 'g5_possibility_modal_rate', 'g5_predictive_modal_rate',
    # G6 — Structural Complexity (6)
    'g6_mean_sent_len', 'g6_mean_dep_dist', 'g6_mean_tree_depth',
    'g6_finite_sub_rate', 'g6_nonfinite_sub_rate', 'g6_noun_verb_ratio',
    # G7 — Stance and Interaction (3)
    'g7_first_person_rate', 'g7_second_person_rate', 'g7_exist_there_rate',
]

assert len(FEATURE_NAMES) == 36, f"Expected 36 features, got {len(FEATURE_NAMES)}"


# ── Feature extraction function ────────────────────────────────────────────────

def extract_features(text):
    """
    Parse text with Stanza and extract all 36 grammatical features.
    Returns a dict of feature_name -> value.
    All rate features normalized per 1,000 tokens.
    g6_noun_verb_ratio returns None for verbless texts.
    """
    if not isinstance(text, str) or not text.strip():
        return {f: None for f in FEATURE_NAMES}

    doc = nlp(text.strip())
    all_words = [w for sent in doc.sentences for w in sent.words]
    n_tokens = len(all_words)

    if n_tokens == 0:
        return {f: None for f in FEATURE_NAMES}

    per_1k = 1000.0 / n_tokens

    # ── G1: POS-Based Features ─────────────────────────────────────────────────
    noun_count            = sum(1 for w in all_words if w.upos == 'NOUN')
    propn_count           = sum(1 for w in all_words if w.upos == 'PROPN')
    verb_count            = sum(1 for w in all_words if w.upos == 'VERB')
    adj_count             = sum(1 for w in all_words if w.upos == 'ADJ')
    adv_count             = sum(1 for w in all_words if w.upos == 'ADV')
    pron_count            = sum(1 for w in all_words if w.upos == 'PRON')
    adp_count             = sum(1 for w in all_words if w.upos == 'ADP')
    nominalization_count  = sum(
        1 for w in all_words
        if w.upos == 'NOUN' and w.text.lower().endswith(NOMINALIZATION_SUFFIXES)
    )

    # ── G2: Dependency Relation Features ──────────────────────────────────────
    deprel_counts = defaultdict(int)
    for w in all_words:
        deprel_counts[w.deprel] += 1

    nsubj_count    = deprel_counts.get('nsubj', 0)
    amod_count     = deprel_counts.get('amod', 0)
    nmod_count     = deprel_counts.get('nmod', 0) + deprel_counts.get('nmod:poss', 0)
    compound_count = deprel_counts.get('compound', 0)
    ccomp_count    = deprel_counts.get('ccomp', 0)
    xcomp_count    = deprel_counts.get('xcomp', 0)
    advcl_count    = deprel_counts.get('advcl', 0)
    acl_count      = deprel_counts.get('acl', 0)        # non-finite only
    relcl_count    = deprel_counts.get('acl:relcl', 0)  # finite relative clauses

    # ── G3: Voice and Aspect Features ─────────────────────────────────────────
    # NOTE: passive counting is per sentence (one event per sentence regardless
    # of how many passive constructions appear). Fine for headlines (single
    # sentences). Revisit this logic for ACL introductions (multi-sentence texts).
    agentless_passive_count = 0
    by_passive_count        = 0
    progressive_count       = 0
    part_clause_count       = 0

    for sent in doc.sentences:
        words = sent.words

        has_nsubj_pass = any(w.deprel == 'nsubj:pass' for w in words)
        has_obl_agent  = any(w.deprel == 'obl:agent'  for w in words)

        if has_nsubj_pass:
            if has_obl_agent:
                by_passive_count += 1
            else:
                agentless_passive_count += 1

        # Progressive: VERB with Tense=Pres|VerbForm=Part that has an aux child
        # with lemma 'be'. Stanza's English tagger never produces an
        # Aspect=Prog feature, so progressive must be detected structurally
        # via the auxiliary "be" attached to the present participle, not via
        # a feats lookup.
        for w in words:
            if w.upos == 'VERB' and w.feats and 'VerbForm=Part' in w.feats and 'Tense=Pres' in w.feats:
                has_be_aux = any(
                    x.head == w.id and x.deprel == 'aux' and x.lemma.lower() == 'be'
                    for x in words
                )
                if has_be_aux:
                    progressive_count += 1

        # Present participial clauses: VERB with VerbForm=Part, Tense=Pres
        # attached via advcl or acl (non-finite adverbial or adnominal clauses)
        for w in words:
            if w.upos == 'VERB' and w.feats and w.deprel in ('advcl', 'acl'):
                if 'VerbForm=Part' in w.feats and 'Tense=Pres' in w.feats:
                    part_clause_count += 1

    # ── G4: Tense Features ────────────────────────────────────────────────────
    present_tense_count = sum(
        1 for w in all_words
        if w.upos == 'VERB' and w.feats
        and 'Tense=Pres' in w.feats and 'VerbForm=Fin' in w.feats
    )
    past_tense_count = sum(
        1 for w in all_words
        if w.upos == 'VERB' and w.feats
        and 'Tense=Past' in w.feats and 'VerbForm=Fin' in w.feats
    )

    # Perfect aspect: AUX lemma 'have' whose head VERB has VerbForm=Part + Tense=Past
    perfect_count = 0
    for sent in doc.sentences:
        for w in sent.words:
            if w.upos == 'AUX' and w.lemma == 'have':
                head_words = [x for x in sent.words if x.id == w.head]
                if head_words:
                    hw = head_words[0]
                    if hw.feats and 'VerbForm=Part' in hw.feats and 'Tense=Past' in hw.feats:
                        perfect_count += 1

    # ── G5: Modal Verb Features ───────────────────────────────────────────────
    necessity_modal_count   = sum(
        1 for w in all_words if w.upos == 'AUX' and w.lemma in NECESSITY_MODALS
    )
    possibility_modal_count = sum(
        1 for w in all_words if w.upos == 'AUX' and w.lemma in POSSIBILITY_MODALS
    )
    predictive_modal_count  = sum(
        1 for w in all_words if w.upos == 'AUX' and w.lemma in PREDICTIVE_MODALS
    )

    # ── G6: Structural Complexity ─────────────────────────────────────────────
    sent_lengths  = [len(sent.words) for sent in doc.sentences]
    mean_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0

    # Mean dependency distance: |word.id - word.head| excluding root
    dep_distances = [abs(w.id - w.head) for w in all_words if w.head != 0]
    mean_dep_dist = sum(dep_distances) / len(dep_distances) if dep_distances else 0

    # Mean dependency tree depth per sentence
    def tree_depth(sent):
        children = defaultdict(list)
        for w in sent.words:
            children[w.head].append(w.id)
        root_ids = [w.id for w in sent.words if w.deprel == 'root']
        if not root_ids:
            return 0
        def depth(node_id, d=0):
            if not children[node_id]:
                return d
            return max(depth(c, d + 1) for c in children[node_id])
        return depth(root_ids[0])

    tree_depths     = [tree_depth(sent) for sent in doc.sentences]
    mean_tree_depth = sum(tree_depths) / len(tree_depths) if tree_depths else 0

    # Finite subordinate clause rate: ccomp + advcl
    finite_sub_count    = ccomp_count + advcl_count
    # Non-finite subordinate clause rate: xcomp + acl (bare, non-finite only)
    # acl:relcl (finite relative clauses) intentionally excluded — captured in g2_acl_relcl_rate
    nonfinite_sub_count = xcomp_count + acl_count

    # Noun-to-verb ratio: None for verbless headlines (retained as meaningful None)
    noun_verb_ratio = (noun_count + propn_count) / verb_count if verb_count > 0 else None

    # ── G7: Stance and Interaction ────────────────────────────────────────────
    first_person_count = sum(
        1 for w in all_words if w.lemma.lower() in FIRST_PERSON_PRONOUNS
    )
    second_person_count = sum(
        1 for w in all_words if w.lemma.lower() in SECOND_PERSON_PRONOUNS
    )
    # Existential there: 'there' tagged as expletive (deprel == 'expl')
    exist_there_count = sum(
        1 for w in all_words
        if w.text.lower() == 'there' and w.deprel == 'expl'
    )

    # ── Build feature dict ────────────────────────────────────────────────────
    return {
        # G1
        'g1_noun_rate':              noun_count            * per_1k,
        'g1_propn_rate':             propn_count           * per_1k,
        'g1_verb_rate':              verb_count            * per_1k,
        'g1_adj_rate':               adj_count             * per_1k,
        'g1_adv_rate':               adv_count             * per_1k,
        'g1_pron_rate':              pron_count            * per_1k,
        'g1_adp_rate':               adp_count             * per_1k,
        'g1_nominalization_rate':    nominalization_count  * per_1k,
        # G2
        'g2_nsubj_rate':             nsubj_count           * per_1k,
        'g2_amod_rate':              amod_count            * per_1k,
        'g2_nmod_rate':              nmod_count            * per_1k,
        'g2_compound_rate':          compound_count        * per_1k,
        'g2_ccomp_rate':             ccomp_count           * per_1k,
        'g2_xcomp_rate':             xcomp_count           * per_1k,
        'g2_advcl_rate':             advcl_count           * per_1k,
        'g2_acl_rate':               acl_count             * per_1k,
        'g2_acl_relcl_rate':         relcl_count           * per_1k,
        # G3
        'g3_agentless_passive_rate': agentless_passive_count * per_1k,
        'g3_by_passive_rate':        by_passive_count        * per_1k,
        'g3_progressive_rate':       progressive_count       * per_1k,
        'g3_part_clause_rate':       part_clause_count       * per_1k,
        # G4
        'g4_present_tense_rate':     present_tense_count   * per_1k,
        'g4_past_tense_rate':        past_tense_count      * per_1k,
        'g4_perfect_aspect_rate':    perfect_count         * per_1k,
        # G5
        'g5_necessity_modal_rate':   necessity_modal_count   * per_1k,
        'g5_possibility_modal_rate': possibility_modal_count * per_1k,
        'g5_predictive_modal_rate':  predictive_modal_count  * per_1k,
        # G6
        'g6_mean_sent_len':          mean_sent_len,
        'g6_mean_dep_dist':          mean_dep_dist,
        'g6_mean_tree_depth':        mean_tree_depth,
        'g6_finite_sub_rate':        finite_sub_count    * per_1k,
        'g6_nonfinite_sub_rate':     nonfinite_sub_count * per_1k,
        'g6_noun_verb_ratio':        noun_verb_ratio,       # None for verbless headlines
        # G7
        'g7_first_person_rate':      first_person_count  * per_1k,
        'g7_second_person_rate':     second_person_count * per_1k,
        'g7_exist_there_rate':       exist_there_count   * per_1k,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    print(f"Loading master dataset from {MASTER_FILE}")
    df = pd.read_csv(MASTER_FILE)
    print(f"Loaded {len(df)} rows\n")

    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT):
        done_df    = pd.read_csv(CHECKPOINT)
        start_idx  = len(done_df)
        results    = done_df.to_dict('records')
        print(f"Resuming from checkpoint: {start_idx} rows done, "
              f"{len(df) - start_idx} remaining\n")
    else:
        results   = []
        start_idx = 0

    total = len(df)

    for i, (_, row) in enumerate(df.iloc[start_idx:].iterrows(), start=start_idx):
        text     = str(row.get('text', ''))
        features = extract_features(text)

        record = {
            'prompt_id': row['prompt_id'],
            'genre':     row['genre'],
            'source':    row['source'],
            'model':     row.get('model', None),
            'stage':     row.get('stage', None),
        }
        record.update(features)
        results.append(record)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total} rows...")

        if len(results) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
            print(f"  Checkpoint saved: {len(results)} rows")

    # Save final output
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Saved {len(out_df)} rows to {OUTPUT_FILE}")

    # Remove checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print("Checkpoint removed.")

    # Sanity check
    feature_cols = [c for c in out_df.columns if c in FEATURE_NAMES]
    print(f"\nFeature columns: {len(feature_cols)}")
    nulls = out_df[feature_cols].isnull().sum()
    non_ratio_nulls = nulls.drop('g6_noun_verb_ratio', errors='ignore')
    if non_ratio_nulls.sum() == 0:
        print("  No unexpected nulls — all features extracted successfully.")
        ratio_nulls = nulls.get('g6_noun_verb_ratio', 0)
        print(f"  g6_noun_verb_ratio: {ratio_nulls} verbless headlines (expected)")
    else:
        print("  Unexpected nulls found:")
        print(nulls[nulls > 0])


if __name__ == "__main__":
    main()
