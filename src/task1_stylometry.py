"""
Task 1 — The Fingerprint.
Computes stylometric features per paragraph and groups analysis by label and author.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
import textstat
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .config import CFG, PROJECT_ROOT
from .utils import load_csv

NLP = spacy.load("en_core_web_sm", disable=["ner"])
PUNCT_LIST = [";", "—", "!", "?", ",", ".", ":", "(", ")"]

FUNC_WORDS = ["the", "of", "and", "to", "a", "in", "that", "it", "is", "was", "for", "on", "as", "with", "his", "he", "be", "at", "by", "this"]

def tree_depth(token) -> int:
    depth = 0
    cur = token
    while cur is not None:
        depth += 1
        cur = cur.head if cur.head != cur else None
    return depth

def avg_dep_depth(doc) -> float:
    depths = [tree_depth(t) for t in doc]
    return float(np.mean(depths)) if depths else 0.0

def stylometric_features(text: str) -> dict:
    doc = NLP(text)
    tokens = [t for t in doc if t.is_alpha]
    types = set(t.lemma_.lower() for t in tokens)
    n_tokens = max(1, len(tokens))
    ttr = len(types) / n_tokens

    pos_counts = Counter(t.pos_ for t in doc)
    n_nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    n_adj   = pos_counts.get("ADJ", 0)
    adj_noun_ratio = n_adj / max(1, n_nouns)
    depth = avg_dep_depth(doc)
    n_chars = max(1, len(text))
    punct_counts = {p: text.count(p) / n_chars * 1000 for p in PUNCT_LIST}
    fk = textstat.flesch_kincaid_grade(text)
    
    # Calculate function word frequencies (per 1000 words)
    words_lower = [t.text.lower() for t in tokens]
    func_counts = {f"func_{fw}": words_lower.count(fw) / n_tokens * 1000 for fw in FUNC_WORDS}

    return {
        "ttr": ttr, "n_tokens": len(tokens), "n_types": len(types),
        "adj_noun_ratio": adj_noun_ratio, "avg_dep_depth": depth,
        "flesch_kincaid": fk, "n_sentences": len(list(doc.sents)),
        "avg_sentence_len": len(tokens) / max(1, len(list(doc.sents))),
        **{f"punct_{p}": punct_counts[p] for p in PUNCT_LIST},
        **func_counts
    }

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    feats = [stylometric_features(txt) for txt in tqdm(df["text"], desc="Stylometry")]
    feat_df = pd.DataFrame(feats)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)

def hapax_legomena(texts, target_words: int = 5000) -> int:
    pool = []
    for t in texts:
        toks = [w.lower() for w in t.split() if w.isalpha()]
        pool.extend(toks)
        if len(pool) >= target_words: break
    pool = pool[:target_words]
    counts = Counter(pool)
    return sum(1 for w, c in counts.items() if c == 1)

def make_figures(feat_df: pd.DataFrame):
    fig_dir = PROJECT_ROOT / CFG["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    feat_df["category"] = feat_df["label"] + "_" + feat_df["author"].fillna("Generic")

    punct_cols = [c for c in feat_df.columns if c.startswith("punct_")]
    pivot = feat_df.groupby("category")[punct_cols].mean()
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Punctuation Density (per 1000 chars) by Category")
    plt.tight_layout()
    plt.savefig(fig_dir / "punctuation_heatmap.png", dpi=150); plt.close()

    for col, name in [("ttr", "Type-Token Ratio"), ("flesch_kincaid", "Flesch-Kincaid Grade"),
                      ("avg_dep_depth", "Avg Dependency Tree Depth"), ("adj_noun_ratio", "Adjective / Noun Ratio")]:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=feat_df, x="category", y=col, palette="Set2")
        plt.title(name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_dir / f"box_{col}.png", dpi=150); plt.close()

    hapax = {}
    for cat, sub in feat_df.groupby("category"):
        hapax[cat] = hapax_legomena(sub["text"].tolist(), 5000)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(hapax.keys()), y=list(hapax.values()), palette="Set2")
    plt.title("Hapax Legomena (5000-word pooled sample)")
    plt.xticks(rotation=45)
    plt.ylabel("Unique-once words")
    plt.tight_layout()
    plt.savefig(fig_dir / "hapax_legomena.png", dpi=150); plt.close()
    return hapax

def main():
    df = load_csv("all.csv")
    feat_df = build_feature_table(df)
    feat_df.to_csv(PROJECT_ROOT / CFG["paths"]["processed_dir"] / "features.csv", index=False)
    hapax = make_figures(feat_df)
    print("Hapax legomena per category:", hapax)
    print(feat_df.groupby("category")[["ttr","adj_noun_ratio","avg_dep_depth","flesch_kincaid"]].mean().round(3))

if __name__ == "__main__":
    main()
