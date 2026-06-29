"""
Task 2 — Tier A: The Statistician.
Random Forest on stylometric features. 
Reads pre-split CSVs generated in Task 0 to ensure strict boundary alignment.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from seaborn import heatmap as sns_heatmap
import joblib

from .config import CFG, PROJECT_ROOT
from .utils import load_csv

# Add func words to feature columns
FUNC_WORDS = ["the", "of", "and", "to", "a", "in", "that", "it", "is", "was", "for", "on", "as", "with", "his", "he", "be", "at", "by", "this"]

FEAT_COLS = [
    "ttr","n_tokens","n_types","adj_noun_ratio","avg_dep_depth",
    "flesch_kincaid","n_sentences","avg_sentence_len",
] + [f"punct_{p}" for p in [";","—","!","?",",",".",":","(",")"]] + [f"func_{fw}" for fw in FUNC_WORDS]

LABEL_MAP = {"Human": 0, "AI_Generic": 1, "AI_Impostor": 2}

def load_splits_with_features():
    train = load_csv("train.csv")
    val = load_csv("val.csv")
    test = load_csv("test.csv")
    
    feats = pd.read_csv(PROJECT_ROOT / CFG["paths"]["processed_dir"] / "features.csv")
    train = train.merge(feats, on="sample_id", suffixes=("", "_feat"))
    val = val.merge(feats, on="sample_id", suffixes=("", "_feat"))
    test = test.merge(feats, on="sample_id", suffixes=("", "_feat"))
    
    # After merging, the 'label' and 'author' cols might be duplicated. We just use the original ones.
    return train, val, test

def fit_and_eval():
    train, val, test = load_splits_with_features()
    
    X_tr, y_tr = train[FEAT_COLS].values, train["label"].map(LABEL_MAP).values
    X_test, y_test = test[FEAT_COLS].values, test["label"].map(LABEL_MAP).values

    clf = RandomForestClassifier(
        n_estimators=CFG["tier_a"]["n_estimators"],
        max_depth=CFG["tier_a"]["max_depth"],
        class_weight=CFG["tier_a"]["class_weight"],
        random_state=42, n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=CFG["tier_a"]["cv_folds"], shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_tr, y_tr, cv=skf, scoring="accuracy")
    print(f"[Tier A] 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_test)
    print("\n[Tier A] Test classification report:")
    print(classification_report(y_test, y_pred, target_names=list(LABEL_MAP.keys())))

    # ROC one-vs-rest
    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    y_proba = clf.predict_proba(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    for i, name in enumerate(LABEL_MAP.keys()):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    macro_auc = np.mean(list(roc_auc.values()))

    fig_dir = PROJECT_ROOT / CFG["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,5))
    for i, name in enumerate(LABEL_MAP.keys()):
        plt.plot(fpr[i], tpr[i], label=f"{name} (AUC={roc_auc[i]:.3f})")
    plt.plot([0,1],[0,1],"k--", alpha=.4)
    plt.title(f"Tier A — One-vs-Rest ROC (macro AUC={macro_auc:.3f})")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "tier_a_roc.png", dpi=150); plt.close()

    # SHAP global
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        sv_arr = np.stack(shap_values, axis=-1)  # older versions (n_classes, n_samples, n_features) to (n_samples, n_features, n_classes)
    else:
        sv_arr = shap_values # newer versions natively output array
        
    shap.summary_plot(sv_arr.mean(axis=-1), feature_names=FEAT_COLS, features=X_test, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "tier_a_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    model_dir = PROJECT_ROOT / CFG["paths"]["models_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "tier_a_rf.joblib")
    
    print(f"[Tier A] Test macro AUC = {macro_auc:.3f}")

if __name__ == "__main__":
    fit_and_eval()
