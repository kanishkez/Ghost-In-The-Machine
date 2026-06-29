"""
Task 3 — The Smoking Gun.
- Tier A: SHAP TreeExplainer for global feature importance.
- Tier C: Captum LayerIntegratedGradients for token-level saliency on impostor paragraphs.
- Error analysis: Human misclassified as AI, split by author.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap

from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import joblib

from .config import CFG, PROJECT_ROOT
from .utils import load_csv
from .task2_tier_a import FEAT_COLS, LABEL_MAP, load_splits_with_features

def shap_tier_a():
    train, val, test = load_splits_with_features()
    clf = joblib.load(PROJECT_ROOT / CFG["paths"]["models_dir"] / "tier_a_rf.joblib")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(test[FEAT_COLS].values)
    
    if isinstance(shap_values, list):
        sv = np.stack(shap_values, axis=-1)
    else:
        sv = shap_values
        
    fig_dir = PROJECT_ROOT / CFG["paths"]["figures_dir"]
    shap.summary_plot(sv.mean(axis=-1), feature_names=FEAT_COLS, features=test[FEAT_COLS].values, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "tier_a_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

def load_tier_c():
    base = AutoModelForSequenceClassification.from_pretrained(CFG["tier_c"]["base_model"], num_labels=3)
    model = PeftModel.from_pretrained(base, PROJECT_ROOT / CFG["paths"]["models_dir"] / "tier_c_lora")
    tok = AutoTokenizer.from_pretrained(PROJECT_ROOT / CFG["paths"]["models_dir"] / "tier_c_lora")
    return model, tok

def captum_saliency(text: str, target_class: int, model, tok, device="cpu"):
    model.eval().to(device)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    input_ids = inputs["input_ids"]
    ref_ids = torch.full_like(input_ids, tok.pad_token_id)

    def forward(input_ids):
        attn = (input_ids != tok.pad_token_id).long()
        return model(input_ids=input_ids, attention_mask=attn).logits

    try:
        embed_layer = model.base_model.model.distilbert.embeddings
    except AttributeError:
        try:
            embed_layer = model.base_model.distilbert.embeddings
        except AttributeError:
            # Fallback that tries to find the word embeddings
            embed_layer = model.get_base_model().distilbert.embeddings.word_embeddings

    lig = LayerIntegratedGradients(forward, embed_layer)
    attributions, delta = lig.attribute(inputs=input_ids, baselines=ref_ids, target=target_class, return_convergence_delta=True)
    attributions = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    tokens = tok.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())
    return tokens, attributions

def plot_saliency(tokens, attributions, title, out_path):
    fig, ax = plt.subplots(figsize=(14, 3))
    colors = ["red" if a < 0 else "green" for a in attributions]
    ax.bar(range(len(tokens)), attributions, color=colors)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_title(title)
    ax.set_ylabel("Integrated gradient")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

def run_captum_on_impostors(n: int = 3):
    model, tok = load_tier_c()
    # Try putting to MPS if available to speed it up
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    test = load_csv("test.csv")
    fig_dir = PROJECT_ROOT / CFG["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    for author in CFG["project"]["authors"]:
        impostors = test[(test["label"] == "AI_Impostor") & (test["author"] == author)]
        if len(impostors) == 0: continue
        
        impostors = impostors.sample(min(n, len(impostors)), random_state=42)
        safe_author = author.split()[-1]
        
        for i, (_, row) in enumerate(impostors.iterrows()):
            tokens, attrs = captum_saliency(row["text"], target_class=2, model=model, tok=tok, device=device)
            plot_saliency(tokens, attrs, f"Captum LIG — {safe_author} Impostor {i+1}", fig_dir / f"captum_{safe_author.lower()}_imp_{i+1}.png")
            top = sorted(zip(tokens, attrs), key=lambda x: -abs(x[1]))[:15]
            print(f"\n[{safe_author} Sample {i+1}] Top attributions:")
            for t, a in top: print(f"  {t:15s} {a:+.4f}")

def error_analysis():
    test = load_csv("test.csv")
    model, tok = load_tier_c()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    preds = []
    with torch.no_grad():
        for text in test["text"]:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            logits = model(**inputs).logits
            preds.append(logits.argmax(-1).item())
            
    test["pred"] = preds
    
    misclassified_human = test[(test["label"] == "Human") & (test["pred"] != 0)]
    print(f"\n[Error analysis] {len(misclassified_human)} Human paragraphs misclassified as AI.")
    
    samples = misclassified_human.head(3)
    for i, (_, row) in enumerate(samples.iterrows()):
        pred_label = [k for k, v in LABEL_MAP.items() if v == row['pred']][0]
        print(f"\n--- Misclassified Human sample {i+1} by {row['author']} (predicted = {pred_label}) ---")
        print(row["text"][:600], "...")

def main():
    print("=== SHAP on Tier A ===")
    shap_tier_a()
    print("\n=== Captum on Tier C ===")
    run_captum_on_impostors(n=3)
    print("\n=== Error analysis ===")
    error_analysis()

if __name__ == "__main__":
    main()
