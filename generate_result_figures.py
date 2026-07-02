import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Setup
out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Detection Accuracy by Tier
def plot_detection_accuracy():
    tiers = ["Tier A\n(Stylometric RF)", "Tier B\n(Semantic MLP)", "Tier C\n(DistilBERT LoRA)"]
    acc = [97.0, 95.0, 93.0]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=tiers, y=acc, palette=["#2a3b5c", "#3d8c83", "#d26a5c"], ax=ax)
    
    # Line plot on top for trend
    ax.plot(range(len(tiers)), acc, marker='o', color='black', linewidth=2)
    
    for i, v in enumerate(acc):
        ax.text(i, v + 0.5, f"{v}%", ha='center', fontweight='bold')
        
    ax.set_ylim(80, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Detection Accuracy by Tier")
    plt.tight_layout()
    plt.savefig(out_dir / "detection_accuracy.png", dpi=300)
    plt.close()

# 2. Cross-Author Transfer
def plot_cross_author():
    tests = ["Dickens → Austen", "Austen → Dickens"]
    acc = [83.8, 95.4]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=tests, y=acc, palette=["#2a3b5c", "#3d8c83"], ax=ax)
    ax.axhline(50, ls='--', color='gray', linewidth=1)
    
    for i, v in enumerate(acc):
        ax.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')
        
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cross-Author Transfer Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "cross_author_transfer.png", dpi=300)
    plt.close()

# 3. Feature Ablation
def plot_feature_ablation():
    configs = ["Full\nfeature set", "Remove\nLexical\nRichness", "Remove\nFunction\nWords", 
               "Only\nFunction\nWords", "Only\nLexical\nRichness"]
    acc = [97.0, 96.9, 93.0, 92.2, 62.5]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=configs, y=acc, palette=["#2a3b5c", "#3d8c83", "#d26a5c", "#d26a5c", "#888888"], ax=ax)
    
    for i, v in enumerate(acc):
        ax.text(i, v + 1, f"{v}%", ha='center', fontweight='bold', fontsize=9)
        
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Tier A Feature Ablation (Random Forest)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_ablation.png", dpi=300)
    plt.close()

# 4. GA Evasion
def plot_ga_evasion():
    gens = [0, 7]
    probs = [17.2, 90.3]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, probs, marker='o', color='#d26a5c', linewidth=3, markersize=10)
    
    ax.axhline(50, ls='--', color='gray', linewidth=1)
    ax.text(4, 51, "Decision boundary (50%)", color='gray', fontsize=8, ha='center')
    
    ax.text(gens[0], probs[0] + 3, f"Gen 0: {probs[0]}%", fontweight='bold', ha='center')
    ax.text(gens[1], probs[1] + 3, f"Gen 7: {probs[1]}%", fontweight='bold', ha='center')
    
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 7.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("P(classified as Human) by Tier C")
    ax.set_title("Genetic Algorithm Evasion of Tier C Detector")
    plt.tight_layout()
    plt.savefig(out_dir / "ga_evasion.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_detection_accuracy()
    plot_cross_author()
    plot_feature_ablation()
    plot_ga_evasion()
    print("All result figures generated successfully.")
