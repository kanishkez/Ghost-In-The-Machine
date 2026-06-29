"""
Task 2 — Tier C: The Transformer.
Fine-tune DistilBERT-base-uncased with LoRA for 3-class classification.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CFG, PROJECT_ROOT, seed_everything
from .utils import load_csv

LABEL_MAP = {"Human": 0, "AI_Generic": 1, "AI_Impostor": 2}

class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(texts.tolist(), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, torch.tensor(self.labels[i], dtype=torch.long)

def main():
    seed_everything(CFG["project"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train, val, test = load_csv("train.csv"), load_csv("val.csv"), load_csv("test.csv")

    tok = AutoTokenizer.from_pretrained(CFG["tier_c"]["base_model"])
    base = AutoModelForSequenceClassification.from_pretrained(CFG["tier_c"]["base_model"], num_labels=3)

    lora_cfg = LoraConfig(
        r=CFG["tier_c"]["lora"]["r"],
        lora_alpha=CFG["tier_c"]["lora"]["alpha"],
        lora_dropout=CFG["tier_c"]["lora"]["dropout"],
        target_modules=CFG["tier_c"]["lora"]["target_modules"],
        bias=CFG["tier_c"]["lora"]["bias"],
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base, lora_cfg).to(device)
    model.print_trainable_parameters()

    g = torch.Generator()
    g.manual_seed(CFG["project"]["seed"])

    tr_dl = DataLoader(TextDS(train["text"], train["label"].map(LABEL_MAP).values, tok, CFG["tier_c"]["max_length"]), batch_size=CFG["tier_c"]["batch_size"], shuffle=True, generator=g)
    val_dl = DataLoader(TextDS(val["text"], val["label"].map(LABEL_MAP).values, tok, CFG["tier_c"]["max_length"]), batch_size=CFG["tier_c"]["batch_size"])
    test_dl = DataLoader(TextDS(test["text"], test["label"].map(LABEL_MAP).values, tok, CFG["tier_c"]["max_length"]), batch_size=CFG["tier_c"]["batch_size"])

    total_steps = len(tr_dl) * CFG["tier_c"]["epochs"]
    warmup = int(total_steps * CFG["tier_c"]["warmup_ratio"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG["tier_c"]["lr"]))
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_macro_auc = 0
    best_state = None
    patience = 3
    epochs_no_improve = 0
    
    for ep in range(CFG["tier_c"]["epochs"]):
        model.train()
        for batch in tr_dl:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            opt.zero_grad()
            out = model(**inputs).logits
            loss = loss_fn(out, labels)
            loss.backward(); opt.step(); sched.step()

        model.eval()
        all_logits, all_ys = [], []
        with torch.no_grad():
            for batch in val_dl:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs).logits.cpu().numpy()
                all_logits.append(logits); all_ys.append(labels.numpy())
        logits = np.vstack(all_logits); ys = np.concatenate(all_ys)
        
        y_bin_val = label_binarize(ys, classes=[0,1,2])
        val_macro_auc = roc_auc_score(y_bin_val, logits, average="macro", multi_class="ovr")
        acc = (logits.argmax(1) == ys).mean()
        
        print(f"epoch {ep+1:02d} val_acc={acc:.4f} val_macro_auc={val_macro_auc:.4f}")
        
        if val_macro_auc > best_macro_auc:
            best_macro_auc = val_macro_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {ep+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    all_logits, all_ys = [], []
    with torch.no_grad():
        for batch in test_dl:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits.cpu().numpy()
            all_logits.append(logits); all_ys.append(labels.numpy())
    logits = np.vstack(all_logits); ys = np.concatenate(all_ys)
    preds = logits.argmax(1)

    print("\n[Tier C] Test classification report:")
    print(classification_report(ys, preds, target_names=list(LABEL_MAP.keys())))
    y_bin = label_binarize(ys, classes=[0,1,2])
    macro_auc = roc_auc_score(y_bin, logits, average="macro", multi_class="ovr")
    print(f"[Tier C] Test macro AUC = {macro_auc:.3f}")

    out_dir = PROJECT_ROOT / CFG["paths"]["models_dir"] / "tier_c_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
