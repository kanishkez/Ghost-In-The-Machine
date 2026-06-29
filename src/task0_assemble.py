import json
import os
import glob
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from config import load_config

def get_human_paragraphs(author_id, author_name, min_words, max_words, limit):
    raw_dir = "data/raw"
    files = glob.glob(os.path.join(raw_dir, f"{author_id}_*.txt"))
    paragraphs = []
    
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Import clean_text to utilize Deep Gutenberg Cleaning
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils import clean_text
        
        text = clean_text(text)
        blocks = text.split("\n\n")
        
        for block in blocks:
            block = block.strip().replace("\n", " ")
            if not block: continue
            
            words = block.split()
            if min_words <= len(words) <= max_words:
                paragraphs.append({
                    "text": block,
                    "label_class": "A",
                    "label_author": author_name,
                    "source_id": f"{author_name}_{author_id}"
                })
                if len(paragraphs) >= limit:
                    return paragraphs
    return paragraphs

def main():
    config = load_config()
    
    # 1. Human (Class A) — 50 per author, 10 per book
    print("Extracting Class A (Human)...")
    dickens_ids = [book["id"] for book in config["gutenberg_books"]["Charles Dickens"]]
    austen_ids = [book["id"] for book in config["gutenberg_books"]["Jane Austen"]]
    
    PER_BOOK = 10  # 5 books × 10 = 50 per author
    
    class_a = []
    for d_id in dickens_ids:
        class_a.extend(get_human_paragraphs(d_id, "Dickens", 100, 200, PER_BOOK))
    for a_id in austen_ids:
        class_a.extend(get_human_paragraphs(a_id, "Austen", 100, 200, PER_BOOK))
        
    print(f"Got {len(class_a)} Class A paragraphs.")

    # 2. AI Generic (Class B)
    with open("data/processed/class_b.json", "r") as f:
        raw_b = json.load(f)
    class_b = [{"text": x["text"], "label_class": "B", "label_author": "None", "source_id": f"AI_Gen_{i}"} for i, x in enumerate(raw_b)]
    print(f"Got {len(class_b)} Class B paragraphs.")

    # 3. AI Impostors (Class C)
    with open("data/processed/class_c_dickens.json", "r") as f:
        c_dickens = json.load(f)
    with open("data/processed/class_c_austen.json", "r") as f:
        c_austen = json.load(f)
        
    class_c = [{"text": x["text"], "label_class": "C", "label_author": "Dickens", "source_id": f"AI_Imp_D_{i}"} for i, x in enumerate(c_dickens)]
    class_c += [{"text": x["text"], "label_class": "C", "label_author": "Austen", "source_id": f"AI_Imp_A_{i}"} for i, x in enumerate(c_austen)]
    print(f"Got {len(class_c)} Class C paragraphs.")

    # Custom strict Train/Val/Test Split to prevent Book Leakage
    train_data, val_data, test_data = [], [], []
    import random
    seed = config["project"]["seed"]
    random.seed(seed)
    
    # Split Human data strictly by book
    for author_name, ids in [("Dickens", dickens_ids), ("Austen", austen_ids)]:
        random.shuffle(ids)
        # 5 books -> 3 Train, 1 Val, 1 Test
        train_ids = [f"{author_name}_{i}" for i in ids[:3]]
        val_ids = [f"{author_name}_{i}" for i in ids[3:4]]
        test_ids = [f"{author_name}_{i}" for i in ids[4:]]
        
        for p in class_a:
            if p["label_author"] == author_name:
                if p["source_id"] in train_ids: train_data.append(p)
                elif p["source_id"] in val_ids: val_data.append(p)
                elif p["source_id"] in test_ids: test_data.append(p)
                
    # Split AI data randomly 60/20/20
    import random
    seed = config["project"]["seed"]
    random.seed(seed)
    random.shuffle(class_b)
    random.shuffle(class_c)
    
    def split_ai(data_list):
        n = len(data_list)
        t_end = int(n * 0.6)
        v_end = int(n * 0.8)
        train_data.extend(data_list[:t_end])
        val_data.extend(data_list[t_end:v_end])
        test_data.extend(data_list[v_end:])
        
    split_ai(class_b)
    split_ai(class_c)
    
    def finalize(data_list, name):
        df = pd.DataFrame(data_list)
        df["label"] = df["label_class"].map({"A": "Human", "B": "AI_Generic", "C": "AI_Impostor"})
        df["author"] = df["label_author"]
        df = df.sample(frac=1, random_state=config["project"]["seed"]).reset_index(drop=True)
        df["sample_id"] = [f"{name}_{i}" for i in range(len(df))]
        df.to_csv(f"data/processed/{name}.csv", index=False)
        return df
        
    train_df = finalize(train_data, "train")
    val_df = finalize(val_data, "val")
    test_df = finalize(test_data, "test")
    
    print(f"Saved strict splits! Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save all for consistency
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df.to_csv("data/processed/all.csv", index=False)

if __name__ == "__main__":
    main()
