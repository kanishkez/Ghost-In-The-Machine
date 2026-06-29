"""
Task 4 — The Turing Test.
Genetic algorithm targeting a specific author's style.
Example: python -m src.task4_genetic --target_author "Charles Dickens"
"""
from __future__ import annotations
import json
import random
import argparse
from pathlib import Path
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from .config import CFG, PROJECT_ROOT, seed_everything
from .utils import GeminiClient

class HumanProbOracle:
    def __init__(self, device="cpu"):
        self.device = device
        base = AutoModelForSequenceClassification.from_pretrained(CFG["tier_c"]["base_model"], num_labels=3)
        self.model = PeftModel.from_pretrained(base, PROJECT_ROOT / CFG["paths"]["models_dir"] / "tier_c_lora").to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(PROJECT_ROOT / CFG["paths"]["models_dir"] / "tier_c_lora")

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        enc = self.tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=CFG["tier_c"]["max_length"]).to(self.device)
        logits = self.model(**enc).logits.cpu().numpy()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return probs[:, 0]

def initial_population(client: GeminiClient, topics: List[str], n: int, author: str) -> List[str]:
    pop = []
    style_map = {
        "Charles Dickens": "Mimic Charles Dickens: long, serpentine sentences; rich imagery; sentimental moral reflection; semicolons; concrete London detail.",
        "Jane Austen": "Mimic Jane Austen: precise, balanced clauses; sharp irony; free indirect discourse; focus on manners and social nuance."
    }
    for i in range(n):
        topic = topics[i % len(topics)]
        prompt = f"Write a 100-200 word paragraph on the topic: \"{topic}\". {style_map[author]} Return only the paragraph."
        pop.append(client.generate(prompt))
    return pop

def mutate(client: GeminiClient, paragraph: str, prompts: List[str]) -> str:
    p = random.choice(prompts)
    full = f"{p}\n\nParagraph:\n\"\"\"\n{paragraph}\n\"\"\"\n\nReturn only the rewritten paragraph."
    return client.generate(full)

def crossover(client: GeminiClient, parent1: str, parent2: str) -> str:
    prompt = f"Combine elements of the following two paragraphs to create a single coherent paragraph of 100-200 words. Keep the strongest stylistic aspects of both.\n\nParagraph 1:\n{parent1}\n\nParagraph 2:\n{parent2}\n\nReturn only the combined paragraph."
    return client.generate(prompt)

def run_ga(target_author: str):
    seed_everything(CFG["project"]["seed"])
    safe_name = target_author.split()[-1].lower()
    log_dir = PROJECT_ROOT / CFG["paths"]["ga_logs"] / safe_name
    log_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    oracle = HumanProbOracle(device=device)
    client = GeminiClient()
    topics = CFG["dataset"]["topics"]
    mut_prompts = CFG["genetic_algorithm"]["mutation_prompts"]

    print(f"=== Initializing GA for target: {target_author} ===")
    pop = initial_population(client, topics, CFG["genetic_algorithm"]["population_size"], target_author)
    log = []

    for gen in range(CFG["genetic_algorithm"]["generations"]):
        probs = oracle.predict_proba(pop)
        ranked = sorted(zip(pop, probs), key=lambda x: -x[1])
        best = ranked[0][1]; mean = float(np.mean(probs))
        print(f"[Gen {gen}] best P(Human)={best:.4f}  mean={mean:.4f}")
        log.append({"generation": gen, "best": float(best), "mean": mean, "best_text": ranked[0][0]})

        if best >= CFG["genetic_algorithm"]["target_human_prob"]:
            print(f"Reached target >= {CFG['genetic_algorithm']['target_human_prob']}")
            break

        elites = [t for t, _ in ranked[:CFG["genetic_algorithm"]["elite_size"]]]
        new_pop = list(elites)
        
        while len(new_pop) < CFG["genetic_algorithm"]["population_size"]:
            # Randomly decide between mutation and crossover
            if random.random() < 0.3 and len(elites) >= 2:
                # Crossover
                p1, p2 = random.sample(elites, 2)
                child = crossover(client, p1, p2)
            else:
                # Mutation
                parent = random.choice(elites)
                child = mutate(client, parent, mut_prompts)
            new_pop.append(child)
        pop = new_pop

    with open(log_dir / "ga_log.json", "w") as f:
        json.dump(log, f, indent=2)
    with open(log_dir / "ga_best.txt", "w") as f:
        f.write(ranked[0][0])
    print(f"\nBest final P(Human) = {ranked[0][1]:.4f}")

def personal_test(sop_path: str):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    oracle = HumanProbOracle(device=device)
    text = Path(sop_path).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.split()) >= 50]
    probs = oracle.predict_proba(paragraphs)
    for i, (p, prob) in enumerate(zip(paragraphs, probs)):
        verdict = "HUMAN" if prob > 0.5 else "AI"
        print(f"[{i+1}] P(Human)={prob:.3f}  -> {verdict}")
        print(f"    {p[:200]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_author", type=str, default="Charles Dickens", help="Author style to evolve")
    parser.add_argument("--sop", type=str, default=None, help="Path to SOP for personal test")
    args = parser.parse_args()
    
    if args.sop:
        personal_test(args.sop)
    else:
        run_ga(args.target_author)
