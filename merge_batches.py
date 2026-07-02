"""
Merge batch JSON files from Gemini chat into the existing class_b/class_c JSON files,
then reassemble the full dataset and regenerate train/val/test splits.

Expected files in data/processed/batches/:
  b1.json .. b5.json     → Class B (generic AI)
  cd1.json, cd2.json     → Class C (Dickens impostor)
  ca1.json, ca2.json     → Class C (Austen impostor)

Each file should be a JSON array of {"text": "...", "topic": "..."}.
"""
import json
import re
import glob
import random
import sys
from pathlib import Path

BATCH_DIR = Path("data/processed/batches")
PROCESSED = Path("data/processed")


def load_and_clean_json(filepath):
    """Load a JSON file, handling common Gemini formatting issues."""
    raw = Path(filepath).read_text(encoding="utf-8")

    # Strip markdown code fences if Gemini wrapped the output
    raw = re.sub(r'^```(?:json)?\s*\n?', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\n?```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse {filepath}: {e}")
        print(f"  First 200 chars: {raw[:200]}")
        return []

    if not isinstance(data, list):
        print(f"  [WARN] {filepath} is not a JSON array, wrapping...")
        data = [data]

    # Validate entries
    valid = []
    for item in data:
        if isinstance(item, dict) and "text" in item:
            text = item["text"].strip()
            if len(text.split()) >= 30:  # at least 30 words
                valid.append({
                    "text": text,
                    "topic": item.get("topic", "unknown")
                })
            else:
                print(f"  [skip] Too short ({len(text.split())} words): {text[:60]}...")
        else:
            print(f"  [skip] Invalid entry: {str(item)[:100]}")

    return valid


def main():
    # ── Load existing data ──
    with open(PROCESSED / "class_b.json") as f:
        existing_b = json.load(f)
    with open(PROCESSED / "class_c_dickens.json") as f:
        existing_cd = json.load(f)
    with open(PROCESSED / "class_c_austen.json") as f:
        existing_ca = json.load(f)

    print(f"Existing: B={len(existing_b)}, C_Dickens={len(existing_cd)}, C_Austen={len(existing_ca)}")

    # ── Merge Class B batches ──
    new_b = []
    for f in sorted(BATCH_DIR.glob("b*.json")):
        entries = load_and_clean_json(f)
        print(f"  {f.name}: {len(entries)} valid paragraphs")
        new_b.extend(entries)
    print(f"  Total new Class B: {len(new_b)}")

    # ── Merge Class C Dickens batches ──
    new_cd = []
    for f in sorted(BATCH_DIR.glob("cd*.json")):
        entries = load_and_clean_json(f)
        print(f"  {f.name}: {len(entries)} valid paragraphs")
        new_cd.extend(entries)
    print(f"  Total new Class C Dickens: {len(new_cd)}")

    # ── Merge Class C Austen batches ──
    new_ca = []
    for f in sorted(BATCH_DIR.glob("ca*.json")):
        entries = load_and_clean_json(f)
        print(f"  {f.name}: {len(entries)} valid paragraphs")
        new_ca.extend(entries)
    print(f"  Total new Class C Austen: {len(new_ca)}")

    # ── Combine with existing ──
    all_b = existing_b + new_b
    all_cd = existing_cd + new_cd
    all_ca = existing_ca + new_ca

    print(f"\nFinal totals: B={len(all_b)}, C_Dickens={len(all_cd)}, C_Austen={len(all_ca)}")
    print(f"Grand total AI paragraphs: {len(all_b) + len(all_cd) + len(all_ca)}")

    # ── Save updated JSON files ──
    with open(PROCESSED / "class_b.json", "w") as f:
        json.dump(all_b, f, indent=2)
    with open(PROCESSED / "class_c_dickens.json", "w") as f:
        json.dump(all_cd, f, indent=2)
    with open(PROCESSED / "class_c_austen.json", "w") as f:
        json.dump(all_ca, f, indent=2)

    print("\n✓ Updated JSON files saved.")
    print("\nNext step: run  python3 reassemble_and_run.py")


if __name__ == "__main__":
    main()
