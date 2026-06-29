"""Shared utilities: Gemini client, text cleaning, IO."""
from __future__ import annotations
import os
import re
import time
from pathlib import Path
from typing import List
import pandas as pd
import google.generativeai as genai

from .config import CFG, PROJECT_ROOT

class GeminiClient:
    """Thin wrapper around google-generativeai with retry/backoff."""
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GEMINI_API_KEY env var.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, max_retries: int = 10, **kwargs) -> str:
        for attempt in range(max_retries):
            try:
                # 12s between calls = 5 RPM, well under 15 RPM free tier limit
                time.sleep(12)
                resp = self.model.generate_content(prompt, **kwargs)
                return resp.text.strip()
            except Exception as e:
                err_str = str(e)
                # Parse retry_delay hint from API error if available
                import re as _re
                match = _re.search(r'retry in (\d+(?:\.\d+)?)s', err_str)
                if match:
                    wait = float(match.group(1)) + 5  # Add 5s buffer
                else:
                    wait = (2 ** attempt) + 65
                print(f"[Gemini retry {attempt+1}/{max_retries}] sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"Gemini failed after {max_retries} retries.")

_GUTENBERG_START = re.compile(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.DOTALL)
_GUTENBERG_END   = re.compile(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.DOTALL)

def strip_gutenberg_boilerplate(text: str) -> str:
    m = _GUTENBERG_START.search(text)
    if m: text = text[m.end():]
    m = _GUTENBERG_END.search(text)
    if m: text = text[:m.start()]
    return text

def clean_text(text: str) -> str:
    text = strip_gutenberg_boilerplate(text)
    
    # Deep Gutenberg Cleaning (Metadata, Headers, Editor Notes)
    # Remove CHAPTER XII, VOLUME II, etc.
    text = re.sub(r"(?im)^(CHAPTER|VOLUME|BOOK|PART)\s+[A-Z0-9IVXLCDM]+\.?\s*$", "", text)
    # Remove all-caps lines (usually titles or editor notes)
    text = re.sub(r"(?m)^[A-Z\s.,;:\"'-]{5,}$", "", text)
    # Remove bracketed editor notes
    text = re.sub(r"\[.*?\]", "", text)
    # Remove isolated roman numerals
    text = re.sub(r"(?im)^[IVXLCDM]+\.?\s*$", "", text)
    
    # Standardize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    return text.strip()

def split_into_paragraphs(text: str, min_words: int = 100, max_words: int = 200) -> List[str]:
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    for p in raw:
        wc = len(p.split())
        if min_words <= wc <= max_words:
            out.append(p)
        elif wc > max_words:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf, buf_wc = [], 0
            for s in sentences:
                sw = len(s.split())
                if buf_wc + sw > max_words and buf:
                    if min_words <= buf_wc <= max_words:
                        out.append(" ".join(buf))
                    buf, buf_wc = [s], sw
                else:
                    buf.append(s); buf_wc += sw
            if buf and min_words <= buf_wc <= max_words:
                out.append(" ".join(buf))
    return out

def save_csv(df: pd.DataFrame, name: str) -> Path:
    out_dir = PROJECT_ROOT / CFG["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    df.to_csv(p, index=False)
    return p

def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / CFG["paths"]["processed_dir"] / name)
