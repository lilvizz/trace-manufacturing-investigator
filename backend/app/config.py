"""
Shared, single-source-of-truth path constants.

This exists only to remove duplicated
`Path(__file__).resolve().parents[N]` guesswork from every script and
test file. It is intentionally tiny: no settings framework, no env
overrides beyond what's needed today.
"""

from pathlib import Path

# backend/app/config.py -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "apex-corpus"
INDEX_PATH = DATA_DIR / "indexes" / "apex_embeddings.json"
