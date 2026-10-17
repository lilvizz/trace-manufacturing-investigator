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

# The project .env lives at the repository root. It is loaded by explicit
# path, so the Gemini key is found regardless of the working directory the
# app, a test or a CLI tool is started from (python-dotenv's default lookup
# falls back to the current directory in interactive/`-c` contexts).
ENV_PATH = PROJECT_ROOT / ".env"


def load_environment() -> bool:
    """Load PROJECT_ROOT/.env without overriding variables already set in
    the process environment. Returns True if the file exists. Never logs
    or returns any value from the file."""
    from dotenv import load_dotenv

    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)
        return True
    return False
