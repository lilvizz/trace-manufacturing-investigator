import sys
from pathlib import Path

# Ensure `app` is importable regardless of the working directory pytest
# is invoked from (console-script `pytest` does not add cwd to
# sys.path the way `python -m pytest` does).
sys.path.insert(0, str(Path(__file__).resolve().parent))
