import sys
from pathlib import Path

# Ensure `app` is importable regardless of the working directory pytest
# is invoked from (console-script `pytest` does not add cwd to
# sys.path the way `python -m pytest` does).
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    # Registered here as well as in pytest.ini so the marker is known (and
    # `-m "not integration"` works without warnings) whichever directory
    # pytest is started from.
    config.addinivalue_line(
        "markers",
        "integration: tests that require live external services or generated embedding indexes",
    )
