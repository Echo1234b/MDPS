import importlib
import sys
from pathlib import Path

# Ensure the project root (parent of this package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the legacy root-level package (which already contains all core logic)
_legacy_root_package = importlib.import_module("__init__")  # Root __init__.py module

# Re-export everything from the legacy root package so that `import mdps` works transparently
for _attr in dir(_legacy_root_package):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_legacy_root_package, _attr)

del importlib, sys, Path, PROJECT_ROOT, _legacy_root_package