from __future__ import annotations

from pathlib import Path
import sys
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_integrity.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
