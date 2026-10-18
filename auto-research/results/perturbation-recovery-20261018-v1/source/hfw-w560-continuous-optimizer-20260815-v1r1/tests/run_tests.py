from __future__ import annotations

import inspect
import tempfile
import time
from pathlib import Path

from . import test_production_contract as suite


def main() -> int:
    tests = [
        (name, value) for name, value in vars(suite).items()
        if name.startswith("test_") and callable(value)
    ]
    failures = []
    started = time.monotonic()
    for name, test in sorted(tests):
        try:
            parameters = inspect.signature(test).parameters
            if "tmp_path" in parameters:
                with tempfile.TemporaryDirectory(prefix="hfw_w560_test_") as directory:
                    test(Path(directory))
            else:
                test()
            print(f"PASS {name}")
        except BaseException as error:
            failures.append((name, repr(error)))
            print(f"FAIL {name}: {error!r}")
    elapsed = time.monotonic() - started
    print(f"RESULT passed={len(tests) - len(failures)} failed={len(failures)} seconds={elapsed:.3f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
