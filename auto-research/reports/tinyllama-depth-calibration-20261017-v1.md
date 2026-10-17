# TinyLlama calibration v1: preserved harness failure

This frozen run completed teacher evaluation, then stopped after constructing the first raw spline fit because the logging call passed `kind` both positionally and by keyword. The algorithm did not fail a scientific or numerical gate. No compressed model was scored and no optimizer update occurred.

- Teacher new-window CE: 2.825760081410408.
- Teacher reused-window CE: 2.727322205901146, exact parent replay.
- Terminal status: FAILED. Wall time: 67.36738296299882 seconds.
- Preserved: original source/protocol/registration, input moments, teacher results, fidelity checks and terminal failure.

The source is not repaired in place. A separately registered v2 renames the logger argument, preserves all scientific choices, and deducts this run from the combined 30-minute measurement budget. See [v2 report](tinyllama-depth-calibration-20261017-v2.md).
