---
id: FIX-0022
timestamp: 2026-09-03T11:46:03+07:00
todo_id: TODO-0010
---

## Prompt

"@BasicML/demo/plot_dynamic_linear.py Remove adaptive learning rate, make it
normal as train_linear"

## Action

- Removed the `OneCycle` schedule dataclass and its `schedule.at(progress)` call.
- Removed early-stopping (`EARLY_STOP_PATIENCE`, `EARLY_STOP_MIN_DELTA`,
  `best_cost`/`no_improve` tracking) and the `MAX_EPOCHS`/`LR_CYCLE`/`WARMUP_FRAC`
  config block.
- Added `EPOCHS = 400` / `LEARN_RATE = 0.1` to match `examples/train_linear.py`;
  `Adam(model.parameters(), lr=LEARN_RATE)` now runs a plain fixed-lr loop.
- Updated module + `train_and_record` docstrings accordingly.
- `pyrefly check` clean for this file (pre-existing `set_3d_properties` warning
  untouched); ran `train_and_record` headless — 400 epochs, final cost 0.0022.

## Decision

The demo is the animated companion to `train_linear.py`; the user wants the two
to teach the same plain full-batch Adam loop. Fixed lr + fixed epoch count, no
schedule, no early stop.

## Conclusion

Fixed. Verified via headless training run and type check.
