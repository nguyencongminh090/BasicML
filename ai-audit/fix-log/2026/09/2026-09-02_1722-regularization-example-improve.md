---
id: FIX-0003
timestamp: 2026-09-02T17:22:02+07:00
todo_id: TODO-0003
---

## Prompt

"okay, you can improve/fix my BasicML/examples/train_regularization.py" — after the user's
own edit (hidden layers switched to `Sigmoid`) made the `L2(0.05)` run collapse to the
trivial 0.5 predictor.

## Action

- Restored `ReLU` hidden activations (`he` init) in `build_model`; added a module docstring
  paragraph on why hidden layers must not saturate here (sigmoid hidden -> weak data gradient
  -> constant `lambda_ * param` decay wins -> dead 0.5 equilibrium).
- Added `History` (per-epoch train/val BCE, every `EVAL_EVERY`) and a `Result` holder with
  `loss_gap` / `acc_gap` properties.
- New `print_summary` comparison table; plot upgraded to 2xN — decision boundary (top) +
  learning curve train vs val (bottom), val separating upward as the overfit tell.
- `N_VAL` 200 -> 800.

## Decision

- Fix over lowering `lambda_`: the demo's point is that the *same* penalty strength either
  helps or doesn't depending on the architecture; ReLU is the correct hidden activation and
  the failure mode is worth an explicit comment rather than silently retuning.
- Kept accuracy / gap inline (no library changes) per the scope chosen for TODO-0003.

## Conclusion

Fixed. Ran under `MPLBACKEND=Agg`, exit 0. no-reg train/val acc 1.000/0.794 (loss gap +2.99);
L2(0.05) 0.875/0.845 (+0.009); L1(0.01) 0.900/0.843 (+0.155) — intended overfit-vs-regularized
contrast restored. `pyrefly check` not run (not on PATH) — user to run before committing via
`/make-commit`.
