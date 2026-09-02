---
id: FIX-0002
timestamp: 2026-09-02T16:54:34+07:00
todo_id: TODO-0003
---

## Prompt

"I have just write Regularizer code. Now I want you to write examples ... write example in
way of comparing without Regularization and within then compare overfit." User confirmed
scope: self-contained example first, shared metrics helper later.

## Action

- Branched `feature/TODO-0003-regularization-example` off `dev`.
- Created TODO-0003 (`in-progress`) + `instructions/TODO-0003.md` + `todo/INDEX.md` row.
- Added `BasicML/examples/train_regularization.py`: over-capacity MLP
  (`Linear(2,64)-ReLU-Linear(64,64)-ReLU-Linear(64,1)-Sigmoid`, `he` init) on a small noisy
  `make_moons` train set (N=40, noise=0.30) vs an 800-point held-out val set. Three runs
  sharing seed/data/init: no reg, `L2(0.05)`, `L1(0.01)`, wiring `regularizer=` into `Momentum`.
  Prints train vs val BCE + accuracy + generalization gap, and `R(theta)` from `reg.penalty()`;
  decision-boundary plot behind `SHOW_PLOT`.
- No library code changed. Accuracy / gap computed inline.

## Decision

- Example home (not a `demo/` animation) — it is a numeric comparison, matches
  `examples/train_*.py`.
- `make_moons` over a 1D threshold set: needs a genuinely nonlinear boundary so an
  over-capacity net has something to overfit.
- Reused seed + `np.random.seed` per run so the only variable across runs is the penalty.
- Reported loss stays data-only (BCE); penalty shown separately — matches TODO-0002 decision.

## Conclusion

Done. Ran via `python3.13` (no `python` on PATH): no-reg overfits hard
(train acc 1.000 / val 0.794, loss gap +2.99); `L2(0.05)` closes it (train 0.875 / val 0.845,
loss gap +0.009); `L1(0.01)` in between. Full script incl. matplotlib path exits 0 under
`MPLBACKEND=Agg`. `pyrefly check` NOT run (pyrefly not on PATH) — user to run.
Follow-ups: user runs pyrefly + commits via `/make-commit`; later TODO to extract
`accuracy`/`train_val_split` into a shared metrics module and upgrade this demo.
