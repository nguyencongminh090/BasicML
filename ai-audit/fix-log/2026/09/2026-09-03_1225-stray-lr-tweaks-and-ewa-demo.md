---
id: FIX-0027
timestamp: 2026-09-03T12:25:11+07:00
todo_id: TODO-0010
---

## Prompt

At `/make-commit` time the working tree also carried hand-made changes not
driven by a specific chat request. User chose "include everything" in the
commit-task classification questions.

## Action

Recorded (not authored by the assistant this session):

- `BasicML/examples/train_linear.py`: `LEARN_RATE` 0.1 -> 0.2 (user tweak).
- `BasicML/demo/plot_dynamic_linear.py`: `LEARN_RATE` 0.1 -> 0.5 (user tweak, on
  top of the FIX-0022 plain-loop refactor).
- `BasicML/demo/plot_ewa.py`: new EWA (exponentially weighted average) teaching
  demo, "AI generated" header, from the TODO-0009 optimizer thread — 3-panel
  figure (EWA vs noisy signal, bias correction, geometric weights), `EWA_SAVE`
  env var for headless save.

## Decision

Bundled into the single TODO-0010 commit per the user's explicit choice rather
than reverted or split out. lr values left as the user set them.

## Conclusion

Committed as part of `feat(loss): ...` on `feature/TODO-0010-absolute-loss`.
`plot_ewa.py` runs headless via `EWA_SAVE=/tmp/ewa.png python BasicML/demo/plot_ewa.py`.
