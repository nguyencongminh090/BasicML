---
id: FIX-0021
timestamp: 2026-09-03T01:44:50+07:00
todo_id: TODO-0009
---

## Prompt

"okay, only the vanishing back to Momentum" -- after the user tested Adam on
`plot_dynamic_vanishing_gradient.py` and observed the Sigmoid+Xavier net
reaching ~98.5% val accuracy, which erases the demo's intended "deep Sigmoid net
fails to train" contrast.

## Action

`git checkout -- BasicML/demo/plot_dynamic_vanishing_gradient.py` on
`feature/TODO-0009-more-optimizers` -- reverts FIX-0020's edits to that single
file: restores `from basicml.optim.momentum import Momentum`, the `MOMENTUM = 0.9`
config const, `LEARN_RATE = 0.10`, `optimizer = Momentum(model.parameters(),
lr=LEARN_RATE, momentum=MOMENTUM)`, and removes the Adam docstring paragraph.
The other 7 demos and both examples stay on Adam.

## Decision

Adam is scale-invariant per parameter (`m̂ / (√v̂ + ε)` ≈ `sign(g)` * O(1)), so
it produces ~lr-sized updates even where backprop has driven the gradient to
1e-7 -- it divides out exactly the magnitude loss that "vanishing gradient"
names. That is a real and important property, but it defeats the one demo whose
entire purpose is to *show* vanishing gradients stalling training. Keeping this
file on Momentum is the deliberate single exception.

## Conclusion

Reverted. Headless run: Sigmoid+Xavier stalls at loss 0.6545 / 60.2% val acc,
ReLU+He reaches loss 0.0 / 98.2% -- the textbook contrast is back. Panel 4 (RMS
by depth) shows the attenuation as before. No other files touched.
