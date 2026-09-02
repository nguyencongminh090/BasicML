---
id: FIX-0017
timestamp: 2026-09-02T23:21:15+07:00
todo_id: TODO-0008
---

## Prompt

"Add more Activation Function @BasicML/basicml/nn/activation.py — I have wrote
those base function, you will add more." Follow-up choice (AskUserQuestion):
core set + parametric extras + simple extras.

## Action

- Branch `feature/TODO-0008-more-activations` off `dev`.
- Rewrote `BasicML/basicml/nn/activation.py`:
  - Added docstrings to `Activation`, `Sigmoid`, `ReLU`, `Tanh` (no behaviour
    change).
  - New `Activation` subclasses, each with a hand-derived `backward`:
    `Identity`, `LeakyReLU(negative_slope)`, `PReLU(num_parameters, init)`,
    `ELU(alpha)`, `SELU`, `Softplus`, `GELU` (tanh approximation),
    `Swish(beta)` (SiLU at beta=1), `Mish`, `Hardtanh(min_val, max_val)`,
    `Hardsigmoid`, `Softmax(axis=-1)`.
  - `PReLU` holds a `requires_grad=True` `Tensor` slope, overrides
    `parameters()`, and accumulates `dL/da` with `+=` (matches `Linear`; needs
    `zero_grad()` between steps). Supports a shared scalar slope or one slope
    per feature.
  - `Softmax.backward` uses the closed-form Jacobian-vector product
    `f * (grad_output - sum(grad_output * f))`, never materialising the Jacobian.
    Numerically stable forward (subtract row max).
- Added TODO-0008 + instructions note + active-index row.
- Follow-up on user note "No comments in my core (basicml)": removed the few
  `#` comments from the new `activation.py` (math folded into docstrings), and
  rewrote the CLAUDE.md `## Code style` comment rule — the core library
  (`BasicML/basicml/`) now carries zero inline comments; `examples/`/`demo/`
  may keep sparse `why`-comments. Pre-existing comments in
  `datasets/synthetic.py` and `visualize/decision_boundary.py` left untouched
  (out of scope for this change).
- Then, on "clean docstring": tightened every docstring in `activation.py` —
  removed RST `` `` ``/`:class:` markup and the extra background prose, kept the
  Google-style summary+Args+Returns+Raises structure with formulae in plain
  text, added the missing `Raises: ValueError` entries. No code change.
- Finally, on "empty as other files": stripped **all** docstrings and comments
  from `activation.py` to match the rest of the bare core (`nn/`, `optim/`,
  `tensor.py` carry neither). Kept the `RuntimeError` forward-order guards.
  Rewrote the CLAUDE.md `## Code style` bullets accordingly — core modules stay
  bare; docstrings required only in `basicml/datasets/` + `basicml/visualize/`.

## Decision

Kept the repo's manual-`backward` convention (no autograd) and the
`self.x` / `self.out` caching pattern already used by `Sigmoid`/`ReLU`/`Tanh`.
Numerically stable formulations chosen where relevant: `np.logaddexp(0, x)` for
softplus/Mish, `np.expm1` for ELU/SELU, max-subtraction for softmax. GELU uses
the tanh approximation (the GPT/BERT variant) so the derivative is elementary.

## Conclusion

Done. Verified with a central-difference gradient check (scratchpad
`gc.py`): all 14 layers max abs error ~1e-9 (< 1e-4); PReLU input gradient,
scalar slope gradient (ana == num), and per-feature slope shape all correct;
`Sequential(Linear, GELU, Linear, Softmax)` forward rows sum to 1 and backward
returns the right shape. `pyrefly check`: no new errors in `activation.py`
(the 9 pre-existing errors in `examples/ref_code.py` and `demo/plot_dynamic_*`
are unrelated and untouched). Not yet committed/merged.
