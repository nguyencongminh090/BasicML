---
id: FIX-0013
timestamp: 2026-09-02T21:31:22+07:00
todo_id: TODO-0007
---

## Prompt

Confirmed the English migration also covers `README.md`, `BasicML/logs/`, and the
language of new commit messages ("Yes it does"). Then: rewrite `README.md` in
English. User-supplied intro framing: the repo aims to learn ML/DL by building it
from scratch in NumPy; inspired by PyTorch architecture for its cohesion / low
coupling.

## Action

- Rewrote `README.md` end-to-end in English.
- Per AskUserQuestion: (1) translate + refresh to match current code, (2) rebuild
  roadmap table from the ai-audit backlog, (3) remove all LaTeX math.
- Refreshed content against the current tree: added `regularization.py`
  (L1/L2/ElasticNet), `nn/dropout.py`, `nn/sequential.py`, `nn/init.py`,
  `datasets/synthetic.py`, `visualize/decision_boundary.py`, `demo/`, the four
  `examples/` scripts, `data.csv`; dropped the stale `nn/models.py`
  (`LogisticRegressionModel`) and the removed `tepmlate_*.py` template files.
- Usage example updated to `Sequential(Linear, ReLU, Linear)` + `SGD`.
- Verified class/function names against source (`regularization.py`,
  `nn/activation.py` Tanh, `optim/sgd.py` + `momentum.py` signatures,
  `nn/linear.py`, `tensor.py`).

## Decision

- Refresh rather than literal translation: the old README described a codebase
  several TODOs out of date (no Dropout/Sequential/regularization, listed files
  that no longer exist), so a faithful translation would ship known-wrong docs.
- Math removed at user's request; descriptions kept in prose so the teaching
  value survives without the KaTeX-in-Markdown rendering caveats.
- Roadmap rebuilt from `todo/archive/INDEX.md` (TODO-0001..0004 done) plus known
  planned work (CNN/RNN/Attention/Transformer, autograd).

## Conclusion

Done. `README.md` is English-only, matches the current library surface, and has
no LaTeX. No code changed; nothing to run.

Follow-ups still under TODO-0007: `examples/train_linear.py`,
`examples/train_logistic.py`, `examples/check_gradients.py` still carry
Vietnamese docstrings; `BasicML/demo/layer_space_transformation.ipynb` markdown
cells are Vietnamese; `BasicML/logs/` older entries pending (now in scope).
