---
id: FIX-0014
timestamp: 2026-09-02T21:49:46+07:00
todo_id: TODO-0007
---

## Prompt

User shared an external analysis report ("how professional OSS projects write
READMEs" + a scorecard of our README) and said: read it and edit the README
accordingly. Report's main recommendations: add badges, move Quick Start above
the long class reference, add a visual hook (GIF/screenshot), add a Gradient
Verification section, add a License section, link to the math write-ups in
`logs/`.

## Action

- Asked the user two decisions: (1) license → **MIT**; (2) visuals → **demo
  table + generated static PNGs** (animations can't be rendered here).
- Added `LICENSE` (MIT, © 2026 nguyenminh).
- Generated `assets/` PNGs by running the real scripts with a patched
  `plt.show` (scratchpad `gen_assets.py`, Agg backend):
  - `assets/layer_transformations.png` — from `demo/plot_layer_transformations.py` (README banner)
  - `assets/regularization.png` — from `examples/train_regularization.py`
  - `assets/logistic_fit.png` — from `examples/train_logistic.py`
- Restructured `README.md`: Hero (title + tagline + 4 badges: Python 3.13+,
  core pure NumPy, no-autograd, MIT) → banner image → Key highlights → Quick
  start (install / minimal MLP / run examples) → Verification & gradient
  checking (real `check_gradients.py` output, `< 1e-7`) → Project architecture
  (tree + components table) → Interactive demos (table + 2 result images) →
  Roadmap → Design notes (links to `BasicML/logs/`) → License.
- Verified all component names and the gradient-check numbers against source
  by running `check_gradients.py`.

## Decision

- Static PNGs over broken GIF links: the report wanted animation, but there are
  no image assets and none can be produced here; real static renders from the
  actual library beat a placeholder.
- Badge "core: pure NumPy" (not "zero-dependency"): the library core is NumPy
  only, but examples/demos need pandas + matplotlib — the badge shouldn't
  overclaim.
- Kept the "no LaTeX math" decision from FIX-0013; `< 1e-7` is plain text.
- MIT chosen by the user; `LICENSE` file added so the badge/section aren't
  dangling.

## Conclusion

Done. `README.md` restructured to the report's target shape, `LICENSE` added,
three real PNGs committed under `assets/`. No library code changed;
`check_gradients.py` and the three plot scripts all run clean.

Follow-ups still under TODO-0007: Vietnamese docstrings in `examples/`
(`train_linear`, `train_logistic`, `check_gradients`) and all `demo/*.py`;
notebook `layer_space_transformation.ipynb`; commit-message language.
