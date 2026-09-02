---
id: FIX-0012
timestamp: 2026-09-02T20:50:00+07:00
todo_id: TODO-0007
---

## Prompt

User asked to translate the two new demo files (and, longer term, the rest of
the repo's Vietnamese code) to English, following the new `## Code style` rules
in CLAUDE.md.

## Action

Rewrote both files added this session, English-only with Google-style docstrings:
- `BasicML/demo/plot_dynamic_mlp_graph.py`
- `BasicML/demo/plot_dynamic_vanishing_gradient.py`

Per file:
- Module docstring translated.
- Every function / nested function got a Google-style docstring
  (summary / `Args:` / `Returns:` / `Raises:` as applicable), including the
  small helpers (`linear_layers`, `weight_grad`, `accuracy`, `act_out`, ...)
  and the inner `update` / `on_key` closures.
- Inline comments translated and trimmed to Clean-Code "why" notes; removed
  restatement-style comments.
- User-facing strings translated: `col_titles` ("plain MLP (no regularization)"),
  axis/panel titles, suptitle ("keys g / w"), print statements, error messages.
- `plot_dynamic_mlp_graph.py`: moved `linear_sizes` / `node_layout` above
  `animate` so definitions precede use (was a runtime-only forward reference).
- No behaviour change — same CONFIG, same computations, same layout.

## Decision

Full rewrite rather than line-by-line edits: the files are new, uncommitted, and
AI-authored this session, so a clean pass is lower-risk than dozens of
find/replace edits, and it let every docstring be brought to the same standard
at once.

## Conclusion

Done. `grep` for Vietnamese (diacritic + common ASCII-Vietnamese tokens) is
clean. `pyrefly check` on both files: 0 errors. Both demos re-run headless
end-to-end (60 epochs, `update()` forced on 3 frames) with no error. The rest of
the repo's Vietnamese comments remain — tracked in TODO-0007 for later sessions.

Commit split: the demo files themselves land already-English on their feature
branches (`feature/TODO-0005-mlp-graph-demo`, `feature/TODO-0006-vanishing-gradient-demo`);
this fix-log and FIX-0011 are committed on `dev` with the CLAUDE.md change. Once
the feature branches merge into `dev` the picture is coherent.
