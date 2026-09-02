---
id: FIX-0011
timestamp: 2026-09-02T20:45:00+07:00
todo_id: TODO-0007
---

## Prompt

User: add rules to CLAUDE.md — English in code and comments; Clean-Code comment
discipline (comment at the right place, not freely); SOLID; a defined docstring
format for functions (purpose, params with type/options, ...). Follow-up: user
intends to internationalize the whole repo to English.

## Action

`CLAUDE.md`:
- "What this repo is": replaced "Documentation and commit history are in
  Vietnamese; code and identifiers are in English." with a statement that the
  repo is being internationalized — code / identifiers / comments / docstrings
  are English-only, existing Vietnamese comments/docs are being migrated
  (TODO-0007), historical commit messages and older `BasicML/logs/` stay as
  written.
- New `## Code style` section (before "AI audit — mandatory triggers"):
  - English only in `.py`.
  - Clean-Code comments: why not what; comment on the line it explains; no
    obvious-restatement comments; no commented-out code.
  - Google-style docstrings for every function / method / class, pointing at
    `basicml/datasets/synthetic.py` and `basicml/visualize/decision_boundary.py`
    as the in-repo reference; summary / `Args:` (meaning, type/shape, options,
    default) / `Returns:` / `Raises:` / math where it teaches.
  - SOLID, each principle mapped to this library's `Module` / `Loss` /
    `Optimizer` abstractions.
- Created `ai-audit` TODO-0007 (status `open`) to track the codebase migration.

## Decision

Google style chosen because the library code already uses it (`synthetic.py`,
`decision_boundary.py`) — no new convention, just make it explicit and
repo-wide. Kept the doc/commit-message language question open rather than
deciding it unilaterally (recorded in TODO-0007 notes).

## Conclusion

Done. CLAUDE.md updated; TODO-0007 filed. Open question for the user: does the
English migration also cover `README.md`, `BasicML/logs/`, and new commit
messages? Uncommitted with the session's other changes.
