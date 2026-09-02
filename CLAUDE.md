# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A from-scratch deep learning learning project (`BasicML/`). The goal is understanding, not production use: every algorithm is implemented in pure NumPy, favoring clarity over performance, with no autograd — gradients are computed by hand in each module's `backward()`. The repo is being internationalized to English: code, identifiers, comments, and docstrings are English-only, and existing Vietnamese comments/docs are being migrated (see `ai-audit` TODO-0007). Historical commit messages and older `BasicML/logs/` entries stay as written.

## Commands

There is no build system, test suite, or package manager beyond `pip install numpy pandas matplotlib`. Python >= 3.13 is required.

Run an example end-to-end (each script self-inserts `BasicML/` onto `sys.path`, so run directly with `python`, not `-m`):

```bash
python BasicML/examples/train_linear.py      # Linear regression on BasicML/data.csv
python BasicML/examples/train_logistic.py    # Logistic regression on synthetic data, plots result
python BasicML/demo/plot_dynamic_linear.py   # Animated training visualization
python BasicML/demo/plot_dynamic_logistic.py
python BasicML/demo/plot_dynamic_3d_logistic.py
```

Type checking (config at `pyrefly.toml`, search path `BasicML`, interpreter `/usr/bin/python3.13`):

```bash
pyrefly check
```

There are no automated tests in the repo.

## Architecture

`BasicML/basicml/` is a small PyTorch-inspired library with a strict separation between tensors, modules, losses, and optimizers — but **backward passes are manual**, not autograd-traced. Every `forward()` caches whatever it needs (`self.x`, `self.out`, ...) so the corresponding `backward(grad_output)` can compute local gradients and return the upstream gradient. Callers are responsible for driving the chain: `loss.backward()` → `model.backward(grad)`, propagating through each layer's own `backward()` in reverse order (see `models.LogisticRegressionModel.backward`, which manually chains `sigmoid.backward` into `linear.backward`).

Key pieces:
- **`tensor.Tensor`** (`basicml/tensor.py`) — thin `numpy.ndarray` wrapper carrying `data`, `grad`, and `requires_grad`. Arithmetic dunders (`+ - * / @`) return new `Tensor`s but do **not** build a graph; they're used for direct value composition, not autodiff.
- **`nn.Module`** (`basicml/nn/module.py`) — abstract base with `forward()` (abstract) and `parameters()` (default: empty list). `__call__` delegates to `forward`. Every layer (`Linear`, `Sigmoid`, `ReLU`) and composite model subclasses this and additionally implements its own `backward()` by convention (not enforced by the ABC).
- **`nn.Linear`** — holds `w` (Xavier or He init, chosen via `init=`) and `b` as `requires_grad=True` Tensors; `backward` accumulates into `w.grad`/`b.grad` (`+=`, so `zero_grad()` must be called between steps) and returns `grad_output @ w.T`.
- **`nn.models.LogisticRegressionModel`** — composes `Linear` + `Sigmoid`; the pattern for building any new composite model.
- **`nn.loss.Loss`** — `__call__(y_pred, y_true)` computes and caches the scalar loss; `backward()` (no args) returns `dL/dy_pred` using the cached values. `BinaryCrossEntropy` clips predictions to avoid `log(0)`.
- **`optim.Optimizer`** — abstract `step()`/`zero_grad()` over a `list[Tensor]` of parameters. `SGD` is plain gradient descent; `Momentum` tracks per-parameter velocity in parallel arrays indexed the same as `self.parameters`.

Standard training loop shape used throughout examples/demos: `forward → loss(y_pred, y) → loss.backward() → model.backward(grad) → optimizer.step() → optimizer.zero_grad()`.

## Repo-specific conventions

- Imports are column-aligned (see any file under `basicml/`) — match this style when editing existing files.
- When editing `README.md` or other Markdown containing LaTeX math: use block math (`$$...$$`) rather than inline math for anything with superscripts/subscripts/sums, and always surround `$$` blocks with blank lines (including when breaking out of a list) — the Markdown/KaTeX renderer used here otherwise fails to trigger or merges text.
- `.claude/skills/` holds project skills for Claude Code (`commit-task`, `git-commit`, `write-logs`, `ai-audit`, `machine-learning`, `documentation-writer`, `find-skills`, `prompt-architect`, `latex-document-skill`); `.claude/commands/` exposes `/make-commit`, `/write-log`, `/todo-add`, and `/fix-log` as shortcuts. `/make-commit` runs `commit-task` (classify → branch-place → delegate to `git-commit`), not `git-commit` directly. **`git-commit`'s trigger description overlaps with `commit-task`'s ("user asks to commit changes") and it has no branch-policy awareness — never invoke `git-commit` directly for any commit intent in this repo, even without the `/make-commit` command; always go through `commit-task` first.** `write-log` indexes the session in `ai-audit/convo/INDEX.md` by default (points at the native session transcript, no rewrite); it only writes a Markdown summary under `BasicML/logs/` (existing logs live there) when explicitly asked for one.
- `.agents/` (gitignored) is a duplicate copy of the same skills for other agent tools (e.g. Copilot, Antigravity) that read that convention instead — keep the two in sync manually if a skill changes.
- `ai-audit/` (repo root) is the audit trail and TODO backlog for AI-assisted work on this repo — timestamped fix-log entries, a TODO backlog, per-TODO instruction notes from chat sessions, and a conversation index. See `ai-audit/README.md` for the full schema; the `ai-audit` skill has the operating fast-paths. **Always read a folder's `INDEX.md` before opening individual item files** — the backlog is designed to survive 100+ items without an agent having to scan the whole tree. `ai-audit/convo/` is gitignored: it indexes conversations by pointing at Claude Code's own native session transcripts (ground truth, no rewrite) rather than requiring a regenerated summary.

## Code style

- **English only in code.** Identifiers, comments, docstrings, and developer-facing string literals are English. No Vietnamese in `.py` files.
- **The core library modules stay bare — no `#` comments, no docstrings.** Every file directly under `BasicML/basicml/` (`nn/`, `optim/`, `tensor.py`, …) carries *zero* `#` comments and *zero* docstrings on its classes, methods, and functions: the code, type hints, and clear names are the whole story. If a line needs a comment to be understood, rename it or extract a helper. Keep the `raise RuntimeError("backward called before forward pass")` guards — they are behaviour, not documentation.
- **Docstrings live only in the "library edge" packages** — `basicml/datasets/` and `basicml/visualize/` — where every function, method, and class gets a Google-style docstring (as in `basicml/datasets/synthetic.py` and `basicml/visualize/decision_boundary.py`):
  - First line: what problem the callable solves / what it produces — not a paraphrase of its name.
  - `Args:` — for each parameter: what it means, expected type / array shape, valid options, and what a default implies.
  - `Returns:` — meaning and type / shape.
  - `Raises:` — each exception and the condition that triggers it.
  - Where it aids understanding (this is a teaching repo), state the math the function implements.
- **`examples/` and `demo/`** may use sparse Clean-Code `why`-comments (never `what`-comments, never commented-out code) and module/function docstrings where they help a reader follow the script.
- **SOLID:**
  - *Single responsibility* — one function/class does one thing; split when its docstring needs an "and".
  - *Open/closed* — add a new `Module` / `Loss` / `Optimizer` subclass rather than branching inside an existing one.
  - *Liskov* — a subclass must honour its base's contract (`forward`/`backward` shapes, `parameters()` semantics).
  - *Interface segregation* — keep base classes minimal; do not force layers to implement what they do not need.
  - *Dependency inversion* — depend on the `Module` / `Loss` / `Optimizer` abstractions, not concrete classes (e.g. a training loop takes a `Module`, not a `Linear`).

## AI audit — mandatory triggers

- **Modify Task (any edit to user code/config/docs).** Whenever you change a file the user owns — not just AI-authored files — you MUST run the `ai-audit` skill and record it: append a dated entry to the relevant `ai-audit/instructions/TODO-XXXX.md` while working, and file a fix-log entry (`ai-audit/fix-log/`) once the change is applied. Trivial, no-op, or purely generated-artifact changes still get a fix-log row. Do not consider a code modification finished until its audit trail exists.
- **Ambiguity or scope touching the user's own work → ask, don't assume.** If a request is unclear, underspecified, could be interpreted multiple ways, or would alter design/architecture/conventions the user established, stop and ask the user for their decision or opinion before acting (use `AskUserQuestion` for concrete choices). Do not pick a default and proceed on the user's behalf for these; only act unprompted when the intent and approach are unambiguous.

## Git branching

Two long-lived branches: `main` (protected — releases only) and `dev` (integration branch). Everything else is a short-lived branch off `dev`:

- **Features** — branch `feature/TODO-XXXX-<short-name>` from `dev`, named after the `ai-audit` TODO it's pulling. Merge back into `dev` only, never straight to `main`.
- **Fixes** — branch `fix/TODO-XXXX-<short-name>` from `dev` (its own branch, never committed straight onto `dev` or `main`). Merge back into `dev`.
- **Kanban WIP limit**: at most 2 TODOs `in-progress` at once (see `ai-audit/README.md`) — a `feature/*`/`fix/*` branch is only created if the WIP cap allows it.
- **`main` is protected**: never commit, push, or merge to it directly. The only thing that reaches `main` is a PR from `dev`. This is enforced both behaviorally and on GitHub (branch protection on `main`: PR required, no force-push, no deletion; admin override left enabled for emergencies).
- **Promoting `dev` → `main`**: once the changes on `dev` are verified — `pyrefly check` plus running whichever `BasicML/examples/`/`BasicML/demo/` scripts exercise the change (there's no automated test suite yet, so this is "tests pass" for now) — ask the user before opening the PR. Never merge `dev` into `main` unassisted, even if verification is clean.
- The `commit-task` skill (triggered by `/make-commit` or "MAKE COMMIT") enforces the feature/fix branch-placement half of this policy at commit time — classifying the change and routing it to the right branch before handing off to `git-commit`.

## Session hygiene

Each conversation should stay on one topic — this keeps `ai-audit/convo/INDEX.md` entries (one row per session) meaningfully searchable by keyword, and keeps each native transcript focused. When the user's request is a clear topic switch from what this conversation has been doing (e.g. "AI audit workspace setup" → "implement a CNN layer"), say so explicitly and suggest starting a new conversation before proceeding — don't just silently switch gears. Use judgment: a follow-up, a related bug, or a natural continuation of the same thread is not a topic change and doesn't need a warning.
