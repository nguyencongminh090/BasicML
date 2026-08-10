---
name: commit-task
description: 'Orchestrates a commit end-to-end: classifies the change (or asks the user its purpose), enforces this repo''s branch policy from CLAUDE.md (main protected; feat/fix each get their own TODO-linked branch off dev, gated by the ai-audit WIP limit), then hands off to the git-commit skill for the actual conventional-commit message and execution. Use whenever the user asks to commit changes, says "make commit", "commit this", or triggers "MAKE COMMIT" — including commits for code the user wrote by hand, not just AI-authored changes.'
---

# Commit Task

Handles "commit this" end-to-end: classify → place it on the right branch → delegate to `git-commit` for the actual commit. Full branch policy is in [CLAUDE.md](../../../CLAUDE.md) under "Git branching" — this skill enforces that policy at commit time, it doesn't replace `git-commit`'s message-crafting logic.

**Don't assume you wrote the change under review.** The user may be asking you to commit their own hand-written work. Always inspect the actual current diff — never infer purpose or scope from conversation history alone, and never fabricate a rationale just to force a clean classification.

## 1. Check where we are

```bash
git branch --show-current
git status --porcelain
git diff --stat              # and: git diff --staged --stat, if anything is already staged
```

## 2. Classify the change

Read the diff and pick a Conventional Commit type — `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert` (see the `git-commit` skill's table for definitions). If the diff is a clean, single, obvious type, proceed with it. If it's ambiguous — mixed purposes, could plausibly read as either a fix or a feature, touches unrelated things, or the diff alone doesn't make the intent clear — **ask the user directly** (e.g. "Is this a new feature, a bug fix, or something else?") rather than guessing.

## 3. Place it on the right branch

- **Currently on `main`** → stop before doing anything else. `main` is protected — nothing commits there directly (see CLAUDE.md). Ask whether to switch to `dev`, or create a `feature/*`/`fix/*` branch off `dev`, before continuing.
- **Type is `feat` or `fix`** → this repo runs Kanban (see `ai-audit/README.md`); every `feature/*`/`fix/*` branch must map to a pulled TODO card, and the branch itself is the WIP-limit gate:
  1. Read `ai-audit/todo/INDEX.md`'s Status column. If **2 TODOs are already `in-progress`** and this change isn't one of them, stop — tell the user the WIP limit is hit and ask them to finish or park (`open`/`blocked`) one before starting new work.
  2. Identify the TODO this branch is for. If the user names an existing one, use its ID. If none exists yet, run the `ai-audit` skill's "New TODO" fast path to create one (`next_id.sh todo`, fill Description/Requirement) so the branch always traces back to a backlog item — don't create a `feature/*`/`fix/*` branch with no TODO behind it.
  3. Set that TODO's `status: in-progress` and `updated` date in its frontmatter and in `todo/INDEX.md`.
  4. If not already on the right branch, create it off `dev`, named after the TODO:
     ```bash
     git fetch origin dev
     git checkout -b feature/TODO-XXXX-<short-slug> origin/dev   # or fix/TODO-XXXX-<short-slug>
     ```
- **Any other type** (`docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`) → fine directly on `dev`, or on the current `feature/*`/`fix/*` branch if one is already checked out. No TODO/WIP requirement — doesn't occupy a WIP slot. Never on `main`.
- **On some other branch** (not `main`/`dev`/`feature/*`/`fix/*`) → ask the user how they want to proceed rather than assuming what it's for.

## 4. Commit

Hand off to the **`git-commit` skill** to stage the right files, generate the conventional-commit message (type/scope/description) from the diff, and run the actual `git commit`.

## 5. After committing

State the branch and the commit made. Offer to push the branch (`git push -u origin <branch>`) but don't push without confirmation, same as any other push. Don't open a PR or touch `main` here — promoting `dev` → `main` is a separate, always-ask-first step already covered by CLAUDE.md's "Git branching" section.
