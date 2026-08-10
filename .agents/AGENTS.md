# Agent Rules

## Git Commit Workflow
When the user says "MAKE COMMIT", or otherwise asks you to commit changes (including changes the user wrote by hand, not just your own edits):
1. Read the `commit-task` SKILL from `.agents/skills/commit-task/SKILL.md` and follow it: classify the change (or ask the user its purpose if ambiguous), place it on the correct branch per the branch policy below, then use the `git-commit` SKILL (`github/awesome-copilot@git-commit`) to actually make the commit.

### Branch policy
`main` and `dev` are the two long-lived branches. `main` is protected on GitHub (PR-only, no direct/force pushes) — never commit or push to it directly. This project runs Kanban (`ai-audit/todo/` status field is the board), capped at **2 TODOs `in-progress` at once**. `feat`-type changes get their own `feature/TODO-XXXX-<slug>` branch off `dev`, `fix`-type changes get `fix/TODO-XXXX-<slug>` — each named after the backlog TODO it's pulling, only created if the WIP cap allows it; everything else (docs/chore/refactor/etc.) can go directly on `dev`. Promoting `dev` → `main` always requires asking the user first — see the "Git branching" section of `CLAUDE.md` for the full policy.

## Write Log Workflow
When the user says "WRITE LOG", "write log", "save log", or "summary log", you must:
1. Read the `write-logs` SKILL from `.agents/skills/write-logs/SKILL.md`.
2. Follow the skill instructions exactly to summarize the conversation and save the log.

## AI Audit Workspace
`ai-audit/` (repo root) is the audit trail and TODO backlog for AI-assisted work on this repo: timestamped fix-log entries, a TODO backlog, per-TODO instruction notes, and a conversation index. Read `ai-audit/README.md` for the schema before writing to it. Always read a folder's `INDEX.md` first — never scan the whole tree — and allocate new IDs with `ai-audit/scripts/next_id.sh {todo|fix|convo}`.

## Markdown Rendering Guidelines
When writing Markdown files containing LaTeX math (e.g., `README.md`), always follow these rules to ensure the IDE parser renders them correctly:
1. **Avoid inline math (`$...$`) for complex equations**: The parser often breaks when inline math contains combinations of parentheses and superscripts (e.g., `(\hat{y} - y)^2`).
2. **Use block math (`$$...$$`) instead**: For equations containing superscripts, subscripts, or sum symbols, use block math.
3. **Always surround block math with empty lines**: If a `$$` block is placed immediately after text (e.g., directly under a list item or paragraph without an empty line in between), the Markdown parser will merge them into a single text paragraph and fail to trigger the math renderer.
4. **Avoid indenting math blocks inside lists**: Break out of bulleted lists before defining complex block equations to prevent list-parsing bugs.
