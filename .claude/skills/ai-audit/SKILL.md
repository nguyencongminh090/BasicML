---
name: ai-audit
description: Operating manual for this repo's ai-audit/ workspace — timestamped fix-log entries, a TODO backlog, per-TODO instruction notes, and a conversation index. Use whenever creating or updating a TODO, logging a fix, recording a decision from a chat session, or searching for a past conversation by topic.
---

# AI Audit Workspace

Full schema and workflow: [ai-audit/README.md](../../../ai-audit/README.md) (repo root). Read it before writing to `ai-audit/` for the first time in a session — this skill only lists the fast paths the schema depends on.

## Golden rule (this backlog can exceed 100 items)

Never read every file to "catch up." Always read the relevant `INDEX.md` first — it's a bounded table. Open one specific item's file only when you already know its ID and need the full detail. Never `grep -r`/glob the whole `ai-audit/` tree looking for context.

## IDs

Allocate every new ID with `ai-audit/scripts/next_id.sh {todo|fix|convo}` — never hand-guess the next number, that's how IDs collide at scale.

## WIP limit (Kanban)

This project runs Kanban, not sprints — `status` is the board column. **At most 2 TODOs may be `in-progress` at once.** Before setting a third to `in-progress`, check `ai-audit/todo/INDEX.md`'s Status column first; if the cap is already hit, finish one, or move it back to `open`/`blocked` before starting the new one.

## Fast paths

- **New TODO** (from a user report or another agent's output): `next_id.sh todo` → copy `ai-audit/todo/TEMPLATE.md` to `ai-audit/todo/TODO-XXXX.md`, fill in Description/Requirement/source, add one row to `ai-audit/todo/INDEX.md`.
- **Working a TODO during a chat session**: append a dated section to `ai-audit/instructions/TODO-XXXX.md` (create from `instructions/TEMPLATE.md` if it doesn't exist yet) — what was discussed, decided, and what's next. Never delete prior entries.
- **Applying a concrete fix**: `next_id.sh fix` → new file under `ai-audit/fix-log/YYYY/MM/` from `fix-log/TEMPLATE.md` (Prompt, Action, Decision, Conclusion) → one row in `ai-audit/fix-log/INDEX.md`. Link `todo_id` in the frontmatter if one applies.
- **Closing a TODO**: set `status: done` (or `cancelled`) in its frontmatter, then move the file *and* its `INDEX.md` row into `ai-audit/todo/archive/`. Its instructions file stays where it is.
- **Saving or finding a conversation**: `/write-log` indexes by default — it appends a row to `ai-audit/convo/INDEX.md` (gitignored, local) with `Transcript` pointing at Claude Code's own native `.jsonl` session file (ground truth, zero rewrite). It does **not** write a Markdown recap to `BasicML/logs/` unless the user explicitly asks for one — that keeps indexing near-free. To find a past conversation, search the index's Keywords column for the user's context, then open `Transcript` for accuracy or `Summary` (if present) for a quick recap.
