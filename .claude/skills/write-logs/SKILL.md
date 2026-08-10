---
name: write-logs
description: 'Indexes the current session in ai-audit/convo/INDEX.md by pointing at Claude Code''s own native session transcript — no re-summarization, near-zero tokens, can''t hallucinate. Optionally also writes a curated human-readable Markdown recap to BasicML/logs/, but only when explicitly asked for one. Use when the user says "write log", "save log", "index this session", "summary log", or "WRITE LOG".'
---

# Session Indexing

## Default: index only (fast, accurate, cheap)

This is what runs for "write log" / "save log" / "WRITE LOG" unless the user explicitly also wants a written recap (see below).

1. **Determine `<project-slug>` and `<session-id>`** — read them straight off the "Scratchpad Directory" path given in the system prompt (`/tmp/claude-<uid>/<project-slug>/<session-id>/scratchpad`). Don't derive or guess them.

2. **Ensure the index exists** — `ai-audit/convo/` is gitignored and local; create `ai-audit/convo/INDEX.md` with its header (see `ai-audit/README.md`) if it doesn't exist yet.

3. **Allocate an ID** — from the repo root: `ai-audit/scripts/next_id.sh convo`.

4. **Write a short, factual Keywords/Topic** — a handful of words/phrases naming what was discussed. Not a narrative — this is the only text generated for the default path, so keep it cheap and low-risk (topic tags, not a recap of decisions).

5. **Append one row** to `ai-audit/convo/INDEX.md`:

   `ID | Date | Keywords/Topic | Transcript | Summary | Related TODO`

   - **Transcript** — `~/.claude/projects/<project-slug>/<session-id>.jsonl`. This is the ground truth: Claude Code writes it automatically, verbatim. Always fill this in.
   - **Summary** — `-` unless the optional recap below was also written this run.
   - **Related TODO** — any `TODO-XXXX` touched this session, else `-`.

6. **Confirm** by printing the ID and the transcript path. Do not write anything to `BasicML/logs/` unless the user asked for that too.

## Optional: written recap (only on explicit request — e.g. "write a full log", "write a detailed summary")

1. Determine current local time, create `BasicML/logs/` if needed, and write `BasicML/logs/<YYYY-MM-DD_HHmm>.md` from the template below.
2. Update the `Summary` column of the row just added in the default path above to point at this file.
3. Confirm by printing the path to the saved log.

### Log Template

```markdown
# Session Log — <YYYY-MM-DD>

## Overview
<1-3 sentences describing what was worked on>

---

## Key Discussions

### 1. <Topic>
- **Question/Task:** ...
- **Decision/Fix:** ...

### 2. <Topic>
...

---

## Files Changed / Created

| File | Change |
|---|---|
| `path/to/file.py` | Description of change |

---

## Architecture Summary

<Optional: code block or bullet list showing current state>

---

## Key Takeaways
1. ...
2. ...
```
