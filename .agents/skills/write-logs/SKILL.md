---
name: write-logs
description: 'Summarizes the current conversation session and writes a structured Markdown log file to the workspace logs/ directory, timestamped with the current date and time. Use when the user says "write log", "save log", "summary log", or "WRITE LOG".'
---

# Write Session Log

## Overview

Summarize the current conversation and save it as a structured Markdown file to the `logs/` directory inside the active workspace root.

## Output File

- **Path:** `<workspace_root>/logs/<YYYY-MM-DD_HHmm>.md`
- **Format:** Markdown
- **Filename:** Current local time in `YYYY-MM-DD_HHmm` format (e.g. `2026-07-27_1724.md`)

## Steps

1. **Determine workspace root** — use the root of the currently active workspace.

2. **Determine current local time** — use the timestamp from the system or the last known local time from the conversation metadata.

3. **Create the `logs/` directory if it does not exist:**
   ```bash
   mkdir -p <workspace_root>/logs
   ```

4. **Summarize the conversation** covering:
   - **Overview** — what was worked on this session (1–3 sentences)
   - **Key Discussions** — numbered list of the major topics discussed, decisions made, bugs found and fixed
   - **Files Changed / Created** — a Markdown table with `File` and `Change` columns listing every file that was modified or created during the session
   - **Final Architecture Summary** — a code block or bullet list showing the current state of the system if applicable
   - **Key Takeaways** — a short numbered list of the most important lessons or design decisions

5. **Write the file** to `<workspace_root>/logs/<YYYY-MM-DD_HHmm>.md` using the `write_to_file` tool or a shell command.

6. **Confirm** by printing the path to the saved log.

## Log Template

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
