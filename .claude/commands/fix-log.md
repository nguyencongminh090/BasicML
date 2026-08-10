---
description: Record a fix-log entry for the change just made in this session
---

Use the `ai-audit` skill's "Applying a concrete fix" fast path: allocate an ID with `ai-audit/scripts/next_id.sh fix`, create the dated entry under `ai-audit/fix-log/YYYY/MM/` from `ai-audit/fix-log/TEMPLATE.md`, and add one row to `ai-audit/fix-log/INDEX.md`. Cover Prompt, Action, Decision, and Conclusion for what was actually done in this session, and link `todo_id` if this fix closes or advances a tracked TODO.
