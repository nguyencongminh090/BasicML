---
description: Index this session in ai-audit/convo (native transcript, no rewrite)
---

Use the `write-logs` skill's default path: append a row to `ai-audit/convo/INDEX.md` pointing at this session's native `.jsonl` transcript. Only write a curated Markdown recap under `BasicML/logs/` if the user's message explicitly asked for a written/detailed log — otherwise skip it.
