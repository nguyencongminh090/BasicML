---
id: FIX-0001
timestamp: 2026-09-02T16:35:26+07:00
todo_id: TODO-0002
---

## Prompt

"add authorize: me (user) and you (AI) in TODO to specify who code" — then
"update ai-audit/claude.md with authorize writing (code_author: user) for new format".

## Action

- Added `code_author` and `ai_role` frontmatter fields to `ai-audit/todo/TEMPLATE.md`.
- Documented the two fields in the TODO-frontmatter block of `ai-audit/README.md`, with a
  paragraph explaining them as the authoritative attribution signal.
- Set `code_author: user` / `ai_role: design + review` on `ai-audit/todo/TODO-0002.md`.
- (No `ai-audit/CLAUDE.md` exists — the schema lives in `README.md`, which is what was updated.)
- The `.claude/skills/ai-audit` skill does not enumerate TODO frontmatter fields, so no
  skill/`.agents` sync was needed.

## Decision

Two flat, greppable fields rather than a nested `authorship:` block — matches the existing
flat frontmatter style (`source`, `priority`) and keeps `grep code_author` trivial.
`ai_role` uses free combination with " + " (e.g. `design + review`) instead of a rigid enum
because AI involvement is genuinely multi-dimensional per TODO.

## Conclusion

Applied. New TODOs created from the template now carry authorship fields; TODO-0002 records
that the user wrote all regularization code and the assistant only advised/reviewed.
Follow-up: keep the field consistent with the eventual commit author and any fix-log entry
for the regularization implementation itself.
