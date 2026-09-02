# AI Audit Workspace

An audit trail and lightweight project-management backlog for AI agents working in this repository — tool-agnostic (Claude Code, Copilot, or any other agent can read/write it). It answers four questions an agent (or a human reviewing its work) needs answered without re-deriving them from git history:

1. **What did an agent do, when, and why?** → `fix-log/`
2. **What still needs doing?** → `todo/`
3. **What was decided, in conversation, about a specific TODO?** → `instructions/`
4. **Where's the transcript of a past conversation, and what was it about?** → `convo/`

## The one rule that matters at scale

This backlog is expected to grow past 100 items. **Never read every file to "catch up."** Every folder has an `INDEX.md` — a bounded table, one row per item — read that first. Open an individual item's file only when you already know its ID and need the full detail. Never `grep -r` or glob the whole tree looking for context; that's how an agent misses items or duplicates work.

## ID scheme

Every item gets a stable ID: `TODO-0001`, `FIX-0001`, `CONVO-0001` — four-digit, zero-padded, monotonically increasing per type, and never reused (even after archiving).

Always allocate the next ID with the script — never hand-guess it:

```bash
ai-audit/scripts/next_id.sh todo   # -> TODO-0007
ai-audit/scripts/next_id.sh fix    # -> FIX-0012
ai-audit/scripts/next_id.sh convo  # -> CONVO-0003
```

IDs are how the four folders cross-reference each other (a fix-log entry names the `todo_id` it closed, a TODO names its instructions file by matching filename, a convo row names related TODOs) — all of it stays plain-text-greppable on purpose.

## Layout

```
ai-audit/
├── README.md                 # this file — the schema
├── scripts/
│   └── next_id.sh             # allocates the next ID for a given type
├── fix-log/
│   ├── INDEX.md               # append-only table, one row per fix
│   ├── TEMPLATE.md
│   └── YYYY/MM/*.md           # one file per fix, partitioned by month
├── todo/
│   ├── INDEX.md               # ACTIVE todos only — stays short forever
│   ├── TEMPLATE.md
│   ├── TODO-XXXX.md           # one file per active todo
│   └── archive/
│       ├── INDEX.md           # done/cancelled todos move here
│       └── TODO-XXXX.md
├── instructions/
│   ├── TEMPLATE.md
│   └── TODO-XXXX.md           # filename == the TODO it belongs to
└── convo/
    └── INDEX.md                # routing table to saved conversation transcripts
```

## 1. Fix-log — `fix-log/`

One file per fix, filed under `fix-log/YYYY/MM/`, named `YYYY-MM-DD_HHMM-<slug>.md`. Frontmatter:

```yaml
---
id: FIX-XXXX
timestamp: 2026-08-10T16:20:00+07:00
todo_id: TODO-XXXX   # optional — omit if this fix isn't tied to a backlog item
---
```

Body has exactly four sections, in this order: **Prompt** (what was asked), **Action** (what was actually done — commands, files touched), **Decision** (why this approach over alternatives), **Conclusion** (result, verification performed, follow-ups). See `TEMPLATE.md`.

Every entry also gets one row in `fix-log/INDEX.md`: `ID | Timestamp | Summary | TODO | File`. Append-only, newest at the bottom.

## 2. TODO backlog — `todo/`

One file per TODO: `todo/TODO-XXXX.md`. Frontmatter:

```yaml
---
id: TODO-XXXX
status: open          # open | in-progress | blocked | done | cancelled
source: user-report    # user-report | agent:<agent-name> | fix-log:FIX-XXXX
created: YYYY-MM-DD
updated: YYYY-MM-DD
priority: medium        # low | medium | high
tags: []
code_author: user       # who writes the implementation code: user | ai | both
ai_role: none           # none | advise | design | review | implement (combine with " + ")
---
```

Body: **Description** (what needs doing), **Requirement** (acceptance criteria / definition of done), **Notes** (pointer to its `instructions/` file). See `TEMPLATE.md`.

`code_author` / `ai_role` record **authorship** on every TODO — who actually wrote the code
vs. what the AI contributed. `code_author: user` with `ai_role: design + review` means the
user typed all the implementation and the AI only advised and reviewed; `code_author: ai`,
`ai_role: implement` means the AI wrote it. This is the authoritative attribution signal for
the work — the matching `fix-log` entry and any commit should agree with it.

`todo/INDEX.md` lists **active items only**: `ID | Status | Priority | Source | Short Description | Instruction | Updated`. When a TODO reaches `done` or `cancelled`, move its file *and* its INDEX row into `todo/archive/` — that's what keeps the live index short no matter how much history the project accumulates. IDs are never reused.

### Kanban / WIP limit

This project follows Kanban (continuous flow, no sprints) — the `status` field on each TODO *is* the board column. The only discipline that makes that work for a single developer is a **WIP limit: at most 2 TODOs in `in-progress` at once.** Before moving a third TODO to `in-progress`, finish or explicitly park one of the other two (back to `open`, or `blocked` with a note in its `instructions/` file on what it's waiting on). This is what prevents the backlog from silently accumulating a dozen half-started items — nothing else enforces it.

## 3. Instructions — `instructions/`

One file per TODO, filename matches the TODO ID exactly: `instructions/TODO-XXXX.md` — no separate index needed, the filename is the lookup. This is where session-by-session progress on a TODO gets recorded while talking to the user: append a new dated section each time (never delete prior ones):

```markdown
## 2026-08-10 14:32
**Discussed:** ...
**Decision:** ...
**Next step:** ...
```

An agent picking up a TODO should read `todo/TODO-XXXX.md` for *what*, then `instructions/TODO-XXXX.md` (if it exists) for *what's already been tried and decided*, before doing anything else.

## 4. Conversation index — `convo/` (local, gitignored)

`convo/INDEX.md` is a **routing table**, not a transcript store, and it is **not committed to git** — it references Claude Code's own native session transcripts, which live outside the repo at `~/.claude/projects/<project-slug>/<session-id>.jsonl` and are inherently machine-local (a teammate's clone has different session files). One row per session:

`ID | Date | Keywords / Topic | Transcript | Summary | Related TODO`

- **Transcript** — the native `.jsonl` path. This is the ground truth: Claude Code writes it automatically, verbatim, with zero extra tokens spent and zero risk of the agent mis-summarizing something. Get `<project-slug>` and `<session-id>` from the "Scratchpad Directory" path already given in the system prompt each session (`/tmp/claude-<uid>/<project-slug>/<session-id>/scratchpad`) — both segments are right there, no derivation needed.
- **Summary** — `-` by default. Only set when the user explicitly asks for a written recap (e.g. "write a full log"), which then gets saved as a curated Markdown write-up (e.g. `BasicML/logs/*.md` via the `write-logs` skill). Nice for humans to skim when it exists; never treat it as more authoritative than the transcript.

To find a past conversation, search the Keywords column against what the user is asking about, then open **Transcript** for the accurate raw record (loadable into a new session for context) or **Summary** for a quick human-readable recap if one exists. `/write-log` indexes by default — appending just the Transcript row, no rewrite, near-zero cost — and only produces a Summary write-up on explicit request.

## Workflow summary

- New ask from the user, or something an agent surfaced → new `TODO`.
- Working that TODO in conversation → append to its `instructions/` file.
- A concrete change gets made → new `fix-log` entry, linked to the `todo_id` if applicable.
- Session ends / gets saved → new `convo` row pointing at the saved transcript.
- TODO finished → flip `status`, move file + index row to `todo/archive/`.
