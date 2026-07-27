# Agent Rules

## Git Commit Workflow
When the user says "MAKE COMMIT", you must:
1. Trigger the appropriate WORKFLOW.
2. Use the installed Git Management SKILLS (like `github/awesome-copilot@git-commit`) to make the commit.

## Write Log Workflow
When the user says "WRITE LOG", "write log", "save log", or "summary log", you must:
1. Read the `write-logs` SKILL from `.agents/skills/write-logs/SKILL.md`.
2. Follow the skill instructions exactly to summarize the conversation and save the log.

## Markdown Rendering Guidelines
When writing Markdown files containing LaTeX math (e.g., `README.md`), always follow these rules to ensure the IDE parser renders them correctly:
1. **Avoid inline math (`$...$`) for complex equations**: The parser often breaks when inline math contains combinations of parentheses and superscripts (e.g., `(\hat{y} - y)^2`).
2. **Use block math (`$$...$$`) instead**: For equations containing superscripts, subscripts, or sum symbols, use block math.
3. **Always surround block math with empty lines**: If a `$$` block is placed immediately after text (e.g., directly under a list item or paragraph without an empty line in between), the Markdown parser will merge them into a single text paragraph and fail to trigger the math renderer.
4. **Avoid indenting math blocks inside lists**: Break out of bulleted lists before defining complex block equations to prevent list-parsing bugs.
