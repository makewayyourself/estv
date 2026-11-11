# Agent skills (find-skills)

- `find-skills` is installed at `.claude/skills/find-skills` (source pinned in `skills-lock.json`).
- In Claude Code cloud sessions, prefix skills CLI commands with `NODE_USE_ENV_PROXY=1`
  (e.g. `NODE_USE_ENV_PROXY=1 npx skills find pptx`). Without it the CLI cannot reach
  skills.sh through the session proxy and prints "No skills found" for every query.
  The environment must also allow the host `skills.sh`.
- Review a skill's `SKILL.md` and scripts before installing it; skills run with full agent permissions.
