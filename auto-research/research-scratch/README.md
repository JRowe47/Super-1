# Research scratch

Start each continuation with [CURRENT.md](CURRENT.md). This is the stable channel for research notes, decisions, small code changes and the next concrete action.

Use this directory on the existing research branch in **JRowe47/Super-1**. Keep completed run evidence under `auto-research/results/<run_id>/` and reports under `auto-research/reports/`, with links in `auto-research/RUN_REGISTRY.json`. The current branch is `research/curve-inception-20261017-v1`, attached to draft PR #3.

## Storage policy

The user requested this workflow on 2026-10-17 America/Denver to avoid relying on chat attachments and transient workspaces.

- Keep notes, code, protocols, small JSON results and precise reproduction instructions in Git.
- Save source and the frozen protocol before starting a new run; save concise results and status promptly.
- Store necessary larger artifacts once in the existing **Geometric Self Attention** Library folder or an established repository download. Reference the exact filename, version, size and hash when available.
- Use the existing `/Geometric Self Attention/` folder for current companion files. The older expanded project is under `/Spaces/Geometric Self Attention/`.
- Prefer changed files and small per-run records over cumulative archives. Do not duplicate old bundles, datasets, runtimes or every checkpoint inside new handoffs.
- Keep enough source, seed/configuration information and final/failure states for replay. Preserve completed evidence; reducing future output size does not authorize deleting sealed old runs.
- After each session, update CURRENT.md with what finished, what failed, what remains unpreserved, and the next action. Distinguish original frozen files from later transcriptions.

Read [the research contract](../../auto-research.md) before new experiments. This folder does not itself launch jobs or expand compute authorization.
