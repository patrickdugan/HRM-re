## Project Overview
This repo contains exploratory work around the Hierarchical Reasoning Model (HRM) and related experiments. It is used alongside:
- `C:\projects\GPTStoryworld` (storyworld factory + tooling)
- `D:\HRMStoryworlds_v2` (storyworld output target)

## Required Skills
- Use the `storyworld-building` skill for any storyworld JSON/spool/encounter edits or validation work in the GPTStoryworld repo.
- Use the `hrm-trainer` skill for any HRM training runs, long jobs, or resource-constrained pipelines.

## Work Guidelines
- Prefer existing scripts in `scripts/` or `tools/` over manual edits when available.
- Validate storyworld JSON with `scripts\sweepweave_validator.py` before and after changes (in GPTStoryworld).
- Keep changes minimal and reversible; document any assumptions in the response.

## Where Things Live
- HRM code and experiments: `C:\projects\HRM-re`
- Storyworld factory/tooling: `C:\projects\GPTStoryworld`
- Storyworld output folder: `D:\HRMStoryworlds_v2`

## Crash/Failure Triage
When a run fails, capture:
- exact command
- working directory
- last ~50 log lines or output file path
- timestamp and runtime duration
