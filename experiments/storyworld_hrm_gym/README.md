# Storyworld HRM Gym (Scripts)

This folder contains a minimal, staged pipeline for the storyworld HRM gym.
It writes a prompt file that explicitly references the `storyworld-building`
skill. You can run that skill manually, or enable the optional Codex CLI hook.

## Stages

1. `gym_generate_storyworld.py`
   - Writes `prompts/storyworld_prompt.txt` and a `run_manifest.json` stub.
   - Run the storyworld builder skill to produce `world.json` and `encounters.jsonl`.

2. `gym_make_dataset.py`
   - Splits `encounters.jsonl` into train/eval JSONL.

3. `gym_train_player.py`
   - Invokes the HRM trainer harness using `train_stub.py`.

4. `gym_eval.py`
   - Evaluates the trained player HRM and writes accuracy and loss to `metrics.json`.

5. `gym_run.py`
   - Orchestrates all steps; still requires manual storyworld generation.

## Config

Edit `gym_config.json`. Paths are relative to the config file unless absolute.
If you want the runner to wait for artifacts, set `wait_for_artifacts: true`.
Verifier is enabled by default; it blocks training until artifacts are validated.

## Optional Codex CLI hook

If you have the official `codex` CLI authenticated, you can enable automatic
execution by setting `codex_exec.enabled: true` and providing the appropriate
flags. The prompt will still reference the `storyworld-building` skill.
The wrapper enforces a wall-clock timeout and classifies timeouts as aborted.

Example:

```json
"codex_exec": {
  "enabled": true,
  "mode": "full-auto",
  "sandbox": "danger-full-access",
  "timeout_seconds": 900,
  "output_last_message": "prompts/codex_last_message.txt",
  "working_dir": ""
}
```

## MLflow (optional)

Use `mlflow_interface.py` to optionally log params/metrics/artifacts. Add
`mlflow` as a dependency only if you enable it.
