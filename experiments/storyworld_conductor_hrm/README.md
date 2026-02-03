# Storyworld Conductor HRM (2-Tier)

This experiment scaffolds a two-tier HRM "conductor" that decides which storyworld-building action to take next, using the verifiers environment metrics as state. The idea is to have the HRM sit above the actual builder (Codex + tools) and orchestrate:

- validation
- spool repair/materialization
- ending reachability tuning
- secret logic / effects enrichment

The H-tier (planner) selects the next action; the L-tier (operator) executes that action (via tools or a human/Codex). This is a realistic conductor architecture: the HRM does not have to write full storyworlds, it has to choose the right operations at the right time.

## Files
- `conductor_env.py`: analyzes a storyworld JSON using the verifiers environment metrics.
- `make_conductor_dataset.py`: builds a JSONL dataset of (state -> action) examples from existing storyworlds.
- `train_hrm_conductor.py`: trains an HRM to predict the next action.
- `benchmark_harness.py`: runs an orchestration loop and reports progress.
- `player_harness.py`: runs a playthrough benchmark (random, greedy, HRM player).
- `run_full_bench.py`: runs builder + player harness across a folder.

## Generate a dataset

```powershell
python .\make_conductor_dataset.py --out .\data_conductor.jsonl --max_examples 50
```

## Train the conductor HRM

```powershell
python .\train_hrm_conductor.py --data .\data_conductor.jsonl --out .\models_conductor.pt --epochs 5 --batch_size 16 --lr 2e-4
```

## Run a single analysis

```powershell
python .\conductor_env.py "C:\projects\GPTStoryworld\storyworlds\mihna_constitutional_alignment.json"
```

## Benchmark harness (orchestration loop)

Run a bounded orchestration loop that applies actions and re-scores the storyworld:

```powershell
python .\benchmark_harness.py --storyworld "C:\projects\GPTStoryworld\storyworlds\mihna_constitutional_alignment.json" --steps 6 --mc_runs 200 --out .\reports\mihna_benchmark.json
```

To use the trained HRM instead of heuristics:

```powershell
python .\benchmark_harness.py --storyworld "C:\projects\GPTStoryworld\storyworlds\mihna_constitutional_alignment.json" --steps 6 --mc_runs 200 --hrm_model .\models_conductor.pt --out .\reports\mihna_benchmark_hrm.json
```

## Player harness

```powershell
python .\player_harness.py --storyworld "C:\projects\GPTStoryworld\storyworlds\mihna_constitutional_alignment.json" --episodes 30 --max_turns 25 --out .\reports\mihna_player.json
```

## Full benchmark (builder + player)

```powershell
python .\run_full_bench.py --out .\reports\full_index.json --steps 6 --mc_runs 200 --episodes 30 --max_turns 25
```

## Player HRM dataset + training

```powershell
python .\make_player_dataset.py --storyworld_dir "C:\projects\GPTStoryworld\storyworlds" --episodes_per_storyworld 100 --out .\data_player.jsonl
python ..\storyworld_secret_ending\train_hrm_storyworld.py --data .\data_player.jsonl --out .\models_player.pt --epochs 3 --batch_size 8 --lr 2e-4 --dim 64 --depth 1 --heads 1
```

Action mapping (generic defaults):
- `SCHEMA_PATCH` -> fill missing `pronoun` and `depth` defaults
- `MATERIALIZE_SPOOLS` -> materialize missing Age spools
- `REPAIR_CONNECTED_SPOOLS` -> fix `connected_spools` by Age title
- `TUNE_ENDING_GATES` -> apply small desirability bias to endings
- `ENDING_REACHABILITY` -> apply smaller desirability bias
- `ADD_SECRET_LOGIC` -> mark a first option as secret
- `ADD_EFFECTS` -> add a placeholder effect to the first reaction

The harness always validates with `sweepweave_validator.py` before and after edits.

## Next steps
- Replace `suggest_action()` heuristics with the trained HRM policy.
- Wire the actions to actual tool executions (scripts in `C:\projects\GPTStoryworld\codex-skills\storyworld-building\scripts`).
- Add a research action that uses Codex to expand thematic content before tool application.
