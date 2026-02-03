# Storyworld Conductor HRM (2-Tier)

This experiment scaffolds a two-tier HRM “conductor” that decides **which storyworld-building action to take next**, using the verifiers environment metrics as state. The idea is to have the HRM sit above the actual builder (Codex + tools) and orchestrate:

- validation
- spool repair/materialization
- ending reachability tuning
- secret logic / effects enrichment

The **H-tier** (planner) selects the next action; the **L-tier** (operator) executes that action (via tools or a human/Codex). This is a realistic “conductor” architecture: the HRM doesn’t have to write full storyworlds, it has to choose the right operations at the right time.

## Files
- `conductor_env.py`: analyzes a storyworld JSON using the verifiers environment metrics.
- `make_conductor_dataset.py`: builds a JSONL dataset of `(state -> action)` examples from existing storyworlds.
- `train_hrm_conductor.py`: trains an HRM to predict the next action.

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

## Next steps
- Replace `suggest_action()` heuristics with the trained HRM policy.
- Wire the actions to actual tool executions (scripts in `C:\projects\GPTStoryworld\codex-skills\storyworld-building\scripts`).
- Add a “research” action that uses Codex to expand thematic content before tool application.
