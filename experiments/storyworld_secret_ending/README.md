# Storyworld Secret Ending Experiment

Minimal, runnable harness to train and evaluate an HRM on secret-ending discovery within a turn budget.

## Quick Start
Run from `C:\projects\HRM-re\experiments\storyworld_secret_ending`:

```
python -m pip install -e ..\..

python .\make_dataset.py --storyworld .\data\mihna_min_storyworld.json --target_ending_id ending_secret_created_quran_accepted --out .\data\mihna_min_dataset.jsonl --episodes 500 --max_turns 10

python .\train_hrm_storyworld.py --data .\data\mihna_min_dataset.jsonl --out .\models\hrm_mihna_min.pt --epochs 3 --batch_size 16 --lr 2e-4 --dim 128

python .\eval_storyworld.py --storyworld .\data\mihna_min_storyworld.json --target_ending_id ending_secret_created_quran_accepted --max_turns 10 --episodes 200 --hrm_model .\models\hrm_mihna_min.pt --report .\reports\results.json --gpt_mock
```

## Mihna Constitutional Alignment Storyworld
Use the existing storyworld JSON with the reader-aligned engine:

```
python .\make_dataset.py --storyworld "C:\projects\GPTStoryworld\storyworlds\mihna_constitutional_alignment.json" --target_ending_id page_end_compliant --out .\data\mihna_constitutional_alignment.jsonl --episodes 500 --max_turns 25

python .\train_hrm_storyworld.py --data .\data\mihna_constitutional_alignment.jsonl --out .\models\hrm_mihna_constitutional_alignment.pt --epochs 3 --batch_size 16 --lr 2e-4 --dim 128

python .\eval_storyworld.py --storyworld "C:\projects\GPTStoryworld\storyworlds\mihna_constitutional_alignment.json" --target_ending_id page_end_compliant --max_turns 25 --episodes 200 --hrm_model .\models\hrm_mihna_constitutional_alignment.pt --report .\reports\results_mihna.json --gpt_mock
```

Note: with the current reader-aligned logic, only `page_end_compliant` is directly reachable; \"secret\" success is defined as reaching `page_end_compliant` after choosing any option whose id contains `_secret_`.
Fast smoke test: add `--max_examples 200 --dim 32 --depth 1 --heads 1` to `train_hrm_storyworld.py`.
If you want to cap GPT runs (to control cost/latency), add `--gpt_episodes 20`.

## Outputs
- Dataset: `experiments/storyworld_secret_ending/data/mihna_min_dataset.jsonl`
- Model: `experiments/storyworld_secret_ending/models/hrm_mihna_min.pt`
- Report: `experiments/storyworld_secret_ending/reports/results.json`

## Notes / TODOs
- GPT-5 mini: API call is wired in `experiments/storyworld_secret_ending/baselines/baseline_gpt5mini.py` (set `OPENAI_API_KEY`).
- Tiny open model: replace stub logic in `experiments/storyworld_secret_ending/baselines/baseline_tiny_open.py` with a real local model if/when weights are available.
