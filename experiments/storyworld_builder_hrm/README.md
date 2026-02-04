# Storyworld Builder HRM

Build a dataset from Sweepweave storyworlds + generated storyworlds and train a planner-style HRM.

## Build dataset

```powershell
python experiments/storyworld_builder_hrm/make_builder_dataset.py --input C:/projects/GPTStoryworld/storyworlds --input D:/HRMStoryworlds/verified --out D:/HRMStoryworlds/datasets/builder_dataset.jsonl --max_encounters 0 --max_options 0 --max_reactions 0
```

## Train

```powershell
python experiments/hrm_trainer_template/train_stub.py --config experiments/storyworld_builder_hrm/builder_train_config.json
```
