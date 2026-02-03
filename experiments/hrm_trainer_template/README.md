# HRM Trainer Template

Minimal harness for HRM training runs with hard caps (Windows Job Objects) or WSL limits.

## Windows (Job Object)

```powershell
cd experiments/hrm_trainer_template
./run_trainer.ps1 -ConfigPath trainer_config.json
```

Safe mode:

```powershell
./run_trainer.ps1 -ConfigPath trainer_config_safe.json
```

Notes:
- RAM and CPU are hard-capped via Job Objects.
- IO caps are soft: the trainer aborts if measured IO exceeds the cap (requires `psutil`).
 - Install dependencies to enable IO monitoring: `psutil`.

## WSL (Optional)

```bash
cd /mnt/c/projects/HRM-re/experiments/hrm_trainer_template
bash run_trainer_wsl.sh trainer_config.json
```

Safe mode:

```bash
bash run_trainer_wsl.sh trainer_config_safe.json
```

Notes:
- If `systemd-run` is available in WSL, it applies `MemoryMax` and `CPUQuota`.
- Otherwise the script uses `ulimit -v` as a soft cap.
- For stable WSL memory control, set a `.wslconfig` cap in Windows user profile.
- See `wslconfig.template` for a minimal example.

## Outputs

The stub trainer writes:
- `outputs/<run_id>/events.jsonl`
- `outputs/<run_id>/summary.json`
- `outputs/<run_id>/checkpoints/`

## Customize

Edit `trainer_config.json` (paths are resolved relative to the config file):
- `max_ram_mb`, `max_cpu_pct`, `max_io_mb_s`
- `checkpoint_every_steps`, `checkpoint_every_seconds`
- `chunk_strategy`, `chunk_size`
- `max_steps`, `max_runtime_seconds`
- `train.data`, `train.out`, `train.epochs`, `train.batch_size`, `train.lr`, `train.device`
- `train.type` (`conductor`, `storyworld`, or `custom`)
- `train.module`, `train.entry` (for `custom`)
- `train.resume_checkpoint`
- `train.resume_latest`

Resume example:

```json
"train": {
  "type": "conductor",
  "resume_checkpoint": "outputs/hrm-template-001/checkpoints/step-100.pt"
}
```

Resume latest example:

```json
"train": {
  "type": "conductor",
  "resume_latest": true
}
```

Recommendation: set `resume_latest: true` for crash recovery during long runs.

## Custom trainer backend

Set in config:

```json
"train": {
  "type": "custom",
  "module": "my_package.my_trainer",
  "entry": "main"
}
```

The custom entry must accept the same CLI args as the conductor trainer.

Replace `train_stub.py` with your real HRM training loop but keep the same logging and checkpointing contract.
