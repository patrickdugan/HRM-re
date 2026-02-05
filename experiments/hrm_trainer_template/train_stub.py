import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_dry_run(events_path, summary_path, max_steps, run_id):
    start = time.time()
    for step in range(1, max_steps + 1):
        time.sleep(0.01)
        if step % 50 == 0:
            append_jsonl(events_path, {"ts": utc_now(), "event": "checkpoint", "step": step})
    summary = {
        "run_id": run_id,
        "status": "completed",
        "abort_reason": None,
        "steps_completed": max_steps,
        "wall_seconds": round(time.time() - start, 3),
    }
    write_json(summary_path, summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = cfg.get("run_id", "hrm-run")
    output_dir = cfg.get("output_dir", "outputs")
    logs_dir = os.path.join(output_dir, run_id)
    ensure_dir(logs_dir)

    events_path = os.path.join(logs_dir, "events.jsonl")
    summary_path = os.path.join(logs_dir, "summary.json")
    checkpoint_dir = os.path.join(logs_dir, "checkpoints")

    append_jsonl(
        events_path,
        {
            "ts": utc_now(),
            "event": "start",
            "run_id": run_id,
            "caps": {
                "ram_mb": cfg.get("max_ram_mb"),
                "cpu_pct": cfg.get("max_cpu_pct"),
                "io_mb_s": cfg.get("max_io_mb_s"),
            },
        },
    )

    config_dir = os.path.dirname(os.path.abspath(args.config))
    train_cfg = cfg.get("train", {})

    def resolve_path(value):
        if not value:
            return value
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(config_dir, value))

    out_path = resolve_path(train_cfg.get("out"))
    if out_path:
        ensure_dir(os.path.dirname(out_path))
    data_path = resolve_path(train_cfg.get("data"))
    if not data_path or not out_path:
        append_jsonl(
            events_path,
            {
                "ts": utc_now(),
                "event": "abort",
                "reason": "missing_train_paths",
                "details": {"data": data_path, "out": out_path},
            },
        )
        write_json(summary_path, {"status": "failed", "abort_reason": "missing_train_paths"})
        sys.exit(1)
    if cfg.get("dry_run", False):
        run_dry_run(events_path, summary_path, int(cfg.get("max_steps", 100)), run_id)
        append_jsonl(events_path, {"ts": utc_now(), "event": "end", "status": "completed"})
        return

    train_type = train_cfg.get("type", "conductor")
    auto_resume = bool(train_cfg.get("resume_latest", False))

    if train_type == "conductor":
        module_name = "experiments.storyworld_conductor_hrm.train_hrm_conductor"
        entry_name = "main"
    elif train_type == "storyworld":
        module_name = "experiments.storyworld_secret_ending.train_hrm_storyworld"
        entry_name = "main"
    elif train_type == "custom":
        module_name = train_cfg.get("module")
        entry_name = train_cfg.get("entry", "main")
    else:
        append_jsonl(
            events_path,
            {"ts": utc_now(), "event": "abort", "reason": "unknown_train_type", "type": train_type},
        )
        write_json(summary_path, {"status": "failed", "abort_reason": "unknown_train_type"})
        sys.exit(1)

    if not module_name:
        append_jsonl(events_path, {"ts": utc_now(), "event": "abort", "reason": "missing_train_module"})
        write_json(summary_path, {"status": "failed", "abort_reason": "missing_train_module"})
        sys.exit(1)

    try:
        trainer_module = importlib.import_module(module_name)
        trainer = getattr(trainer_module, entry_name)
    except Exception as exc:
        append_jsonl(events_path, {"ts": utc_now(), "event": "abort", "reason": "import_failed", "error": str(exc)})
        write_json(summary_path, {"status": "failed", "abort_reason": "import_failed"})
        sys.exit(1)

    argv = [
        "--data",
        data_path,
        "--out",
        out_path,
        "--epochs",
        str(train_cfg.get("epochs", 1)),
        "--batch_size",
        str(train_cfg.get("batch_size", 8)),
        "--lr",
        str(train_cfg.get("lr", 2e-4)),
        "--seed",
        str(train_cfg.get("seed", 7)),
        "--dim",
        str(train_cfg.get("dim", 128)),
        "--depth",
        str(train_cfg.get("depth", 2)),
        "--heads",
        str(train_cfg.get("heads", 4)),
        "--max_examples",
        str(train_cfg.get("max_examples", 0)),
        "--device",
        str(train_cfg.get("device", "cpu")),
        "--fp16" if train_cfg.get("fp16") else None,
        "--checkpoint_dir",
        checkpoint_dir,
        "--checkpoint_every_steps",
        str(cfg.get("checkpoint_every_steps", 0)),
        "--checkpoint_every_seconds",
        str(cfg.get("checkpoint_every_seconds", 0)),
        "--max_steps",
        str(cfg.get("max_steps", 0)),
        "--max_runtime_seconds",
        str(cfg.get("max_runtime_seconds", 0)),
        "--io_mb_s",
        str(cfg.get("max_io_mb_s", 0)),
        "--io_grace_seconds",
        str(cfg.get("io_grace_seconds", 5.0)),
        "--resume_checkpoint",
        str(resolve_path(train_cfg.get("resume_checkpoint")) or ""),
        "--resume_latest" if auto_resume else None,
        "--log_jsonl",
        events_path,
        "--summary_json",
        summary_path,
    ]

    argv = [a for a in argv if a is not None]
    sys.argv = [sys.argv[0]] + argv
    trainer()


if __name__ == "__main__":
    main()
