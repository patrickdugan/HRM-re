import argparse
import json
import os
import random
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def resolve_path(root, path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_json(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    run_root = resolve_path(config_dir, cfg["paths"]["root"])

    storyworld_dir = os.path.join(run_root, cfg["paths"]["storyworld_dir"])
    dataset_dir = os.path.join(run_root, cfg["paths"]["dataset_dir"])
    ensure_dir(dataset_dir)

    world_path = resolve_path(storyworld_dir, cfg["storyworld"]["output_world"])
    encounters_path = resolve_path(storyworld_dir, cfg["storyworld"]["output_encounters"])

    if not os.path.exists(world_path) or not os.path.exists(encounters_path):
        raise FileNotFoundError("Missing storyworld artifacts. Run gym_generate_storyworld.py first.")

    rng = random.Random(cfg.get("seed", 7))
    encounters = list(read_jsonl(encounters_path))
    rng.shuffle(encounters)

    split = int(len(encounters) * (1.0 - cfg["dataset"]["eval_holdout_ratio"]))
    train_rows = encounters[:split]
    eval_rows = encounters[split:]

    train_out = resolve_path(dataset_dir, cfg["dataset"]["train_output"])
    eval_out = resolve_path(dataset_dir, cfg["dataset"]["eval_output"])

    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    manifest_path = os.path.join(run_root, "run_manifest.json")
    manifest = read_json(manifest_path)
    manifest["dataset"] = {
        "created_at": utc_now(),
        "train": train_out,
        "eval": eval_out,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {train_out} ({len(train_rows)} rows)")
    print(f"Wrote {eval_out} ({len(eval_rows)} rows)")


if __name__ == "__main__":
    main()
