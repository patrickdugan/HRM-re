import argparse
import json
import os
import subprocess
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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

    dataset_dir = os.path.join(run_root, cfg["paths"]["dataset_dir"])
    model_dir = os.path.join(run_root, cfg["paths"]["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    train_config_path = resolve_path(config_dir, cfg["train"]["config_path"])
    if not os.path.exists(train_config_path):
        raise FileNotFoundError(f"Missing trainer config: {train_config_path}")

    train_cfg = read_json(train_config_path)
    train_cfg = dict(train_cfg)

    train_cfg["run_id"] = cfg["run_id"] + "-player"
    train_cfg["output_dir"] = model_dir
    train_cfg["train"]["type"] = cfg["train"].get("type", "storyworld")

    train_cfg["train"]["data"] = os.path.join(dataset_dir, cfg["dataset"]["train_output"])
    train_cfg["train"]["out"] = os.path.join(model_dir, "player_hrm.pt")

    tmp_cfg_path = os.path.join(run_root, "train_player_config.json")
    write_json(tmp_cfg_path, train_cfg)

    cmd = ["python", os.path.join(config_dir, "..", "hrm_trainer_template", "train_stub.py"), "--config", tmp_cfg_path]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    manifest_path = os.path.join(run_root, "run_manifest.json")
    manifest = read_json(manifest_path)
    manifest["train_player"] = {
        "created_at": utc_now(),
        "train_config": tmp_cfg_path,
        "model": os.path.join(model_dir, "player_hrm.pt"),
    }
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
