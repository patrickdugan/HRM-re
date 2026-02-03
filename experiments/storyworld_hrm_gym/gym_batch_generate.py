import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def build_run_config(template_path, out_path, run_root, run_id, codex_enabled, wait_for_artifacts, timeout_seconds):
    cfg = read_json(template_path)
    cfg["run_id"] = run_id
    cfg["paths"]["root"] = run_root
    cfg["codex_exec"]["enabled"] = codex_enabled
    cfg["codex_exec"]["timeout_seconds"] = timeout_seconds
    cfg["wait_for_artifacts"] = wait_for_artifacts
    write_json(out_path, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_json(args.config)
    base_dir = cfg["base_dir"]
    target_verified = int(cfg.get("target_verified", 200))
    max_attempts = int(cfg.get("max_attempts", 0))
    max_size_bytes = int(cfg.get("max_size_bytes", 0))
    end_time_local = cfg.get("end_time_local")
    start_index = int(cfg.get("start_index", 1))
    codex_enabled = bool(cfg.get("codex_enabled", True))
    wait_for_artifacts = bool(cfg.get("wait_for_artifacts", True))
    timeout_seconds = int(cfg.get("timeout_seconds", 900))

    runs_dir = os.path.join(base_dir, "runs")
    verified_dir = os.path.join(base_dir, "verified")
    rejected_dir = os.path.join(base_dir, "rejected")
    logs_dir = os.path.join(base_dir, "logs")
    ensure_dir(runs_dir)
    ensure_dir(verified_dir)
    ensure_dir(rejected_dir)
    ensure_dir(logs_dir)

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_config.json")

    verified_count = len([f for f in os.listdir(verified_dir) if f.endswith(".json")])
    idx = start_index
    attempts = 0
    end_time = None
    if end_time_local:
        end_time = datetime.fromisoformat(end_time_local)

    while verified_count < target_verified:
        if max_attempts and attempts >= max_attempts:
            break
        if end_time and datetime.now() >= end_time:
            break

        run_id = f"run_{idx:04d}"
        run_root = os.path.join(runs_dir, run_id)
        ensure_dir(run_root)

        run_cfg_path = os.path.join(run_root, "gym_config.json")
        build_run_config(template_path, run_cfg_path, run_root, run_id, codex_enabled, wait_for_artifacts, timeout_seconds)

        try:
            run(["python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_run.py"), "--config", run_cfg_path])
        except subprocess.CalledProcessError as exc:
            log_path = os.path.join(logs_dir, f"{run_id}_error.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(str(exc))

        manifest_path = os.path.join(run_root, "run_manifest.json")
        if os.path.exists(manifest_path):
            manifest = read_json(manifest_path)
            verifier = manifest.get("verifier", {})
            if verifier.get("status") == "verified":
                world_path = os.path.join(run_root, "storyworld", "world.json")
                if max_size_bytes and os.path.exists(world_path) and os.path.getsize(world_path) > max_size_bytes:
                    dest = os.path.join(rejected_dir, f"{run_id}.json")
                    copy_if_exists(world_path, dest)
                else:
                    dest = os.path.join(verified_dir, f"{run_id}.json")
                    copy_if_exists(world_path, dest)
                    verified_count += 1
            elif verifier.get("status") == "failed":
                world_path = os.path.join(run_root, "storyworld", "world.json")
                dest = os.path.join(rejected_dir, f"{run_id}.json")
                copy_if_exists(world_path, dest)

        attempts += 1
        idx += 1

    summary_path = os.path.join(base_dir, "batch_summary.json")
    write_json(
        summary_path,
        {
            "completed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "verified": verified_count,
            "target": target_verified,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "end_time_local": end_time_local,
            "max_size_bytes": max_size_bytes,
        },
    )


if __name__ == "__main__":
    main()
