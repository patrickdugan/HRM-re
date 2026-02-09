import argparse
import json
import os
import random
import shutil
import subprocess
from datetime import datetime, timedelta


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
    cfg["seed"] = random.randint(0, 2**32 - 1)
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
    progress_interval_minutes = int(cfg.get("progress_interval_minutes", 30))
    trigger_verified = int(cfg.get("trigger_verified", 0))
    trigger_command = cfg.get("trigger_command", "")
    lock_file = cfg.get("lock_file", "batch_lock.json")
    force_lock = bool(cfg.get("force_lock", False))
    idea_factory_dir = cfg.get("idea_factory_dir", "")
    batch_prompt_file = cfg.get("batch_prompt_file", "batch_prompt.txt")
    batch_prompt_max_chars = int(cfg.get("batch_prompt_max_chars_per_file", 1200))
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

    lock_path = os.path.join(base_dir, lock_file)
    if os.path.exists(lock_path) and not force_lock:
        print(f"Lock exists: {lock_path}. Another batch may be running.")
        return
    write_json(
        lock_path,
        {
            "pid": os.getpid(),
            "started_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    )

    batch_prompt_path = os.path.join(base_dir, batch_prompt_file)
    if idea_factory_dir and not os.path.exists(batch_prompt_path):
        tropes_dir = os.path.join(idea_factory_dir, "tropes")
        sources_dir = os.path.join(idea_factory_dir, "sources")
        parts = []
        if os.path.isdir(tropes_dir):
            for name in sorted(os.listdir(tropes_dir)):
                if name.endswith(".md"):
                    with open(os.path.join(tropes_dir, name), "r", encoding="utf-8", errors="ignore") as f:
                        parts.append(f.read()[:batch_prompt_max_chars])
        if os.path.isdir(sources_dir):
            for name in sorted(os.listdir(sources_dir)):
                if name.endswith(".md"):
                    with open(os.path.join(sources_dir, name), "r", encoding="utf-8", errors="ignore") as f:
                        parts.append(f.read()[:batch_prompt_max_chars])
        if parts:
            with open(batch_prompt_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(parts).strip() + "\n")

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_config.json")

    verified_count = len([f for f in os.listdir(verified_dir) if f.endswith(".json")])
    if start_index <= 0:
        existing = []
        for name in os.listdir(runs_dir):
            if name.startswith("run_"):
                try:
                    existing.append(int(name.replace("run_", "")))
                except ValueError:
                    continue
        idx = (max(existing) + 1) if existing else 1
    else:
        idx = start_index
    attempts = 0
    end_time = None
    if end_time_local:
        end_time = datetime.fromisoformat(end_time_local)
    next_report = datetime.now() + timedelta(minutes=progress_interval_minutes)
    trigger_fired = False

    def write_progress(final=False):
        percent = round((verified_count / target_verified) * 100, 2) if target_verified else 0.0
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "verified": verified_count,
            "target": target_verified,
            "percent": percent,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "end_time_local": end_time_local,
            "final": final,
        }
        write_json(os.path.join(logs_dir, "progress.json"), payload)
        with open(os.path.join(logs_dir, "progress.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2))

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
        if os.path.exists(batch_prompt_path):
            run_cfg = read_json(run_cfg_path)
            run_cfg["storyworld"]["prompt_preamble_file"] = batch_prompt_path
            write_json(run_cfg_path, run_cfg)

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
                    if trigger_verified and not trigger_fired and verified_count >= trigger_verified and trigger_command:
                        try:
                            run(["cmd", "/c", trigger_command])
                            trigger_fired = True
                        except subprocess.CalledProcessError as exc:
                            log_path = os.path.join(logs_dir, "trigger_error.txt")
                            with open(log_path, "w", encoding="utf-8") as f:
                                f.write(str(exc))
            elif verifier.get("status") == "failed":
                world_path = os.path.join(run_root, "storyworld", "world.json")
                dest = os.path.join(rejected_dir, f"{run_id}.json")
                copy_if_exists(world_path, dest)

        attempts += 1
        idx += 1
        if datetime.now() >= next_report:
            write_progress(final=False)
            next_report = datetime.now() + timedelta(minutes=progress_interval_minutes)

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
    write_progress(final=True)
    if os.path.exists(lock_path):
        os.remove(lock_path)


if __name__ == "__main__":
    main()
