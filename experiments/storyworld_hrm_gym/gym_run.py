import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone

from mlflow_interface import maybe_init_mlflow, log_params, log_metrics, log_artifact


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


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def artifacts_exist(storyworld_dir, world_name, encounters_name):
    world_path = os.path.join(storyworld_dir, world_name)
    encounters_path = os.path.join(storyworld_dir, encounters_name)
    return os.path.exists(world_path) and os.path.exists(encounters_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_json(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    run_root = resolve_path(config_dir, cfg["paths"]["root"])

    os.makedirs(run_root, exist_ok=True)

    manifest_path = os.path.join(run_root, "run_manifest.json")
    if not os.path.exists(manifest_path):
        write_json(
            manifest_path,
            {
                "run_id": cfg["run_id"],
                "seed": cfg["seed"],
                "created_at": utc_now(),
                "paths": cfg["paths"],
            },
        )

    mlflow = maybe_init_mlflow(cfg.get("mlflow", {}))
    if mlflow:
        mlflow.start_run(run_name=cfg.get("run_id"))
        log_params(mlflow, {"run_id": cfg.get("run_id"), "seed": cfg.get("seed")})

    run(["python", os.path.join(config_dir, "gym_generate_storyworld.py"), "--config", args.config])
    print("Step complete: storyworld prompt written. Generate artifacts, then continue.")

    storyworld_dir = os.path.join(run_root, cfg["paths"]["storyworld_dir"])
    world_name = cfg["storyworld"]["output_world"]
    encounters_name = cfg["storyworld"]["output_encounters"]

    if not artifacts_exist(storyworld_dir, world_name, encounters_name):
        if cfg.get("wait_for_artifacts", False):
            timeout = int(cfg.get("wait_timeout_seconds", 900))
            poll = int(cfg.get("wait_poll_seconds", 10))
            deadline = time.time() + timeout
            print("Waiting for storyworld artifacts...")
            while time.time() < deadline:
                if artifacts_exist(storyworld_dir, world_name, encounters_name):
                    break
                time.sleep(poll)
            if not artifacts_exist(storyworld_dir, world_name, encounters_name):
                raise FileNotFoundError("Timed out waiting for storyworld artifacts.")
        else:
            print("Artifacts not found yet. Run the storyworld builder and re-run gym_run.py.")
            return

    if cfg.get("verifier", {}).get("enabled", True):
        verify_cmd = ["python", os.path.join(config_dir, "gym_verify_storyworld.py"), "--config", args.config]
        print("Running:", " ".join(verify_cmd))
        verify_proc = subprocess.run(verify_cmd, capture_output=True, text=True)
        if verify_proc.returncode != 0:
            print("Verifier failed:", verify_proc.stdout.strip())
            manifest_path = os.path.join(run_root, "run_manifest.json")
            manifest = read_json(manifest_path)
            manifest["verifier"] = {"status": "failed", "details": verify_proc.stdout.strip()}
            write_json(manifest_path, manifest)
            return
        else:
            manifest_path = os.path.join(run_root, "run_manifest.json")
            manifest = read_json(manifest_path)
            manifest["verifier"] = {"status": "verified", "details": verify_proc.stdout.strip()}
            write_json(manifest_path, manifest)

    run(["python", os.path.join(config_dir, "gym_make_dataset.py"), "--config", args.config])
    run(["python", os.path.join(config_dir, "gym_train_player.py"), "--config", args.config])
    run(["python", os.path.join(config_dir, "gym_eval.py"), "--config", args.config])

    if mlflow:
        manifest_path = os.path.join(run_root, "run_manifest.json")
        log_artifact(mlflow, manifest_path)
        metrics_path = os.path.join(run_root, cfg["paths"]["report_dir"], "metrics.json")
        if os.path.exists(metrics_path):
            metrics = read_json(metrics_path)
            log_metrics(mlflow, {"eval_count": metrics.get("eval_count", 0)})
            log_artifact(mlflow, metrics_path)
        mlflow.end_run()


if __name__ == "__main__":
    main()
