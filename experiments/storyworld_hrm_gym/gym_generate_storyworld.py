import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(root, path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def run_with_timeout(cmd, cwd=None, timeout_seconds=None):
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=cwd)
    try:
        proc.wait(timeout=timeout_seconds)
        return proc.returncode, None
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return None, "codex_timeout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_json(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    run_root = resolve_path(config_dir, cfg["paths"]["root"])

    storyworld_dir = os.path.join(run_root, cfg["paths"]["storyworld_dir"])
    prompts_dir = os.path.join(run_root, "prompts")
    ensure_dir(storyworld_dir)
    ensure_dir(prompts_dir)

    prompt_rel = cfg["storyworld"]["prompt_file"]
    prompt_path = resolve_path(run_root, prompt_rel)
    ensure_dir(os.path.dirname(prompt_path))

    world_rel = cfg["storyworld"]["output_world"]
    encounters_rel = cfg["storyworld"]["output_encounters"]
    world_path = resolve_path(storyworld_dir, world_rel)
    encounters_path = resolve_path(storyworld_dir, encounters_rel)

    prompt = f"""
You are using the storyworld-building skill. Generate a storyworld with these constraints:
- max_nodes: {cfg['storyworld']['max_nodes']}
- max_choices_per_node: {cfg['storyworld']['max_choices_per_node']}
- required outputs: world.json and encounters.jsonl
- ensure deterministic structure using seed {cfg['seed']}

Output files:
- {world_path}
- {encounters_path}
""".strip()

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt + "\n")

    codex_cfg = cfg.get("codex_exec", {})
    output_last = ""
    if codex_cfg.get("enabled", False):
        mode = codex_cfg.get("mode", "").strip()
        sandbox = codex_cfg.get("sandbox", "").strip()
        timeout_seconds = codex_cfg.get("timeout_seconds")
        output_last = resolve_path(run_root, codex_cfg.get("output_last_message", ""))
        if output_last:
            ensure_dir(os.path.dirname(output_last))

        cmd = ["codex", "exec"]
        if mode:
            cmd.append(f"--{mode}")
        if sandbox:
            cmd.extend(["--sandbox", sandbox])
        if output_last:
            cmd.extend(["--output-last-message", output_last])
        cmd.append(prompt)

        working_dir = codex_cfg.get("working_dir", "").strip()
        cwd = resolve_path(config_dir, working_dir) if working_dir else None
        returncode, abort_reason = run_with_timeout(cmd, cwd=cwd, timeout_seconds=timeout_seconds)
        codex_result = {
            "enabled": True,
            "command": cmd,
            "returncode": returncode,
            "abort_reason": abort_reason,
            "output_last_message": output_last,
        }
        if abort_reason == "codex_timeout":
            codex_result["status"] = "aborted"
            manifest_path = os.path.join(run_root, "run_manifest.json")
            manifest = {}
            if os.path.exists(manifest_path):
                manifest = read_json(manifest_path)
            manifest["codex_exec"] = codex_result
            write_json(manifest_path, manifest)
            print("Codex exec timed out; aborting safely.")
            return

    manifest_path = os.path.join(run_root, "run_manifest.json")
    manifest = {
        "run_id": cfg["run_id"],
        "seed": cfg["seed"],
        "created_at": utc_now(),
        "paths": {
            "run_root": run_root,
            "storyworld_dir": storyworld_dir,
            "prompt": prompt_path,
            "world": world_path,
            "encounters": encounters_path,
        },
    }
    if codex_cfg.get("enabled", False):
        manifest["codex_exec"] = {
            "enabled": True,
            "output_last_message": output_last,
            "status": "completed",
        }
    write_json(manifest_path, manifest)

    print(f"Wrote prompt: {prompt_path}")
    print("Run the storyworld builder skill to generate the artifacts, then continue with gym_make_dataset.py")


if __name__ == "__main__":
    main()
