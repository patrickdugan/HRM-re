import argparse
import json
import os
import subprocess
import shutil
import time
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


def extract_blocks(text):
    blocks = []
    fence = "```"
    idx = 0
    while True:
        start = text.find(fence, idx)
        if start == -1:
            break
        lang_end = text.find("\n", start + len(fence))
        if lang_end == -1:
            break
        lang = text[start + len(fence) : lang_end].strip()
        end = text.find(fence, lang_end + 1)
        if end == -1:
            break
        content = text[lang_end + 1 : end]
        blocks.append((lang, content.strip()))
        idx = end + len(fence)
    return blocks


def write_artifacts_from_codex(output_last, world_path, encounters_path):
    if not output_last or not os.path.exists(output_last):
        return False
    with open(output_last, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    blocks = extract_blocks(text)
    world_block = None
    encounters_block = None
    for lang, content in blocks:
        if lang == "json" and world_block is None:
            world_block = content
        if lang == "jsonl" and encounters_block is None:
            encounters_block = content
    if world_block:
        ensure_dir(os.path.dirname(world_path))
        with open(world_path, "w", encoding="utf-8") as f:
            f.write(world_block + "\n")
    if encounters_block:
        ensure_dir(os.path.dirname(encounters_path))
        with open(encounters_path, "w", encoding="utf-8") as f:
            f.write(encounters_block + "\n")
    return bool(world_block or encounters_block)


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

    preamble_path = cfg["storyworld"].get("prompt_preamble_file") or ""
    preamble_text = ""
    if preamble_path:
        preamble_path = resolve_path(run_root, preamble_path)
        if os.path.exists(preamble_path):
            with open(preamble_path, "r", encoding="utf-8", errors="ignore") as f:
                preamble_text = f.read().strip()

    prompt = f"""
You are using the storyworld-building skill. Generate a storyworld with these constraints:
- max_nodes: {cfg['storyworld']['max_nodes']}
- max_choices_per_node: {cfg['storyworld']['max_choices_per_node']}
- required outputs: world.json and encounters.jsonl
- ensure deterministic structure using seed {cfg['seed']}

Additional diversity guidance:
{preamble_text}

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

        codex_cmd = codex_cfg.get("command", "codex")
        resolved = shutil.which(codex_cmd)
        if not resolved:
            manifest_path = os.path.join(run_root, "run_manifest.json")
            manifest = {}
            if os.path.exists(manifest_path):
                manifest = read_json(manifest_path)
            manifest["codex_exec"] = {
                "enabled": True,
                "status": "aborted",
                "abort_reason": "codex_not_found",
                "command": codex_cmd,
            }
            write_json(manifest_path, manifest)
            print(f"Codex executable not found: {codex_cmd}")
            return

        cmd = [resolved, "exec"]
        if mode:
            cmd.append(f"--{mode}")
        if sandbox:
            cmd.extend(["--sandbox", sandbox])
        if output_last:
            cmd.extend(["--output-last-message", output_last])
        cmd.append(prompt)

        working_dir = codex_cfg.get("working_dir", "").strip()
        cwd = resolve_path(config_dir, working_dir) if working_dir else None
        max_retries = int(codex_cfg.get("max_retries", 0))
        backoff_seconds = int(codex_cfg.get("backoff_seconds", 10))
        attempt = 0
        last_returncode = None
        last_abort = None
        while attempt <= max_retries:
            returncode, abort_reason = run_with_timeout(cmd, cwd=cwd, timeout_seconds=timeout_seconds)
            last_returncode = returncode
            last_abort = abort_reason
            if abort_reason is None and returncode == 0:
                break
            if abort_reason == "codex_timeout" and attempt < max_retries:
                time.sleep(backoff_seconds)
            else:
                break
            attempt += 1

        codex_result = {
            "enabled": True,
            "command": cmd,
            "returncode": last_returncode,
            "abort_reason": last_abort,
            "output_last_message": output_last,
            "attempts": attempt + 1,
        }
        if last_abort == "codex_timeout":
            codex_result["status"] = "aborted"
            manifest_path = os.path.join(run_root, "run_manifest.json")
            manifest = {}
            if os.path.exists(manifest_path):
                manifest = read_json(manifest_path)
            manifest["codex_exec"] = codex_result
            write_json(manifest_path, manifest)
            print("Codex exec timed out; aborting safely.")
            return
        elif last_returncode != 0:
            codex_result["status"] = "failed"
            manifest_path = os.path.join(run_root, "run_manifest.json")
            manifest = {}
            if os.path.exists(manifest_path):
                manifest = read_json(manifest_path)
            manifest["codex_exec"] = codex_result
            write_json(manifest_path, manifest)
            raise RuntimeError(f"codex exec failed with code {last_returncode}")

        if not (os.path.exists(world_path) and os.path.exists(encounters_path)):
            wrote = write_artifacts_from_codex(output_last, world_path, encounters_path)
            if wrote:
                print("Wrote storyworld artifacts from Codex output.")

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
