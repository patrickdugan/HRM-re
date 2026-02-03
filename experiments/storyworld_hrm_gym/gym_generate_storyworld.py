import argparse
import json
import os
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
    write_json(manifest_path, manifest)

    print(f"Wrote prompt: {prompt_path}")
    print("Run the storyworld builder skill to generate the artifacts, then continue with gym_make_dataset.py")


if __name__ == "__main__":
    main()
