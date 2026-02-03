import argparse
import json
import os


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_path(root, path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def extract_nodes(world):
    if isinstance(world, dict):
        if isinstance(world.get("nodes"), list):
            return world["nodes"]
        if isinstance(world.get("pages"), list):
            return world["pages"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_json(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    run_root = resolve_path(config_dir, cfg["paths"]["root"])

    storyworld_dir = os.path.join(run_root, cfg["paths"]["storyworld_dir"])
    world_path = resolve_path(storyworld_dir, cfg["storyworld"]["output_world"])
    encounters_path = resolve_path(storyworld_dir, cfg["storyworld"]["output_encounters"])

    result = {"status": "failed", "reasons": []}

    if not os.path.exists(world_path):
        result["reasons"].append("missing_world")
    if not os.path.exists(encounters_path):
        result["reasons"].append("missing_encounters")

    if result["reasons"]:
        print(json.dumps(result))
        return 2

    try:
        world = read_json(world_path)
    except Exception:
        result["reasons"].append("invalid_world_json")
        print(json.dumps(result))
        return 2

    try:
        encounters = list(read_jsonl(encounters_path))
    except Exception:
        result["reasons"].append("invalid_encounters_jsonl")
        print(json.dumps(result))
        return 2

    nodes = extract_nodes(world)
    if nodes is None:
        result["reasons"].append("unknown_world_schema")
        print(json.dumps(result))
        return 2

    max_nodes = int(cfg["storyworld"].get("max_nodes", 0))
    max_choices = int(cfg["storyworld"].get("max_choices_per_node", 0))

    if max_nodes and len(nodes) > max_nodes:
        result["reasons"].append("max_nodes_exceeded")

    if max_choices:
        for node in nodes:
            choices = node.get("choices") if isinstance(node, dict) else None
            if choices is not None and len(choices) > max_choices:
                result["reasons"].append("max_choices_exceeded")
                break

    if result["reasons"]:
        print(json.dumps(result))
        return 2

    result = {
        "status": "verified",
        "node_count": len(nodes),
        "encounter_count": len(encounters),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
