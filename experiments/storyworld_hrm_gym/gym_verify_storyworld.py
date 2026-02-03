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
            return "nodes", world["nodes"]
        if isinstance(world.get("pages"), list):
            return "pages", world["pages"]
        if isinstance(world.get("encounters"), list):
            return "encounters", world["encounters"]
    return None, None


def validate_node(node):
    if not isinstance(node, dict):
        return False, "node_not_object"
    if "id" not in node or not node["id"]:
        return False, "node_missing_id"
    choices = node.get("choices")
    if choices is None:
        return False, "node_missing_choices"
    if not isinstance(choices, list) or len(choices) == 0:
        return False, "node_choices_invalid"
    for choice in choices:
        if not isinstance(choice, dict):
            return False, "choice_not_object"
        if "id" not in choice or "text" not in choice:
            return False, "choice_missing_fields"
    return True, ""


def validate_encounter(encounter):
    if not isinstance(encounter, dict):
        return False, "encounter_not_object"
    if "id" not in encounter or not encounter["id"]:
        return False, "encounter_missing_id"
    options = encounter.get("options")
    if options is None:
        return False, "encounter_missing_options"
    if not isinstance(options, list) or len(options) == 0:
        return False, "encounter_options_invalid"
    for opt in options:
        if not isinstance(opt, dict):
            return False, "option_not_object"
        if "id" not in opt or not opt["id"]:
            return False, "option_missing_id"
        reactions = opt.get("reactions")
        if reactions is None:
            return False, "option_missing_reactions"
        if not isinstance(reactions, list):
            return False, "option_reactions_invalid"
        for react in reactions:
            if not isinstance(react, dict):
                return False, "reaction_not_object"
            if "id" not in react or not react["id"]:
                return False, "reaction_missing_id"
    return True, ""


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

    schema, nodes = extract_nodes(world)
    if nodes is None:
        result["reasons"].append("unknown_world_schema")
        print(json.dumps(result))
        return 2

    max_nodes = int(cfg["storyworld"].get("max_nodes", 0))
    max_choices = int(cfg["storyworld"].get("max_choices_per_node", 0))

    if max_nodes and len(nodes) > max_nodes:
        result["reasons"].append("max_nodes_exceeded")

    if schema in ("nodes", "pages"):
        if max_choices:
            for node in nodes:
                choices = node.get("choices") if isinstance(node, dict) else None
                if choices is not None and len(choices) > max_choices:
                    result["reasons"].append("max_choices_exceeded")
                    break

        for node in nodes:
            ok, reason = validate_node(node)
            if not ok:
                result["reasons"].append(reason)
                break

    if schema == "encounters":
        if max_choices:
            for enc in nodes:
                options = enc.get("options") if isinstance(enc, dict) else None
                if options is not None and len(options) > max_choices:
                    result["reasons"].append("max_choices_exceeded")
                    break

        for enc in nodes:
            ok, reason = validate_encounter(enc)
            if not ok:
                result["reasons"].append(reason)
                break

    if result["reasons"]:
        print(json.dumps(result))
        return 2

    result = {
        "status": "verified",
        "node_count": len(nodes),
        "encounter_count": len(encounters),
        "schema": schema,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
