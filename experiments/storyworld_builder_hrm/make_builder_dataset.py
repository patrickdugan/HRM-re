import argparse
import json
import os
import random


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def iter_storyworld_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for name in os.listdir(p):
                if name.endswith(".json"):
                    yield os.path.join(p, name)
        elif os.path.isfile(p) and p.endswith(".json"):
            yield p


def build_records(world, source_name, tools, max_options, max_reactions, max_encounters):
    records = []
    encounters = world.get("encounters", []) if isinstance(world, dict) else []
    spools = world.get("spools", []) if isinstance(world, dict) else []
    world_id = world.get("IFID") or world.get("world_id") or source_name

    if not isinstance(encounters, list):
        return records

    spool_count = len(spools) if isinstance(spools, list) else 0

    encs = encounters[:max_encounters] if max_encounters else encounters
    for enc in encs:
        if not isinstance(enc, dict):
            continue
        enc_id = enc.get("id", "unknown_enc")
        options = enc.get("options", []) if isinstance(enc.get("options"), list) else []
        options = options[:max_options] if max_options else options

        base_state = (
            f"WORLD_ID: {world_id}\n"
            f"SPOOL_COUNT: {spool_count}\n"
            f"ENCOUNTER_ID: {enc_id}\n"
            f"OPTION_COUNT: {len(options)}"
        )

        records.append({"state": base_state, "tools": tools, "action": "ADD_ENCOUNTER"})

        for opt in options:
            opt_id = opt.get("id", "opt") if isinstance(opt, dict) else "opt"
            records.append({"state": base_state + f"\nOPTION_ID: {opt_id}", "tools": tools, "action": "ADD_OPTION"})
            reactions = opt.get("reactions", []) if isinstance(opt, dict) else []
            if isinstance(reactions, list):
                reactions = reactions[:max_reactions] if max_reactions else reactions
                for react in reactions:
                    react_id = react.get("id", "react") if isinstance(react, dict) else "react"
                    records.append({"state": base_state + f"\nOPTION_ID: {opt_id}\nREACTION_ID: {react_id}", "tools": tools, "action": "ADD_REACTION"})

        if isinstance(enc.get("connected_spools"), list) and enc.get("connected_spools"):
            records.append({"state": base_state, "tools": tools, "action": "LINK_SPOOL"})

        records.append({"state": base_state, "tools": tools, "action": "FINALIZE_ENCOUNTER"})

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_encounters", type=int, default=0)
    parser.add_argument("--max_options", type=int, default=0)
    parser.add_argument("--max_reactions", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    tools = [
        "codex",
        "storyworld-building",
        "verifier",
        "json_editor",
        "diversity_sampler",
        "tropes_resource",
    ]
    records = []

    files = list(iter_storyworld_files(args.input))
    random.shuffle(files)

    for path in files:
        try:
            world = read_json(path)
        except Exception:
            continue
        records.extend(
            build_records(
                world,
                os.path.basename(path),
                tools,
                args.max_options,
                args.max_reactions,
                args.max_encounters,
            )
        )
        world_id = world.get("IFID") or world.get("world_id") or os.path.basename(path)
        base_state = f"WORLD_ID: {world_id}\\nPHASE: PLAN"
        for action in [
            "ADD_HIDDEN_VAR",
            "ADD_SECRET_ENDING",
            "ADD_TENSION_CURVE",
            "ADD_VALUE_CONFLICT",
            "ADD_RIVAL_FACTION",
            "ADD_NEGOTIATION_PACT",
            "ADD_IRREVERSIBLE_CHOICE",
        ]:
            records.append({"state": base_state, "tools": tools, "action": action})

    write_jsonl(args.out, records)
    print(f"Wrote {args.out} ({len(records)} records)")


if __name__ == "__main__":
    main()
