import argparse
import json
import os

from conductor_env import analyze_storyworld, summarize_state, suggest_action


def list_storyworlds(root: str):
    for name in os.listdir(root):
        if name.endswith(".json"):
            yield os.path.join(root, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyworld_dir", default=r"C:\projects\GPTStoryworld\storyworlds")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--mc_runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tools = [
        "SWEEPWEAVE_VALIDATE",
        "MATERIALIZE_SPOOLS",
        "REPAIR_CONNECTED_SPOOLS",
        "TUNE_ENDING_GATES",
        "ENDING_REACHABILITY",
        "ADD_SECRET_LOGIC",
        "ADD_EFFECTS",
        "FINALIZE",
    ]

    records = []
    for idx, path in enumerate(list_storyworlds(args.storyworld_dir)):
        metrics = analyze_storyworld(path, mc_runs=args.mc_runs, seed=args.seed)
        state = summarize_state(metrics)
        action = suggest_action(metrics)
        records.append({
            "id": os.path.basename(path),
            "state": state,
            "tools": tools,
            "action": action,
        })
        if args.max_examples and len(records) >= args.max_examples:
            break

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} examples to {args.out}")


if __name__ == "__main__":
    main()
