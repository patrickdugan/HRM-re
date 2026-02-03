import argparse
import json
import os
import random
from typing import Dict, List

from experiments.storyworld_secret_ending.env_storyworld import make_env


def choose_target_ending_id(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "endings" in data:
        endings = data.get("endings", [])
        for ending in endings:
            if ending.get("type") == "secret":
                return ending["id"]
        if endings:
            return endings[0]["id"]
    spools = data.get("spools", [])
    for sp in spools:
        name = (sp.get("spool_name") or sp.get("id") or "").lower()
        if "ending" in name or "end" in name:
            encs = sp.get("encounters", [])
            if encs:
                return encs[0]
    for enc in data.get("encounters", []):
        eid = enc.get("id", "")
        if eid.startswith("page_secret_"):
            return eid
    for enc in data.get("encounters", []):
        eid = enc.get("id", "")
        if eid.startswith("page_end_"):
            return eid
    if data.get("encounters"):
        return data["encounters"][0]["id"]
    if data.get("nodes"):
        return data["nodes"][0]["id"]
    raise ValueError("No encounters/nodes to select target ending")


def collect_episode(env, rng: random.Random) -> Dict:
    obs = env.reset(seed=rng.randint(0, 1_000_000))
    steps = []
    done = False
    while not done:
        choices = [c["id"] for c in obs["choices"]]
        if not choices:
            break
        action = rng.choice(choices)
        result = env.step(action)
        steps.append({
            "obs": obs,
            "action": action,
            "outcome": {
                "ending_id": result.info.get("ending_id"),
                "target_hit": result.info.get("target_hit"),
            },
        })
        obs = result.obs
        done = result.done
    return {"steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyworld_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes_per_storyworld", type=int, default=50)
    parser.add_argument("--max_turns", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--subset", nargs="*", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = []
    subset = set(args.subset) if args.subset else None
    for entry in os.listdir(args.storyworld_dir):
        if not entry.endswith(".json"):
            continue
        if subset and entry not in subset:
            continue
        sw_path = os.path.join(args.storyworld_dir, entry)
        target = choose_target_ending_id(sw_path)
        env = make_env(sw_path, target, max_turns=args.max_turns)
        for _ in range(args.episodes_per_storyworld):
            records.append(collect_episode(env, rng))

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} episodes to {args.out}")


if __name__ == "__main__":
    main()
