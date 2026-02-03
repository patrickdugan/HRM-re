import argparse
import json
import random
from datetime import datetime

from env_storyworld import make_env

TARGET_KEYWORDS = ["bicameral", "bicam"]
SECRET_KEY = "_secret_"

WEIGHTS = {
    "Compliance_Resistance": 0.3,
    "Theological_Conviction": 0.4,
    "Public_Standing": 0.3,
    "Aql_Naql": 0.0,
}


def choose_action(env, obs, target_tag=None):
    choices = obs.get("choices", [])

    # Prefer bicameral options explicitly
    for c in choices:
        cid = c.get("id", "")
        if target_tag and target_tag in cid:
            return cid, "Choose target bicameral option when available."
    for c in choices:
        cid = c.get("id", "")
        if any(k in cid for k in TARGET_KEYWORDS):
            return cid, "Choose bicameral option when available."

    # Trigger secret path when possible
    for c in choices:
        if SECRET_KEY in c.get("id", ""):
            return c["id"], "Trigger secret path."

    # Heuristic: prefer options with positive deltas on key properties
    effects_map = getattr(env, "get_choice_effects", lambda: {})()
    best_id = None
    best_score = None
    for c in choices:
        cid = c.get("id")
        deltas = effects_map.get(cid, {})
        score = 0.0
        for key, weight in WEIGHTS.items():
            score += weight * deltas.get(key, 0.0)
        if best_score is None or score > best_score:
            best_score = score
            best_id = cid

    if best_id is None:
        return env.sample_choice(), "Fallback random choice."
    return best_id, "Improve core alignment properties."


def run_episode(env, target_tag=None):
    obs = env.reset(seed=env._rng.randint(0, 1_000_000))
    steps = []
    done = False
    safety_limit = max(5, env.max_turns * 5)
    while not done:
        action, plan = choose_action(env, obs, target_tag=target_tag)
        result = env.step(action)
        steps.append({
            "t": len(steps),
            "obs": obs,
            "action": action,
            "plan": plan,
            "outcome": {
                "next_node_id": result.obs["node_id"],
                "visible_vars": result.obs["visible_vars"],
                "done": result.done,
                "ending_id": result.info.get("ending_id"),
            },
        })
        obs = result.obs
        done = result.done
        if len(steps) >= safety_limit and not done:
            return steps, {"ending_id": None, "target_hit": False}
    return steps, result.info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyworld", required=True)
    parser.add_argument("--target_ending_id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_turns", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = make_env(args.storyworld, args.target_ending_id, max_turns=args.max_turns)
    env._rng = rng
    target_tag = args.target_ending_id.replace("page_end_", "")

    counts = {"success": 0, "normal": 0, "timeout": 0}

    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.episodes):
            steps, info = run_episode(env, target_tag=target_tag)
            category = "timeout"
            if info.get("ending_id") is not None:
                if info.get("target_hit"):
                    category = "success"
                else:
                    category = "normal"
            record = {
                "episode_id": f"ep_{sum(counts.values()):05d}",
                "target_ending_id": args.target_ending_id,
                "max_turns": args.max_turns,
                "steps": steps,
            }
            f.write(json.dumps(record) + "\n")
            counts[category] += 1

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "counts": counts,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
