import argparse
import json
import random
from datetime import datetime

from env_storyworld import make_env


TARGET_SCORE = {
    "Aql_Naql": 1.0,
    "Compliance_Resistance": 0.8,
    "Public_Standing": 0.5,
    "Theological_Conviction": 0.8,
    "rationality": 1.0,
    "favor": 1.0,
    "suspicion": -1.2,
}


def _choice_score(choice_id, effects_map):
    score = 0.0
    effects = effects_map.get(choice_id, {})
    for key, weight in TARGET_SCORE.items():
        score += weight * effects.get(key, 0)
    return score


def _advance_choices(env):
    node = env.get_node()
    if not node or "options" not in node:
        return None
    choices = []
    for opt in node.get("options", []):
        reactions = opt.get("reactions") or []
        if not reactions:
            continue
        if reactions[0].get("consequence_id"):
            choices.append(opt["id"])
    return choices


def secret_policy(env, obs):
    # Prefer secret-tagged options if present
    for c in obs["choices"]:
        if "_secret_" in c["id"]:
            return c["id"], "Trigger secret path, then optimize alignment variables."

    effects_map = getattr(env, "get_choice_effects", lambda: {})()
    advancing = _advance_choices(env) or [c["id"] for c in obs["choices"]]

    best_id = max(advancing, key=lambda cid: _choice_score(cid, effects_map))
    best = next(c for c in obs["choices"] if c["id"] == best_id)
    return best["id"], "Optimize alignment variables toward target ending."


def normal_policy(env, obs):
    # Push toward non-secret outcomes by avoiding secret options
    non_secret = [c for c in obs["choices"] if "_secret_" not in c["id"]]
    if non_secret:
        advancing = _advance_choices(env) or [c["id"] for c in non_secret]
        return advancing[0], "Avoid secret path; steer to normal ending."
    return obs["choices"][0]["id"], "Force a normal ending quickly."


def random_policy(env, obs):
    return env.sample_choice(), "Explore randomly to induce timeouts."


def run_episode(env, policy_fn):
    obs = env.reset(seed=env._rng.randint(0, 1_000_000))
    steps = []
    done = False
    safety_limit = max(5, env.max_turns * 5)
    while not done:
        action, plan = policy_fn(env, obs)
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
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict_mix", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = make_env(args.storyworld, args.target_ending_id, max_turns=args.max_turns)
    env._rng = rng

    target_success = int(args.episodes * 0.3)
    target_normal = int(args.episodes * 0.3)
    target_timeout = args.episodes - target_success - target_normal

    counts = {"success": 0, "normal": 0, "timeout": 0}

    with open(args.out, "w", encoding="utf-8") as f:
        attempts = 0
        max_attempts = args.episodes * (40 if args.strict_mix else 4)
        while sum(counts.values()) < args.episodes and attempts < max_attempts:
            attempts += 1
            if counts["success"] < target_success:
                policy = secret_policy
                expected = "success"
            elif counts["normal"] < target_normal:
                policy = normal_policy
                expected = "normal"
            else:
                policy = random_policy
                expected = "timeout"

            steps, info = run_episode(env, policy)
            category = "timeout"
            if info.get("ending_id") is not None:
                if info.get("target_hit"):
                    category = "success"
                else:
                    category = "normal"

            if args.strict_mix and category != expected:
                continue

            episode_id = f"ep_{sum(counts.values()):05d}"
            record = {
                "episode_id": episode_id,
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
