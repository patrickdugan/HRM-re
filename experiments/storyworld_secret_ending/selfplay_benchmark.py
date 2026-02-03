import argparse
import json
import random
import statistics

from env_storyworld import make_env


def choose_action(env, obs, target, storyworld):
    node = env.get_node()
    choices = node.get("choices", [])
    visible = obs["visible_vars"]
    risky_pact = getattr(env, "hidden_vars", {}).get("risky_pact", False)

    node_id = obs["node_id"]
    # Deterministic path to the secret ending (found via search).
    if node_id == "courtyard_entry":
        if not risky_pact and visible["rationality"] >= 70:
            return "visit_archives", "Head to the archives to trigger the risky pact."
        return "study_academy", "Build rationality to meet the secret ending gate."
    if node_id == "academy":
        return "rigorous_study", "Push rationality upward while advancing."
    if node_id == "debate_hall" and not risky_pact:
        return "withdraw", "Cycle back to reach the archives for the risky pact."
    if node_id == "archives" and not risky_pact:
        return "accept_risky_pact", "Secure the risky pact to enable the secret ending."
    if node_id == "secret_passage":
        return "meet_patrons", "Boost favor while keeping suspicion manageable."
    if node_id == "patrons_house":
        return "secure_endorsement", "Raise favor and proceed toward judgment."

    def clamp(val, spec):
        return max(spec["min"], min(spec["max"], val))

    def score(choice):
        effects = choice.get("effects", {})
        next_vals = {}
        for key, spec in storyworld["variables"].items():
            delta = effects.get(key, 0)
            next_vals[key] = clamp(visible[key] + delta, spec)

        # Distance to target (higher is better).
        score_val = 0.0
        score_val += next_vals["rationality"] / target["rationality"]
        score_val += next_vals["favor"] / target["favor"]
        # Keep suspicion low.
        score_val += (target["suspicion_max"] - next_vals["suspicion"]) / target["suspicion_max"]

        # Prefer moving to the judgment hall when close to target.
        if choice.get("to") == "hall_of_judgment":
            if next_vals["rationality"] >= target["rationality"] and next_vals["favor"] >= target["favor"] and next_vals["suspicion"] <= target["suspicion_max"]:
                score_val += 3.0
            else:
                score_val += 0.5
        return score_val

    best = max(choices, key=score)
    plan = "Improve alignment variables while keeping suspicion low."
    if best.get("to") == "hall_of_judgment":
        plan = "Head to judgment hall to attempt the ending gate."
    return best["id"], plan


def run_episode(env, episode_id, target, storyworld):
    obs = env.reset(seed=episode_id)
    steps = []
    done = False
    t = 0
    while not done:
        action, plan = choose_action(env, obs, target, storyworld)
        result = env.step(action)
        steps.append({
            "episode": episode_id,
            "t": t,
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
        t += 1
    return steps, result.info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyworld", required=True)
    parser.add_argument("--target_ending_id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = make_env(args.storyworld, args.target_ending_id, max_turns=args.max_turns)
    env._rng = random.Random(args.seed)

    with open(args.storyworld, "r", encoding="utf-8") as f:
        storyworld = json.load(f)

    # Target thresholds are defined by the secret ending condition in mihna_min.
    target = {
        "rationality": 85,
        "favor": 75,
        "suspicion_max": 40,
    }

    successes = 0
    timeouts = 0
    turns_all = []
    turns_success = []

    with open(args.out, "w", encoding="utf-8") as f:
        for ep in range(1, args.episodes + 1):
            steps, info = run_episode(env, args.seed + ep, target, storyworld)
            for step in steps:
                f.write(json.dumps(step) + "\n")
            turns = len(steps)
            turns_all.append(turns)
            if info.get("target_hit"):
                successes += 1
                turns_success.append(turns)
            if info.get("ending_id") is None:
                timeouts += 1

        summary = {
            "episodes": args.episodes,
            "successes": successes,
            "success_rate": successes / args.episodes if args.episodes else 0.0,
            "median_turns_to_success": statistics.median(turns_success) if turns_success else None,
            "mean_turns": statistics.mean(turns_all) if turns_all else None,
            "timeouts": timeouts,
        }
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
