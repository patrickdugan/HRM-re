import random


def choose_action(obs, choice_effects=None, rng=None):
    rng = rng or random.Random(0)
    # Optional exploration
    if rng.random() < 0.1:
        return rng.choice(obs["choices"])["id"]

    if choice_effects is None:
        # Fallback to deterministic choice if no effects are supplied
        return obs["choices"][0]["id"]

    def score(choice_id):
        effects = choice_effects.get(choice_id, {})
        return effects.get("rationality", 0) + effects.get("favor", 0) - effects.get("suspicion", 0)

    best = max((c["id"] for c in obs["choices"]), key=score)
    return best


if __name__ == "__main__":
    demo_obs = {
        "choices": [{"id": "x"}, {"id": "y"}]
    }
    demo_effects = {"x": {"rationality": 1}, "y": {"favor": 2}}
    print(choose_action(demo_obs, demo_effects))
