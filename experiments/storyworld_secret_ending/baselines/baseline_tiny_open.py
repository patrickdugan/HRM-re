import random


def choose_action(obs, choice_effects=None, rng=None):
    rng = rng or random.Random(0)
    # Simple local stub that mimics a tiny open model by keyword matching.
    keywords = ["study", "patron", "pact", "argument", "defense", "discreet"]
    best = None
    best_score = -1
    for choice in obs["choices"]:
        text = choice["text"].lower()
        score = sum(1 for k in keywords if k in text)
        if score > best_score:
            best_score = score
            best = choice["id"]
    if best is not None:
        return best
    return rng.choice(obs["choices"])["id"]


if __name__ == "__main__":
    demo_obs = {
        "choices": [
            {"id": "a", "text": "Study at the academy."},
            {"id": "b", "text": "Withdraw."}
        ]
    }
    print(choose_action(demo_obs))
