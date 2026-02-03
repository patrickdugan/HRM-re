import random

WEIGHTS = {
    "Compliance_Resistance": 1.0,
    "Theological_Conviction": -1.0,
    "Public_Standing": 0.2,
    "Aql_Naql": 0.0,
}


def choose_action(obs, choice_effects=None, rng=None):
    rng = rng or random.Random(0)
    choice_effects = choice_effects or {}

    best_id = None
    best_score = None

    for choice in obs.get("choices", []):
        cid = choice.get("id")
        deltas = choice_effects.get(cid, {})
        score = 0.0
        for key, weight in WEIGHTS.items():
            score += weight * deltas.get(key, 0.0)

        # small bias to avoid dead-ends if no effects are visible
        score += rng.random() * 0.001

        if best_score is None or score > best_score:
            best_score = score
            best_id = cid

    if best_id is None:
        return rng.choice(obs["choices"])["id"]
    return best_id
