import random


def choose_action(obs, choice_effects=None, rng=None):
    rng = rng or random.Random(0)
    return rng.choice(obs["choices"])["id"]


if __name__ == "__main__":
    demo_obs = {
        "choices": [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    }
    print(choose_action(demo_obs))
