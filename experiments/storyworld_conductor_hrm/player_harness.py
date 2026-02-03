import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

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


def serialize_obs(obs: Dict) -> str:
    choices = "; ".join([f"{c['id']}: {c['text']}" for c in obs["choices"]])
    vars_text = ", ".join([f"{k}={v}" for k, v in obs["visible_vars"].items()])
    return (
        f"NODE_ID: {obs['node_id']}\n"
        f"NODE_TEXT: {obs['node_text']}\n"
        f"CHOICES: {choices}\n"
        f"VARS: {vars_text}\n"
        f"TURNS_LEFT: {obs['turns_left']}"
    )


def tokenize(text: str) -> List[str]:
    return text.replace("\n", " ").split()


def load_hrm(model_path: str, device: str):
    payload = torch.load(model_path, map_location=device)
    vocab = payload["vocab"]
    config = payload["config"]
    from HRM.hrm import HRM

    model = HRM(
        networks=config["networks"],
        dim=config["dim"],
        num_tokens=len(vocab),
        reasoning_steps=config["reasoning_steps"],
        relative_period=config["relative_period"],
        causal=False,
        ignore_index=-1,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, vocab


def hrm_choose_action(model, vocab, obs: Dict, device: str, valid_choices: List[str]) -> str:
    text = serialize_obs(obs)
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids.append(vocab["<ACT>"])
    seq = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        pred, _ = model(seq)
    logits = pred[0, -1]
    action_id = int(torch.argmax(logits).item())
    for tok, tok_id in vocab.items():
        if tok_id == action_id and tok.startswith("ACTION::"):
            choice_id = tok.split("ACTION::", 1)[1]
            if choice_id in valid_choices:
                return choice_id
    return random.choice(valid_choices)


def greedy_choose_action(obs: Dict, choice_effects: Dict[str, Dict[str, float]], rng: random.Random) -> str:
    if not choice_effects:
        return obs["choices"][0]["id"]
    def score(cid):
        effects = choice_effects.get(cid, {})
        return sum(float(v) for v in effects.values())
    return max((c["id"] for c in obs["choices"]), key=score)


def run_agent(env, episodes: int, seed: int, agent: str, hrm_model=None, hrm_vocab=None, device: str = "cpu") -> Dict[str, float]:
    rng = random.Random(seed)
    successes = 0
    timeouts = 0
    turns_all = []
    turns_success = []

    for i in range(episodes):
        obs = env.reset(seed=seed + i)
        done = False
        turns = 0
        safety_limit = max(5, env.max_turns * 5)
        while not done:
            valid_choices = [c["id"] for c in obs["choices"]]
            if not valid_choices:
                break
            if agent == "hrm" and hrm_model is not None:
                action = hrm_choose_action(hrm_model, hrm_vocab, obs, device, valid_choices)
            elif agent == "greedy":
                choice_effects = getattr(env, "get_choice_effects", lambda: {})()
                action = greedy_choose_action(obs, choice_effects, rng)
            else:
                action = rng.choice(valid_choices)
            result = env.step(action)
            obs = result.obs
            done = result.done
            turns += 1
            if turns >= safety_limit and not done:
                result.info["ending_id"] = None
                done = True
        turns_all.append(turns)
        if result.info.get("target_hit"):
            successes += 1
            turns_success.append(turns)
        if result.info.get("ending_id") is None:
            timeouts += 1

    success_rate = successes / episodes if episodes else 0.0
    median_turns = None
    if turns_success:
        turns_success.sort()
        median_turns = turns_success[len(turns_success) // 2]
    mean_turns = sum(turns_all) / len(turns_all) if turns_all else None

    return {
        "success_rate": success_rate,
        "median_turns_to_success": median_turns,
        "mean_turns": mean_turns,
        "timeout_rate": timeouts / episodes if episodes else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyworld", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_turns", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hrm_model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    target_ending_id = choose_target_ending_id(args.storyworld)
    env = make_env(args.storyworld, target_ending_id, max_turns=args.max_turns)

    hrm_model = None
    hrm_vocab = None
    if args.hrm_model:
        hrm_model, hrm_vocab = load_hrm(args.hrm_model, args.device)

    report = {
        "storyworld": args.storyworld,
        "target_ending_id": target_ending_id,
        "max_turns": args.max_turns,
        "agents": {}
    }

    report["agents"]["random"] = run_agent(env, args.episodes, args.seed + 1000, "random")
    report["agents"]["greedy"] = run_agent(env, args.episodes, args.seed + 2000, "greedy")
    if hrm_model is not None:
        report["agents"]["hrm"] = run_agent(env, args.episodes, args.seed, "hrm", hrm_model, hrm_vocab, args.device)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
