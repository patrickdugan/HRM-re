import argparse
import inspect
import json
import os
import random
import statistics
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from env_storyworld import make_env
from baselines import baseline_random, baseline_greedy, baseline_gpt5mini, baseline_tiny_open, baseline_codex


def serialize_obs(obs):
    choices = "; ".join([f"{c['id']}: {c['text']}" for c in obs["choices"]])
    vars_text = ", ".join([f"{k}={v}" for k, v in obs["visible_vars"].items()])
    return (
        f"NODE_ID: {obs['node_id']}\n"
        f"NODE_TEXT: {obs['node_text']}\n"
        f"CHOICES: {choices}\n"
        f"VARS: {vars_text}\n"
        f"TURNS_LEFT: {obs['turns_left']}"
    )


def tokenize(text):
    return text.replace("\n", " ").split()


def load_hrm(model_path, device):
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


def hrm_choose_action(model, vocab, obs, device, valid_choices):
    text = serialize_obs(obs)
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids.append(vocab["<ACT>"])
    seq = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        pred, _ = model(seq)
    logits = pred[0, -1]
    action_id = int(torch.argmax(logits).item())
    action_token = None
    for tok, tok_id in vocab.items():
        if tok_id == action_id and tok.startswith("ACTION::"):
            action_token = tok
            break
    if action_token is None:
        return random.choice(valid_choices)
    choice_id = action_token.split("ACTION::", 1)[1]
    if choice_id not in valid_choices:
        return random.choice(valid_choices)
    return choice_id


def _call_agent(agent_fn, obs, choice_effects, rng):
    params = list(inspect.signature(agent_fn).parameters)
    if len(params) == 1:
        return agent_fn(obs)
    if len(params) == 2:
        return agent_fn(obs, choice_effects)
    return agent_fn(obs, choice_effects, rng)


def run_agent(env, agent_fn, episodes, seed, hrm_model=None, hrm_vocab=None, device="cpu"):
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
            if hrm_model is not None:
                action = hrm_choose_action(hrm_model, hrm_vocab, obs, device, valid_choices)
            else:
                choice_effects = getattr(env, "get_choice_effects", lambda: {})()
                action = _call_agent(agent_fn, obs, choice_effects, rng) if agent_fn else rng.choice(valid_choices)
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

    success_rate = successes / episodes
    timeout_rate = timeouts / episodes
    median_turns = statistics.median(turns_success) if turns_success else None
    mean_turns = statistics.mean(turns_all) if turns_all else None

    return {
        "success_rate": success_rate,
        "median_turns_to_success": median_turns,
        "mean_turns": mean_turns,
        "timeout_rate": timeout_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyworld", required=True)
    parser.add_argument("--target_ending_id", required=True)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hrm_model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpt_mock", action="store_true")
    parser.add_argument("--gpt_model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--gpt_episodes", type=int, default=None)
    parser.add_argument("--codex_episodes", type=int, default=50)
    parser.add_argument("--skip_gpt", action="store_true")
    args = parser.parse_args()

    env = make_env(args.storyworld, args.target_ending_id, max_turns=args.max_turns)

    hrm_model, hrm_vocab = load_hrm(args.hrm_model, args.device)

    results = {
        "storyworld_id": env.storyworld.get("id") or env.storyworld.get("title"),
        "target_ending_id": args.target_ending_id,
        "max_turns": args.max_turns,
        "agents": {},
    }

    results["agents"]["hrm"] = run_agent(env, None, args.episodes, args.seed, hrm_model=hrm_model, hrm_vocab=hrm_vocab, device=args.device)
    results["agents"]["random"] = run_agent(env, baseline_random.choose_action, args.episodes, args.seed + 1000)
    results["agents"]["greedy"] = run_agent(env, baseline_greedy.choose_action, args.episodes, args.seed + 2000)
    results["agents"]["tiny_open_stub"] = run_agent(env, baseline_tiny_open.choose_action, args.episodes, args.seed + 2500)

    if not args.skip_gpt:
        def gpt_agent(obs, choice_effects, rng):
            return baseline_gpt5mini.choose_action(obs, mock=args.gpt_mock, rng=rng, model=args.gpt_model)

        gpt_episodes = args.gpt_episodes if args.gpt_episodes is not None else args.episodes
        results["agents"]["gpt5mini"] = run_agent(env, gpt_agent, gpt_episodes, args.seed + 3000)

    def codex_agent(obs, choice_effects, rng):
        return baseline_codex.choose_action(obs, choice_effects=choice_effects, rng=rng)

    results["agents"]["codex_self"] = run_agent(env, codex_agent, args.codex_episodes, args.seed + 4000)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
