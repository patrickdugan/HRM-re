import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from conductor_env import analyze_storyworld, benchmark_metrics, benchmark_pass, summarize_state, suggest_action


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKILL_SCRIPTS_DIR = r"C:\projects\GPTStoryworld\codex-skills\storyworld-building\scripts"
VALIDATOR = os.path.join(SKILL_SCRIPTS_DIR, "sweepweave_validator.py")
MATERIALIZE = os.path.join(SKILL_SCRIPTS_DIR, "materialize_spools.py")
REPAIR = os.path.join(SKILL_SCRIPTS_DIR, "repair_connected_spools.py")
ENDING_REACH = os.path.join(SKILL_SCRIPTS_DIR, "ending_reachability_balance.py")
SPOOL_SEQ = os.path.join(SKILL_SCRIPTS_DIR, "spool_sequencing.py")
ADD_SECRET = os.path.join(SKILL_SCRIPTS_DIR, "add_secret_logic.py")
ADD_EFFECTS = os.path.join(SKILL_SCRIPTS_DIR, "add_effects.py")


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<ACT>"]


def _run(cmd: List[str]) -> Tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _validate(path: str) -> Tuple[bool, str]:
    code, out = _run(["python", VALIDATOR, "validate", path])
    return code == 0, out


def _load_storyworld(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_patch(path: str) -> bool:
    data = _load_storyworld(path)
    changed = False
    for char in data.get("characters", []):
        if "pronoun" not in char:
            char["pronoun"] = "they"
            changed = True
    for prop in data.get("authored_properties", []):
        if "depth" not in prop:
            prop["depth"] = 0
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    return changed


def _pick_first_ids(path: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    data = _load_storyworld(path)
    for enc in data.get("encounters", []):
        enc_id = enc.get("id")
        opts = enc.get("options", [])
        if not opts:
            continue
        opt_id = opts[0].get("id")
        rxns = opts[0].get("reactions", [])
        if not rxns:
            return enc_id, opt_id, None
        return enc_id, opt_id, 0
    return None, None, None


def _apply_action(path: str, action: str) -> str:
    if action == "SWEEPWEAVE_VALIDATE":
        return "validate"
    if action == "SCHEMA_PATCH":
        changed = _schema_patch(path)
        return "schema_patch" if changed else "schema_patch_noop"
    if action == "MATERIALIZE_SPOOLS":
        tmp = path + ".tmp"
        _run(["python", MATERIALIZE, path, tmp])
        shutil.move(tmp, path)
        return "materialize_spools"
    if action == "REPAIR_CONNECTED_SPOOLS":
        _run(["python", REPAIR, path])
        return "repair_connected_spools"
    if action == "TUNE_ENDING_GATES":
        _run(["python", ENDING_REACH, path, "--apply", "--bias", "0.01", "--runs", "200", "--seed", "42"])
        return "ending_reachability_apply"
    if action == "ENDING_REACHABILITY":
        _run(["python", ENDING_REACH, path, "--apply", "--bias", "0.005", "--runs", "200", "--seed", "42"])
        return "ending_reachability_apply_small"
    if action == "ADD_SECRET_LOGIC":
        enc_id, opt_id, _ = _pick_first_ids(path)
        if enc_id and opt_id:
            _run(["python", ADD_SECRET, path, enc_id, opt_id])
            return "add_secret_logic"
        return "add_secret_logic_skipped"
    if action == "ADD_EFFECTS":
        enc_id, opt_id, rxn_idx = _pick_first_ids(path)
        if enc_id and opt_id and rxn_idx is not None:
            _run(["python", ADD_EFFECTS, path, enc_id, opt_id, str(rxn_idx)])
            return "add_effects"
        return "add_effects_skipped"
    if action == "SPOOL_SEQUENCING":
        _run(["python", SPOOL_SEQ, path])
        return "spool_sequencing"
    return "noop"


def tokenize(text: str) -> List[str]:
    return text.replace("\n", " ").split()


def serialize_input(state: str, tools: list) -> str:
    return f"STATE: {state}\nTOOLS: {', '.join(tools)}"


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


def hrm_choose_action(model, vocab, state_text: str, tools: list, device: str) -> str:
    text = serialize_input(state_text, tools)
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
            return tok.split("ACTION::", 1)[1]
    return "SWEEPWEAVE_VALIDATE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyworld", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--mc_runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--hrm_model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tools = [
        "SWEEPWEAVE_VALIDATE",
        "SCHEMA_PATCH",
        "MATERIALIZE_SPOOLS",
        "REPAIR_CONNECTED_SPOOLS",
        "TUNE_ENDING_GATES",
        "ENDING_REACHABILITY",
        "ADD_SECRET_LOGIC",
        "ADD_EFFECTS",
        "SPOOL_SEQUENCING",
        "FINALIZE",
    ]

    run_dir = args.workdir or os.path.join(os.path.dirname(args.out), f"run_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    work_path = os.path.join(run_dir, os.path.basename(args.storyworld))
    shutil.copy2(args.storyworld, work_path)

    model = None
    vocab = None
    if args.hrm_model:
        model, vocab = load_hrm(args.hrm_model, args.device)

    trace = []

    for step in range(args.steps):
        metrics = analyze_storyworld(work_path, mc_runs=args.mc_runs, seed=args.seed)
        state_text = summarize_state(metrics)
        bench = benchmark_metrics(work_path, mc_runs=args.mc_runs, seed=args.seed)
        bench_ok = benchmark_pass(work_path, mc_runs=args.mc_runs, seed=args.seed)
        valid, validate_out = _validate(work_path)

        if model is not None:
            action = hrm_choose_action(model, vocab, state_text, tools, args.device)
        else:
            action = suggest_action(metrics)

        if action == "FINALIZE" or bench_ok:
            trace.append({
                "step": step,
                "action": action,
                "state": state_text,
                "benchmark": bench,
                "benchmark_pass": bench_ok,
                "valid": valid,
                "validate_out": validate_out,
                "applied": "finalize" if action == "FINALIZE" else "benchmark_pass",
            })
            break

        applied = _apply_action(work_path, action)
        post_valid, post_out = _validate(work_path)

        trace.append({
            "step": step,
            "action": action,
            "state": state_text,
            "benchmark": bench,
            "benchmark_pass": bench_ok,
            "valid": valid,
            "validate_out": validate_out,
            "applied": applied,
            "post_valid": post_valid,
            "post_validate_out": post_out,
        })

    final_metrics = analyze_storyworld(work_path, mc_runs=args.mc_runs, seed=args.seed)
    final_bench = benchmark_metrics(work_path, mc_runs=args.mc_runs, seed=args.seed)
    final_pass = benchmark_pass(work_path, mc_runs=args.mc_runs, seed=args.seed)
    report = {
        "input_storyworld": args.storyworld,
        "work_storyworld": work_path,
        "steps": len(trace),
        "trace": trace,
        "final_state": summarize_state(final_metrics),
        "final_benchmark": final_bench,
        "benchmark_pass": final_pass,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
