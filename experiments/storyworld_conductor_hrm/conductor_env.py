import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Dict, List


STORYWORLD_ENV_PATH = r"C:\projects\GPTStoryworld\storyworld-env\__init__.py"


@dataclass
class StoryworldMetrics:
    path: str
    valid: bool
    errors: List[str]
    structural_score: float
    effect_diversity: float
    gating_score: float
    ending_diversity: float
    dead_end_rate: float
    max_ending_share: float
    min_ending_share: float
    late_block_rate: float
    secret_reachability: float


def _load_storyworld_env():
    spec = importlib.util.spec_from_file_location("storyworld_env", STORYWORLD_ENV_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load storyworld env from {STORYWORLD_ENV_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze_storyworld(path: str, mc_runs: int = 200, seed: int = 42) -> StoryworldMetrics:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    env = _load_storyworld_env()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid, errors = env.SweepweaveValidator.validate_structure(data)
    requirements = {
        "min_characters": 2,
        "min_encounters": 5,
        "min_spools": 2,
        "min_options_per_encounter": 2,
    }

    structural_score = env.SweepweaveValidator.compute_structural_score(data, requirements)
    effect_diversity = env.SweepweaveValidator.compute_effect_diversity(data)
    gating_score = env.SweepweaveValidator.compute_gating_score(data)
    ending_diversity = env.SweepweaveValidator.compute_ending_diversity(data)

    dead_end_rate = env.SweepweaveValidator.compute_dead_end_rate(data, runs=mc_runs, seed=seed)
    max_share, min_share = env.SweepweaveValidator.compute_ending_balance(data, runs=mc_runs, seed=seed)
    late_block_rate = env.SweepweaveValidator.compute_late_block_rate(data, runs=mc_runs, seed=seed)
    secret_reach = env.SweepweaveValidator.compute_secret_reachability(data, runs=mc_runs, seed=seed)

    return StoryworldMetrics(
        path=path,
        valid=valid,
        errors=errors,
        structural_score=structural_score,
        effect_diversity=effect_diversity,
        gating_score=gating_score,
        ending_diversity=ending_diversity,
        dead_end_rate=dead_end_rate,
        max_ending_share=max_share,
        min_ending_share=min_share,
        late_block_rate=late_block_rate,
        secret_reachability=secret_reach,
    )


def benchmark_metrics(path: str, mc_runs: int = 200, seed: int = 42) -> Dict[str, float]:
    env = _load_storyworld_env()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return env.evaluate_benchmark(data, runs=mc_runs, seed=seed)


def benchmark_pass(path: str, mc_runs: int = 200, seed: int = 42) -> bool:
    env = _load_storyworld_env()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = env.evaluate_benchmark(data, runs=mc_runs, seed=seed)
    return env.benchmark_pass(metrics)


def summarize_state(m: StoryworldMetrics) -> str:
    error_text = "; ".join(m.errors[:6]) if m.errors else "none"
    return (
        f"PATH: {m.path}\n"
        f"VALID: {m.valid}\n"
        f"ERRORS: {error_text}\n"
        f"STRUCTURAL_SCORE: {m.structural_score:.3f}\n"
        f"EFFECT_DIVERSITY: {m.effect_diversity:.3f}\n"
        f"GATING_SCORE: {m.gating_score:.3f}\n"
        f"ENDING_DIVERSITY: {m.ending_diversity:.3f}\n"
        f"DEAD_END_RATE: {m.dead_end_rate:.3f}\n"
        f"MAX_ENDING_SHARE: {m.max_ending_share:.3f}\n"
        f"MIN_ENDING_SHARE: {m.min_ending_share:.3f}\n"
        f"LATE_BLOCK_RATE: {m.late_block_rate:.3f}\n"
        f"SECRET_REACHABILITY: {m.secret_reachability:.3f}\n"
    )


def suggest_action(m: StoryworldMetrics) -> str:
    errors = " ".join(m.errors).lower()
    if not m.valid:
        if "spools" in errors and "starts_active" in errors:
            return "MATERIALIZE_SPOOLS"
        if "consequence_id" in errors:
            return "REPAIR_CONNECTED_SPOOLS"
        return "SWEEPWEAVE_VALIDATE"

    if m.dead_end_rate > 0.05:
        return "TUNE_ENDING_GATES"
    if m.max_ending_share > 0.30 or m.min_ending_share < 0.01:
        return "ENDING_REACHABILITY"
    if m.secret_reachability <= 0.0 and m.gating_score < 0.2:
        return "ADD_SECRET_LOGIC"
    if m.effect_diversity < 0.2:
        return "ADD_EFFECTS"

    return "FINALIZE"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("storyworld", help="Path to storyworld JSON")
    parser.add_argument("--mc_runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = analyze_storyworld(args.storyworld, mc_runs=args.mc_runs, seed=args.seed)
    print(summarize_state(metrics))
    print(f"SUGGESTED_ACTION: {suggest_action(metrics)}")
