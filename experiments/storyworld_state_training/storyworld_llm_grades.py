import importlib.util
import os
from typing import Any, Dict


STORYWORLD_ENV_DIR = r"C:\projects\GPTStoryworld\storyworld-env"
STORYWORLD_ENV_INIT = os.path.join(STORYWORLD_ENV_DIR, "__init__.py")


def _load_storyworld_env():
    spec = importlib.util.spec_from_file_location("storyworld_env", STORYWORLD_ENV_INIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load storyworld-env from {STORYWORLD_ENV_INIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_storyworld_grades(data: Dict[str, Any], mc_runs: int = 200, seed: int = 42) -> Dict[str, float]:
    """Compute the storyworld-env grading metrics used for LLM evaluation."""
    env = _load_storyworld_env()
    SweepweaveValidator = env.SweepweaveValidator
    detect_endings = env.detect_endings

    grades: Dict[str, float] = {}
    valid, errors = SweepweaveValidator.validate_structure(data)
    grades["valid"] = 1.0 if valid else 0.0
    grades["error_count"] = float(len(errors))

    requirements = {
        "min_characters": 2,
        "min_encounters": 6,
        "min_spools": 2,
        "min_options_per_encounter": 2,
        "min_reactions_per_option": 1,
        "min_effects_per_reaction": 1,
    }
    grades["structural_score"] = SweepweaveValidator.compute_structural_score(data, requirements)
    grades["effect_diversity"] = SweepweaveValidator.compute_effect_diversity(data)
    grades["gating_score"] = SweepweaveValidator.compute_gating_score(data)
    grades["ending_diversity"] = SweepweaveValidator.compute_ending_diversity(data)
    grades["dead_end_rate"] = SweepweaveValidator.compute_dead_end_rate(data, runs=mc_runs, seed=seed)
    max_share, min_share = SweepweaveValidator.compute_ending_balance(data, runs=mc_runs, seed=seed)
    grades["ending_max_share"] = max_share
    grades["ending_min_share"] = min_share
    grades["late_block_rate"] = SweepweaveValidator.compute_late_block_rate(data, runs=mc_runs, seed=seed)
    grades["secret_reachability"] = SweepweaveValidator.compute_secret_reachability(data, runs=mc_runs, seed=seed)
    grades["unreachable_endings_score"] = SweepweaveValidator.compute_unreachable_endings_score(
        data, runs=mc_runs, seed=seed
    )
    grades["effects_per_reaction"] = SweepweaveValidator.compute_effects_per_reaction(data)
    grades["reactions_per_option"] = SweepweaveValidator.compute_reactions_per_option(data)
    grades["options_per_encounter"] = SweepweaveValidator.compute_options_per_encounter(data)
    grades["desirability_vars_per_reaction"] = SweepweaveValidator.compute_desirability_vars_per_reaction(data)
    grades["secret_gate_quality"] = SweepweaveValidator.compute_secret_gate_quality(data)
    grades["major_turn_quality"] = SweepweaveValidator.compute_major_turn_quality(data)
    grades["pvalue_desirability_alignment"] = SweepweaveValidator.compute_pvalue_desirability_alignment(data)
    grades["effect_script_quality"] = SweepweaveValidator.compute_effect_script_quality(data)
    act_stats = SweepweaveValidator.compute_act_gating_stats(data)
    grades["act2_gated_pct"] = act_stats.get("act2", (0.0, 0.0))[0]
    grades["act2_gated_vars"] = act_stats.get("act2", (0.0, 0.0))[1]
    grades["act3_gated_pct"] = act_stats.get("act3", (0.0, 0.0))[0]
    grades["act3_gated_vars"] = act_stats.get("act3", (0.0, 0.0))[1]
    grades["secret_metric_distance_quality"] = SweepweaveValidator.compute_secret_metric_distance_quality(data)
    grades["text_length_compliance"] = SweepweaveValidator.compute_text_length_compliance(data)
    grades["num_endings"] = float(len(detect_endings(data)))

    return grades
