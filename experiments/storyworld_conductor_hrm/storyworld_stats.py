import json
from typing import Dict, List, Tuple


def compute_storyworld_stats(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    encounters = data.get("encounters", []) if "encounters" in data else []
    num_enc = len(encounters)
    if num_enc == 0:
        return {
            "encounters": 0,
            "avg_options": 0.0,
            "avg_reactions": 0.0,
            "avg_effects": 0.0,
        }

    opt_counts = []
    rxn_counts = []
    eff_counts = []

    for enc in encounters:
        opts = enc.get("options", [])
        opt_counts.append(len(opts))
        for opt in opts:
            rxns = opt.get("reactions", [])
            rxn_counts.append(len(rxns))
            for rxn in rxns:
                effs = rxn.get("after_effects", [])
                eff_counts.append(len(effs))

    avg_options = sum(opt_counts) / len(opt_counts) if opt_counts else 0.0
    avg_reactions = sum(rxn_counts) / len(rxn_counts) if rxn_counts else 0.0
    avg_effects = sum(eff_counts) / len(eff_counts) if eff_counts else 0.0

    return {
        "encounters": num_enc,
        "avg_options": avg_options,
        "avg_reactions": avg_reactions,
        "avg_effects": avg_effects,
    }
