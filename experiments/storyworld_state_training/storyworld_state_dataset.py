import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class StoryworldStateSample:
    state_t: torch.Tensor
    state_t_next: torch.Tensor
    operators: torch.Tensor
    meta: Dict


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_inputs(inputs: List[str]) -> Iterable[Tuple[str, str, Dict]]:
    for inp in inputs:
        if os.path.isdir(inp):
            for name in os.listdir(inp):
                if name.endswith(".jsonl") or name.endswith(".json"):
                    path = os.path.join(inp, name)
                    if name.endswith(".jsonl"):
                        for row in _iter_jsonl(path):
                            yield path, "jsonl", row
                    else:
                        with open(path, "r", encoding="utf-8") as f:
                            yield path, "storyworld", json.load(f)
        elif os.path.isfile(inp):
            if inp.endswith(".jsonl"):
                for row in _iter_jsonl(inp):
                    yield inp, "jsonl", row
            elif inp.endswith(".json"):
                with open(inp, "r", encoding="utf-8") as f:
                    yield inp, "storyworld", json.load(f)


def _extract_visible_vars(step: Dict, key: str) -> Dict:
    node = step.get(key, {})
    if not isinstance(node, dict):
        return {}
    vars_map = node.get("visible_vars", {})
    if isinstance(vars_map, dict):
        return vars_map
    return {}


def _infer_ops(state_t: Dict, state_t_next: Dict) -> Dict[str, float]:
    ops = {}
    for name, next_val in state_t_next.items():
        prev_val = state_t.get(name)
        if prev_val is None:
            continue
        if isinstance(prev_val, (int, float)) and isinstance(next_val, (int, float)):
            if next_val != prev_val:
                ops[name] = next_val - prev_val
    return ops


def _is_number(value) -> bool:
    return isinstance(value, (int, float))


def _world_var_name(character_id: str, keyring: List[str]) -> str:
    if not keyring:
        return character_id
    return f"{character_id}." + ".".join(keyring)


def _extract_pointer_var(pointer: Dict) -> Optional[Tuple[str, List[str]]]:
    if not isinstance(pointer, dict):
        return None
    character = pointer.get("character")
    keyring = pointer.get("keyring") or []
    if not character or not isinstance(keyring, list):
        return None
    return character, keyring


def _eval_pointer(pointer: Dict, state: Dict[str, float]) -> Optional[float]:
    if not isinstance(pointer, dict):
        return None
    pointer_type = pointer.get("pointer_type", "")
    if "Constant" in pointer_type:
        value = pointer.get("value")
        return float(value) if _is_number(value) else None
    if "Pointer" in pointer_type or pointer.get("script_element_type") == "Pointer":
        var = _extract_pointer_var(pointer)
        if not var:
            return None
        name = _world_var_name(var[0], var[1])
        value = state.get(name)
        return float(value) if _is_number(value) else None
    return None


def _eval_operator(node: Dict, state: Dict[str, float]) -> Optional[float]:
    operator_type = node.get("operator_type")
    operands = node.get("operands") or []
    if not operator_type or not isinstance(operands, list):
        return None
    values = []
    for operand in operands:
        val = _eval_node(operand, state)
        if val is None:
            return None
        values.append(val)
    if not values:
        return None
    if operator_type in ("Nudge", "Addition", "Sum"):
        return float(sum(values))
    if operator_type == "Multiplication":
        out = 1.0
        for v in values:
            out *= v
        return float(out)
    if operator_type == "Blend":
        if len(values) >= 3:
            alpha, a, b = values[0], values[1], values[2]
            return float((1 - alpha) * a + alpha * b)
        if len(values) == 2:
            return float(0.5 * values[0] + 0.5 * values[1])
    return None


def _eval_node(node: Dict, state: Dict[str, float]) -> Optional[float]:
    if not isinstance(node, dict):
        return None
    if "operator_type" in node:
        return _eval_operator(node, state)
    return _eval_pointer(node, state)


class StoryworldStateDataset(Dataset):
    """Builds (state_t, state_t+1, operators) tensors from storyworld JSONL episodes.

    Expected input format (from experiments/storyworld_secret_ending/make_dataset.py):
    - Each JSONL row has "steps".
    - Each step has "obs.visible_vars" and "outcome.visible_vars".
    """

    def __init__(
        self,
        inputs: List[str],
        max_vars: int = 1024,
        op_basis: Optional[List[str]] = None,
        include_delta: bool = True,
    ):
        self.inputs = inputs
        self.max_vars = max_vars
        self.op_basis = op_basis or ["Set", "Nudge", "Blend", "Reverse", "Proximity"]
        self.include_delta = include_delta

        self.var_to_idx = self._build_var_index()
        self.samples = self._build_samples()

    def _build_var_index(self) -> Dict[str, int]:
        vocab = {}
        for _, kind, row in _iter_inputs(self.inputs):
            if kind == "jsonl":
                for step in row.get("steps", []):
                    for key in ("obs", "outcome"):
                        vars_map = _extract_visible_vars(step, key)
                        for name in vars_map.keys():
                            if name not in vocab:
                                vocab[name] = len(vocab)
                            if len(vocab) >= self.max_vars:
                                return vocab
            elif kind == "storyworld":
                for char in row.get("characters", []) or []:
                    char_id = char.get("id")
                    props = char.get("bnumber_properties", {})
                    if not char_id or not isinstance(props, dict):
                        continue
                    for prop_name in props.keys():
                        name = f"{char_id}.{prop_name}"
                        if name not in vocab:
                            vocab[name] = len(vocab)
                        if len(vocab) >= self.max_vars:
                            return vocab
        return vocab

    def _vectorize(self, vars_map: Dict) -> torch.Tensor:
        vec = torch.zeros(self.max_vars, dtype=torch.float32)
        for name, val in vars_map.items():
            idx = self.var_to_idx.get(name)
            if idx is None or idx >= self.max_vars:
                continue
            if isinstance(val, bool):
                vec[idx] = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                vec[idx] = float(val)
        return vec

    def _ops_vector(self, state_t: Dict, state_t_next: Dict) -> torch.Tensor:
        ops_vec = torch.zeros(len(self.op_basis), dtype=torch.float32)
        deltas = _infer_ops(state_t, state_t_next)
        if deltas:
            # Minimal baseline: any numeric change is treated as a "Set" op.
            # You can replace this with schema-driven operator tags later.
            set_idx = self.op_basis.index("Set") if "Set" in self.op_basis else None
            if set_idx is not None:
                ops_vec[set_idx] = 1.0
        return ops_vec

    def _ops_vector_from_labels(self, labels: List[str]) -> torch.Tensor:
        ops_vec = torch.zeros(len(self.op_basis), dtype=torch.float32)
        for label in labels:
            if label in self.op_basis:
                ops_vec[self.op_basis.index(label)] = 1.0
        return ops_vec

    def _operator_label(self, effect: Dict) -> str:
        to_node = effect.get("to")
        if isinstance(to_node, dict):
            op_type = to_node.get("operator_type")
            if op_type in ("Nudge", "Addition", "Sum"):
                return "Nudge"
            if op_type == "Blend":
                return "Blend"
            if op_type == "Multiplication":
                return "Blend"
            if op_type == "Reverse":
                return "Reverse"
        return "Set"

    def _apply_effects(self, base_state: Dict[str, float], effects: List[Dict]) -> Tuple[Dict[str, float], List[str]]:
        state = dict(base_state)
        labels: List[str] = []
        for eff in effects:
            if not isinstance(eff, dict):
                continue
            target = eff.get("Set") or {}
            var_info = _extract_pointer_var(target)
            if not var_info:
                continue
            var_name = _world_var_name(var_info[0], var_info[1])
            value = _eval_node(eff.get("to", {}), state)
            if value is None:
                continue
            state[var_name] = float(value)
            labels.append(self._operator_label(eff))
        return state, labels

    def _build_samples(self) -> List[StoryworldStateSample]:
        samples: List[StoryworldStateSample] = []
        for path, kind, row in _iter_inputs(self.inputs):
            if kind == "jsonl":
                for step in row.get("steps", []):
                    state_t = _extract_visible_vars(step, "obs")
                    state_t_next = _extract_visible_vars(step, "outcome")
                    if not state_t or not state_t_next:
                        continue
                    state_vec = self._vectorize(state_t)
                    next_vec = self._vectorize(state_t_next)
                    ops_vec = self._ops_vector(state_t, state_t_next)
                    meta = {
                        "source": path,
                        "episode_id": row.get("episode_id") or row.get("trajectory_id"),
                        "t": step.get("t"),
                    }
                    samples.append(
                        StoryworldStateSample(
                            state_t=state_vec,
                            state_t_next=next_vec,
                            operators=ops_vec,
                            meta=meta,
                        )
                    )
            elif kind == "storyworld":
                base_state: Dict[str, float] = {}
                for char in row.get("characters", []) or []:
                    char_id = char.get("id")
                    props = char.get("bnumber_properties", {})
                    if not char_id or not isinstance(props, dict):
                        continue
                    for prop_name, value in props.items():
                        name = f"{char_id}.{prop_name}"
                        if isinstance(value, bool):
                            base_state[name] = 1.0 if value else 0.0
                        elif _is_number(value):
                            base_state[name] = float(value)
                for enc in row.get("encounters", []) or []:
                    for opt in enc.get("options", []) or []:
                        for react in opt.get("reactions", []) or []:
                            effects = react.get("after_effects") or []
                            if not effects:
                                continue
                            next_state, labels = self._apply_effects(base_state, effects)
                            state_vec = self._vectorize(base_state)
                            next_vec = self._vectorize(next_state)
                            ops_vec = self._ops_vector_from_labels(labels)
                            meta = {
                                "source": path,
                                "encounter_id": enc.get("id"),
                                "option_id": opt.get("id"),
                                "reaction_id": react.get("id"),
                            }
                            samples.append(
                                StoryworldStateSample(
                                    state_t=state_vec,
                                    state_t_next=next_vec,
                                    operators=ops_vec,
                                    meta=meta,
                                )
                            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> StoryworldStateSample:
        return self.samples[idx]
