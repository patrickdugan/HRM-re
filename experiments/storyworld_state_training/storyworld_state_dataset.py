import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class StoryworldStateSample:
    state_t: torch.Tensor
    state_t_next: torch.Tensor
    operators: torch.Tensor
    state_delta: Optional[torch.Tensor] = None
    encounter_acceptability: Optional[float] = None
    encounter_desirability: Optional[float] = None
    option_visibility: Optional[float] = None
    option_performability: Optional[float] = None
    reaction_desirability: Optional[float] = None
    ending_scores: Optional[torch.Tensor] = None
    ending_acceptability: Optional[torch.Tensor] = None
    ending_structural_distance: Optional[torch.Tensor] = None
    ending_metric_distance: Optional[torch.Tensor] = None
    meta: Dict = field(default_factory=dict)


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
    if operator_type == "Arithmetic Comparator":
        subtype = node.get("operator_subtype") or node.get("operator_type_subtype")
        if len(values) < 2:
            return None
        a, b = values[0], values[1]
        if subtype in ("GTE", "Greater Than or Equal To"):
            return 1.0 if a >= b else 0.0
        if subtype in ("GT", "Greater Than"):
            return 1.0 if a > b else 0.0
        if subtype in ("LTE", "Less Than or Equal To"):
            return 1.0 if a <= b else 0.0
        if subtype in ("LT", "Less Than"):
            return 1.0 if a < b else 0.0
        if subtype in ("EQ", "Equal To"):
            return 1.0 if a == b else 0.0
        if subtype in ("NEQ", "Not Equal To"):
            return 1.0 if a != b else 0.0
        return None
    if operator_type == "If Then":
        if len(values) >= 2:
            cond = values[0]
            return float(values[1] if cond else (values[2] if len(values) > 2 else 0.0))
    if operator_type == "Blend":
        if len(values) >= 3:
            alpha, a, b = values[0], values[1], values[2]
            return float((1 - alpha) * a + alpha * b)
        if len(values) == 2:
            return float(0.5 * values[0] + 0.5 * values[1])
    if operator_type == "Desideratum":
        if len(values) >= 3:
            alpha, target, base = values[0], values[1], values[2]
            return float((1 - alpha) * base + alpha * target)
        if len(values) == 2:
            return float(0.5 * values[0] + 0.5 * values[1])
    return None


def _eval_node(node: Dict, state: Dict[str, float]) -> Optional[float]:
    if not isinstance(node, dict):
        return None
    if "operator_type" in node:
        return _eval_operator(node, state)
    return _eval_pointer(node, state)


def _extract_comparator_constraints(node: Dict) -> List[Tuple[str, str, float]]:
    if not isinstance(node, dict):
        return []
    constraints: List[Tuple[str, str, float]] = []
    if node.get("operator_type") == "Arithmetic Comparator":
        subtype = node.get("operator_subtype") or node.get("operator_type_subtype") or ""
        operands = node.get("operands") or []
        if len(operands) >= 2:
            left = operands[0]
            right = operands[1]
            left_ptr = _extract_pointer_var(left) if isinstance(left, dict) else None
            right_ptr = _extract_pointer_var(right) if isinstance(right, dict) else None
            left_const = _eval_pointer(left, {}) if isinstance(left, dict) else None
            right_const = _eval_pointer(right, {}) if isinstance(right, dict) else None

            if left_ptr and right_const is not None:
                var_name = _world_var_name(left_ptr[0], left_ptr[1])
                constraints.append((var_name, subtype, float(right_const)))
            elif right_ptr and left_const is not None:
                var_name = _world_var_name(right_ptr[0], right_ptr[1])
                flipped = {
                    "GTE": "LTE",
                    "Greater Than or Equal To": "Less Than or Equal To",
                    "GT": "LT",
                    "Greater Than": "Less Than",
                    "LTE": "GTE",
                    "Less Than or Equal To": "Greater Than or Equal To",
                    "LT": "GT",
                    "Less Than": "Greater Than",
                }.get(subtype, subtype)
                constraints.append((var_name, flipped, float(left_const)))

    for key in ("operands", "conditions", "branches"):
        children = node.get(key) if isinstance(node, dict) else None
        if isinstance(children, list):
            for child in children:
                constraints.extend(_extract_comparator_constraints(child))
    return constraints


def _constraint_margin(value: Optional[float], op: str, target: float) -> Optional[float]:
    if value is None:
        return None
    if op in ("GTE", "Greater Than or Equal To"):
        return max(0.0, target - value)
    if op in ("GT", "Greater Than"):
        return max(0.0, target - value + 1e-6)
    if op in ("LTE", "Less Than or Equal To"):
        return max(0.0, value - target)
    if op in ("LT", "Less Than"):
        return max(0.0, value - target + 1e-6)
    if op in ("EQ", "Equal To"):
        return abs(value - target)
    if op in ("NEQ", "Not Equal To"):
        return 0.0 if value != target else 1.0
    return None


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
        include_scripts: bool = True,
        include_endings: bool = True,
    ):
        self.inputs = inputs
        self.max_vars = max_vars
        self.op_basis = op_basis or ["Set", "Nudge", "Blend", "Reverse", "Proximity"]
        self.include_delta = include_delta
        self.include_scripts = include_scripts
        self.include_endings = include_endings

        self.var_to_idx = self._build_var_index()
        self.ending_ids = self._build_endings() if self.include_endings else []
        self.ending_to_idx = {eid: i for i, eid in enumerate(self.ending_ids)}
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

    def _script_value(self, node, state: Dict[str, float]) -> Optional[float]:
        if node is None:
            return None
        if isinstance(node, bool):
            return 1.0 if node else 0.0
        if _is_number(node):
            return float(node)
        if isinstance(node, dict):
            return _eval_node(node, state)
        return None

    def _build_endings(self) -> List[str]:
        ending_ids: List[str] = []
        for _, kind, row in _iter_inputs(self.inputs):
            if kind != "storyworld":
                continue
            ending_spool_ids = set()
            for spool in row.get("spools", []) or []:
                spool_type = spool.get("spool_type", "")
                spool_id = spool.get("id", "")
                if spool_type == "Endings" or "ending" in spool_id.lower():
                    ending_spool_ids.add(spool_id)
            spool_map = {s.get("id"): s for s in row.get("spools", []) or []}
            for spool_id in ending_spool_ids:
                spool = spool_map.get(spool_id, {})
                for enc_id in spool.get("encounters", []) or []:
                    if enc_id not in ending_ids:
                        ending_ids.append(enc_id)
        return ending_ids

    def _build_graph(self, world: Dict) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for enc in world.get("encounters", []) or []:
            enc_id = enc.get("id")
            if not enc_id:
                continue
            next_ids: List[str] = []
            for opt in enc.get("options", []) or []:
                for react in opt.get("reactions", []) or []:
                    nxt = react.get("consequence_id")
                    if isinstance(nxt, str) and nxt:
                        next_ids.append(nxt)
            graph[enc_id] = next_ids
        return graph

    def _shortest_distances(self, graph: Dict[str, List[str]], targets: List[str]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        if not targets:
            return dist
        reverse: Dict[str, List[str]] = {}
        for src, outs in graph.items():
            for dst in outs:
                reverse.setdefault(dst, []).append(src)
        queue: List[Tuple[str, int]] = []
        for t in targets:
            dist[t] = 0
            queue.append((t, 0))
        idx = 0
        while idx < len(queue):
            node, d = queue[idx]
            idx += 1
            for prev in reverse.get(node, []):
                if prev not in dist:
                    dist[prev] = d + 1
                    queue.append((prev, d + 1))
        return dist

    def _ending_constraints(self, end_enc: Dict) -> List[Tuple[str, str, float]]:
        constraints: List[Tuple[str, str, float]] = []
        for key in ("acceptability_script", "desirability_script"):
            node = end_enc.get(key)
            if isinstance(node, dict):
                constraints.extend(_extract_comparator_constraints(node))
        return constraints

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
                    delta_vec = next_vec - state_vec if self.include_delta else None
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
                            state_delta=delta_vec,
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
                graph = self._build_graph(row)
                distances = self._shortest_distances(graph, self.ending_ids)
                endings_by_id = {e.get("id"): e for e in row.get("encounters", []) or []}
                ending_constraints = {
                    eid: self._ending_constraints(endings_by_id.get(eid, {})) for eid in self.ending_ids
                }
                for enc in row.get("encounters", []) or []:
                    encounter_accept = None
                    encounter_desire = None
                    if self.include_scripts:
                        encounter_accept = self._script_value(enc.get("acceptability_script"), base_state)
                        encounter_desire = self._script_value(enc.get("desirability_script"), base_state)
                    for opt in enc.get("options", []) or []:
                        option_visible = None
                        option_performable = None
                        if self.include_scripts:
                            option_visible = self._script_value(opt.get("visibility_script"), base_state)
                            option_performable = self._script_value(opt.get("performability_script"), base_state)
                        for react in opt.get("reactions", []) or []:
                            effects = react.get("after_effects") or []
                            if not effects:
                                continue
                            next_state, labels = self._apply_effects(base_state, effects)
                            state_vec = self._vectorize(base_state)
                            next_vec = self._vectorize(next_state)
                            ops_vec = self._ops_vector_from_labels(labels)
                            delta_vec = next_vec - state_vec if self.include_delta else None
                            reaction_desire = None
                            if self.include_scripts:
                                reaction_desire = self._script_value(react.get("desirability_script"), base_state)
                            ending_scores = None
                            ending_accept = None
                            ending_distance = None
                            ending_metric = None
                            if self.include_endings and self.ending_ids:
                                ending_scores = torch.zeros(len(self.ending_ids), dtype=torch.float32)
                                ending_accept = torch.zeros(len(self.ending_ids), dtype=torch.float32)
                                ending_distance = torch.full(
                                    (len(self.ending_ids),), float("inf"), dtype=torch.float32
                                )
                                ending_metric = torch.zeros(len(self.ending_ids), dtype=torch.float32)
                                for end_enc in row.get("encounters", []) or []:
                                    end_id = end_enc.get("id")
                                    if end_id not in self.ending_to_idx:
                                        continue
                                    idx = self.ending_to_idx[end_id]
                                    acc = self._script_value(end_enc.get("acceptability_script"), next_state)
                                    des = self._script_value(end_enc.get("desirability_script"), next_state)
                                    if acc is not None:
                                        ending_accept[idx] = float(acc)
                                    if des is not None:
                                        ending_scores[idx] = float(des)
                                for eid, idx in self.ending_to_idx.items():
                                    enc_id = enc.get("id")
                                    if enc_id in distances:
                                        ending_distance[idx] = float(distances.get(enc_id, float("inf")))
                                    constraints = ending_constraints.get(eid) or []
                                    margins: List[float] = []
                                    for var_name, op, target in constraints:
                                        margin = _constraint_margin(next_state.get(var_name), op, target)
                                        if margin is not None:
                                            margins.append(margin)
                                    if margins:
                                        ending_metric[idx] = float((sum(m * m for m in margins)) ** 0.5)
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
                                    state_delta=delta_vec,
                                    encounter_acceptability=encounter_accept,
                                    encounter_desirability=encounter_desire,
                                    option_visibility=option_visible,
                                    option_performability=option_performable,
                                    reaction_desirability=reaction_desire,
                                    ending_scores=ending_scores,
                                    ending_acceptability=ending_accept,
                                    ending_structural_distance=ending_distance,
                                    ending_metric_distance=ending_metric,
                                    meta=meta,
                                )
                            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> StoryworldStateSample:
        return self.samples[idx]
