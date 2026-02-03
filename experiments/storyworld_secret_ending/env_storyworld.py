import json
import random
import re
from dataclasses import dataclass


COND_ALLOWED_RE = re.compile(r"^[A-Za-z0-9_ ()<>=!&|.+-]*$")


def _normalize_condition(cond):
    if cond is None:
        return ""
    text = cond.strip()
    if not text:
        return ""
    text = re.sub(r"\bAND\b", "and", text, flags=re.IGNORECASE)
    text = re.sub(r"\bOR\b", "or", text, flags=re.IGNORECASE)
    text = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfalse\b", "False", text, flags=re.IGNORECASE)
    return text


def _safe_eval_condition(cond, visible_vars, hidden_vars):
    normalized = _normalize_condition(cond)
    if not normalized:
        return True
    if not COND_ALLOWED_RE.match(normalized):
        raise ValueError(f"Unsafe condition: {cond}")
    context = {}
    context.update(visible_vars)
    context.update(hidden_vars)
    try:
        return bool(eval(normalized, {"__builtins__": {}}, context))
    except Exception as exc:
        raise ValueError(f"Failed to evaluate condition '{cond}': {exc}")


def _script_to_text(script):
    if script is None:
        return ""
    if isinstance(script, str):
        return script
    if isinstance(script, dict) and "value" in script:
        return script["value"]
    return ""


@dataclass
class StepResult:
    obs: dict
    reward: float
    done: bool
    info: dict


class StoryworldEnv:
    def __init__(self, storyworld_path, target_ending_id, max_turns=10):
        with open(storyworld_path, "r", encoding="utf-8") as f:
            self.storyworld = json.load(f)
        self.target_ending_id = target_ending_id
        self.max_turns = max_turns
        self.nodes = {node["id"]: node for node in self.storyworld["nodes"]}
        self.endings = self.storyworld.get("endings", [])
        self._rng = random.Random()
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self._rng.seed(seed)
        self.turn = 0
        self.node_id = self.storyworld["nodes"][0]["id"]
        self.visible_vars = {
            name: spec["start"] for name, spec in self.storyworld["variables"].items()
        }
        self.hidden_vars = {
            name: spec.get("start", False)
            for name, spec in self.storyworld.get("hidden_variables", {}).items()
        }
        return self._make_obs()

    def _make_obs(self):
        node = self.nodes[self.node_id]
        return {
            "node_id": node["id"],
            "node_text": node["text"],
            "choices": [{"id": c["id"], "text": c["text"]} for c in node["choices"]],
            "visible_vars": dict(self.visible_vars),
            "turns_left": self.max_turns - self.turn,
        }

    def _apply_effects(self, effects):
        if not effects:
            return
        for key, delta in effects.items():
            if key not in self.visible_vars:
                continue
            spec = self.storyworld["variables"][key]
            new_val = self.visible_vars[key] + delta
            self.visible_vars[key] = max(spec["min"], min(spec["max"], new_val))

    def _apply_hidden_effects(self, effects):
        if not effects:
            return
        for key, val in effects.items():
            if key not in self.hidden_vars:
                continue
            self.hidden_vars[key] = bool(val)

    def _check_endings(self):
        for ending in self.endings:
            if ending["node_id"] != self.node_id:
                continue
            if _safe_eval_condition(ending.get("condition", ""), self.visible_vars, self.hidden_vars):
                return ending
        return None

    def step(self, choice_id):
        node = self.nodes[self.node_id]
        choice = None
        for c in node["choices"]:
            if c["id"] == choice_id:
                choice = c
                break
        if choice is None:
            raise ValueError(f"Invalid choice_id '{choice_id}' for node '{self.node_id}'")

        self.turn += 1
        self._apply_effects(choice.get("effects", {}))
        self._apply_hidden_effects(choice.get("hidden_effects", {}))
        self.node_id = choice["to"]

        ending = self._check_endings()
        done = False
        reward = -1.0
        info = {
            "ending_id": None,
            "ending_type": None,
            "target_hit": False,
        }

        if ending is not None:
            done = True
            info["ending_id"] = ending["id"]
            info["ending_type"] = ending["type"]
            reward += 10.0
            if ending["id"] == self.target_ending_id:
                reward += 100.0
                info["target_hit"] = True
            elif ending["type"] == "normal":
                reward -= 5.0

        if self.turn >= self.max_turns and not done:
            done = True
            info["ending_id"] = None
            info["ending_type"] = None

        obs = self._make_obs()
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def sample_choice(self):
        node = self.nodes[self.node_id]
        return self._rng.choice(node["choices"])["id"]

    def available_choices(self):
        node = self.nodes[self.node_id]
        return [c["id"] for c in node["choices"]]

    def get_node(self):
        return self.nodes[self.node_id]


class SweepWeaveEnv:
    def __init__(self, storyworld_path, target_ending_id, max_turns=10):
        with open(storyworld_path, "r", encoding="utf-8") as f:
            self.storyworld = json.load(f)
        self.target_ending_id = target_ending_id
        self.max_turns = max_turns
        self._rng = random.Random()
        self.encounters = {e["id"]: e for e in self.storyworld["encounters"]}
        self.spools = self.storyworld.get("spools", [])
        self.ending_spool = self._find_ending_spool()
        self.player_id = self.storyworld["characters"][0]["id"]
        self.visible_props = [
            p["property_name"]
            for p in self.storyworld.get("authored_properties", [])
            if not p["property_name"].startswith("p")
        ]
        self.reset()

    def _find_ending_spool(self):
        for spool in self.storyworld.get("spools", []):
            if "ending" in spool.get("spool_name", "").lower() or "ending" in spool.get("id", "").lower():
                return spool
        return None

    def reset(self, seed=None):
        if seed is not None:
            self._rng.seed(seed)
        self.turn = 0
        self.pending_reaction = None
        self.hidden_vars = {"secret_path": False}
        self.characters = {}
        for char in self.storyworld["characters"]:
            props = dict(char.get("bnumber_properties", {}))
            for key, val in list(props.items()):
                if isinstance(val, dict):
                    props[key] = 0.0
            self.characters[char["id"]] = {
                "name": char.get("name", char["id"]),
                "properties": props,
            }
        first_spool = next((s for s in self.spools if s.get("starts_active")), None)
        if first_spool and first_spool.get("encounters"):
            self.current_id = first_spool["encounters"][0]
        else:
            self.current_id = self.storyworld["encounters"][0]["id"]
        return self._make_obs()

    def _make_obs(self):
        enc = self.encounters[self.current_id]
        choices = []
        for opt in enc.get("options", []):
            if self._eval_visibility(opt.get("visibility_script", True)):
                choices.append({
                    "id": opt["id"],
                    "text": _script_to_text(opt.get("text_script")),
                })
        visible_vars = {k: self.characters[self.player_id]["properties"].get(k, 0.0) for k in self.visible_props}
        return {
            "node_id": enc["id"],
            "node_text": _script_to_text(enc.get("text_script")),
            "choices": choices,
            "visible_vars": visible_vars,
            "turns_left": self.max_turns - self.turn,
        }

    def _nudge(self, current, amount):
        if amount > 0:
            return current + amount * (1 - current)
        return current + amount * (1 + current)

    def _eval_visibility(self, script):
        if script is True or script is False:
            return bool(script)
        if script is None:
            return True
        if isinstance(script, (int, float)):
            return bool(script)
        if isinstance(script, dict):
            return bool(self._eval_script(script))
        return True

    def _eval_script(self, script):
        if not isinstance(script, dict):
            return script
        if "operator_type" in script:
            op = script.get("operator_type")
            operands = script.get("operands") or []
            if op == "And":
                return all(self._eval_script(o) for o in operands)
            if op == "Or":
                return any(self._eval_script(o) for o in operands)
            if op == "Not":
                return not self._eval_script(operands[0]) if operands else False
            if op == "Arithmetic Comparator":
                subtype = script.get("operator_subtype")
                left = self._eval_script(operands[0]) if len(operands) > 0 else 0
                right = self._eval_script(operands[1]) if len(operands) > 1 else 0
                if subtype == "Greater Than or Equal To":
                    return left >= right
                if subtype == "Less Than or Equal To":
                    return left <= right
                if subtype == "Greater Than":
                    return left > right
                if subtype == "Less Than":
                    return left < right
                if subtype == "Equal To":
                    return left == right
                return False
            return False
        if script.get("pointer_type") == "Bounded Number Pointer":
            char_id = script.get("character")
            keyring = script.get("keyring") or []
            if not char_id or not keyring:
                return 0
            return self.characters.get(char_id, {}).get("properties", {}).get(keyring[0], 0)
        if script.get("pointer_type") == "Bounded Number Constant":
            return script.get("value", 0)
        if script.get("pointer_type") == "String Constant":
            return script.get("value", "")
        if script.get("pointer_type") == "Boolean Constant":
            return bool(script.get("value", False))
        return False

    def _apply_effects(self, effects):
        for effect in effects or []:
            if effect.get("effect_type") != "Bounded Number Effect":
                continue
            set_ptr = effect.get("Set")
            if not set_ptr:
                continue
            char_id = set_ptr.get("character")
            keyring = set_ptr.get("keyring") or []
            if not char_id or not keyring:
                continue
            prop_name = keyring[0]
            char = self.characters.get(char_id)
            if not char:
                continue
            if prop_name not in char["properties"]:
                continue
            to_expr = effect.get("to")
            if to_expr and to_expr.get("operator_type") == "Nudge":
                operands = to_expr.get("operands") or []
                if len(operands) > 1 and isinstance(operands[1], dict):
                    raw = operands[1].get("value", 0.0)
                    if isinstance(raw, (int, float)):
                        nudge_amount = raw
                    else:
                        nudge_amount = 0.0
                else:
                    nudge_amount = 0.0
                char["properties"][prop_name] = self._nudge(char["properties"][prop_name], nudge_amount)

    def _is_ending(self, encounter_id):
        if self.ending_spool:
            if encounter_id in self.ending_spool.get("encounters", []):
                return True
        enc = self.encounters.get(encounter_id)
        return not enc.get("options")

    def _find_choice(self, choice_id):
        enc = self.encounters[self.current_id]
        for opt in enc.get("options", []):
            if opt.get("id") == choice_id:
                return opt
        return None

    def step(self, choice_id):
        option = self._find_choice(choice_id)
        if not option:
            raise ValueError(f"Invalid choice_id '{choice_id}' for encounter '{self.current_id}'")

        if "_secret_" in choice_id:
            self.hidden_vars["secret_path"] = True

        reactions = option.get("reactions") or []
        if not reactions:
            next_id = None
            reaction_text = ""
            after_effects = []
        else:
            reaction = reactions[0]
            reaction_text = _script_to_text(reaction.get("text_script"))
            next_id = reaction.get("consequence_id")
            after_effects = reaction.get("after_effects") or []

        self._apply_effects(after_effects)
        if next_id:
            self.current_id = next_id
            self.turn += 1

        done = False
        reward = -1.0
        info = {
            "ending_id": None,
            "ending_type": None,
            "target_hit": False,
            "reaction_text": reaction_text,
        }

        if self._is_ending(self.current_id):
            done = True
            info["ending_id"] = self.current_id
            info["ending_type"] = "secret" if self.hidden_vars.get("secret_path") else "normal"
            reward += 10.0
            if self.current_id == self.target_ending_id and info["ending_type"] == "secret":
                reward += 100.0
                info["target_hit"] = True
            elif info["ending_type"] == "normal":
                reward -= 5.0

        if self.turn >= self.max_turns and not done:
            done = True

        obs = self._make_obs()
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def sample_choice(self):
        return self._rng.choice(self.available_choices())

    def available_choices(self):
        enc = self.encounters[self.current_id]
        return [
            o["id"]
            for o in enc.get("options", [])
            if self._eval_visibility(o.get("visibility_script", True))
        ]

    def get_node(self):
        return self.encounters[self.current_id]

    def get_choice_effects(self):
        enc = self.encounters[self.current_id]
        effects = {}
        for opt in enc.get("options", []):
            if not self._eval_visibility(opt.get("visibility_script", True)):
                continue
            reaction = (opt.get("reactions") or [None])[0]
            if not reaction:
                effects[opt["id"]] = {}
                continue
            deltas = {}
            for effect in reaction.get("after_effects") or []:
                if effect.get("effect_type") != "Bounded Number Effect":
                    continue
                set_ptr = effect.get("Set")
                keyring = set_ptr.get("keyring") if set_ptr else []
                if not keyring:
                    continue
                prop = keyring[0]
                if prop.startswith("p"):
                    continue
                to_expr = effect.get("to")
                if to_expr and to_expr.get("operator_type") == "Nudge":
                    operands = to_expr.get("operands") or []
                    nudge_amount = operands[1].get("value", 0.0) if len(operands) > 1 else 0.0
                    deltas[prop] = deltas.get(prop, 0.0) + nudge_amount
            effects[opt["id"]] = deltas
        return effects


def make_env(storyworld_path, target_ending_id, max_turns=10):
    with open(storyworld_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "encounters" in data and "spools" in data:
        return SweepWeaveEnv(storyworld_path, target_ending_id, max_turns=max_turns)
    return StoryworldEnv(storyworld_path, target_ending_id, max_turns=max_turns)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--storyworld", required=True)
    parser.add_argument("--target_ending_id", required=True)
    parser.add_argument("--max_turns", type=int, default=10)
    args = parser.parse_args()

    env = make_env(args.storyworld, args.target_ending_id, max_turns=args.max_turns)
    obs = env.reset(seed=0)
    print(json.dumps(obs, indent=2))
