import json
import os
import random
import sys
import urllib.request
import urllib.error


def _build_prompt(obs):
    choices = "\n".join([f"- {c['id']}: {c['text']}" for c in obs["choices"]])
    vars_map = obs.get("visible_vars", {})
    vars_text = ", ".join([f"{k}={v}" for k, v in vars_map.items()])
    return (
        "You are playing a storyworld. Choose exactly one option id.\n"
        f"Node: {obs['node_id']}\n"
        f"Text: {obs['node_text']}\n"
        f"Vars: {vars_text}\n"
        f"Turns left: {obs['turns_left']}\n"
        "Options:\n"
        f"{choices}\n"
        "Return only the option id."
    )


def _responses_api_call(prompt, api_key, model, max_output_tokens=32, temperature=0.2, timeout=30):
    url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/responses")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    org = os.environ.get("OPENAI_ORG")
    project = os.environ.get("OPENAI_PROJECT")
    if org:
        headers["OpenAI-Organization"] = org
    if project:
        headers["OpenAI-Project"] = project
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _extract_text(response):
    if "output_text" in response and response["output_text"]:
        return response["output_text"].strip()
    output = response.get("output", [])
    for item in output:
        if item.get("type") == "message":
            content = item.get("content", [])
            for block in content:
                if block.get("type") in ("output_text", "text") and block.get("text"):
                    return block["text"].strip()
    return ""


def choose_action(obs, api_key=None, mock=True, rng=None, model=None):
    rng = rng or random.Random(0)
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT5MINI_API_KEY")
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

    if api_key and not mock:
        prompt = _build_prompt(obs)
        try:
            response = _responses_api_call(prompt, api_key, model=model)
            text = _extract_text(response)
            for choice in obs["choices"]:
                if text.strip() == choice["id"]:
                    return choice["id"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if os.environ.get("OPENAI_DEBUG"):
                print(f"[gpt5mini] API call failed: {exc}", file=sys.stderr)
            if os.environ.get("OPENAI_STRICT"):
                raise

    # Deterministic mock policy
    vars_map = obs.get("visible_vars", {})
    rationality = vars_map.get("rationality", 50)
    favor = vars_map.get("favor", 50)
    suspicion = vars_map.get("suspicion", 50)

    for choice in obs["choices"]:
        text = choice["text"].lower()
        if "pact" in text:
            return choice["id"]

    if favor < 70:
        for choice in obs["choices"]:
            if "patron" in choice["text"].lower():
                return choice["id"]

    if rationality < 80:
        for choice in obs["choices"]:
            if "study" in choice["text"].lower() or "argument" in choice["text"].lower():
                return choice["id"]

    if suspicion > 60:
        for choice in obs["choices"]:
            if "withdraw" in choice["text"].lower() or "discreet" in choice["text"].lower():
                return choice["id"]

    return rng.choice(obs["choices"])["id"]


if __name__ == "__main__":
    demo_obs = {
        "choices": [
            {"id": "c1", "text": "Accept a risky pact."},
            {"id": "c2", "text": "Study at the academy."}
        ],
        "visible_vars": {"rationality": 60, "favor": 40, "suspicion": 20},
    }
    print(choose_action(demo_obs))
