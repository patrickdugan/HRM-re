import argparse
import json
import os
from datetime import datetime, timezone

import torch
from torch.utils.data import Dataset, DataLoader

from HRM.hrm import HRM


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def resolve_path(root, path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


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


class EvalDataset(Dataset):
    def __init__(self, examples, vocab):
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, action_token = self.examples[idx]
        token_ids = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in tokens]
        action_id = self.vocab.get(action_token, self.vocab["<UNK>"])
        return token_ids, action_id


def build_examples(eval_rows, vocab):
    examples = []
    for record in eval_rows:
        for step in record["steps"]:
            obs_text = serialize_obs(step["obs"])
            tokens = tokenize(obs_text)
            action_token = f"ACTION::{step['action']}"
            if action_token in vocab:
                examples.append((tokens, action_token))
    return examples


def collate_batch(batch, pad_id, act_id):
    max_len = max(len(item[0]) for item in batch) + 1
    input_ids = []
    labels = []
    for tokens, action_id in batch:
        seq = tokens + [act_id]
        seq_ids = seq + [pad_id] * (max_len - len(seq))
        label = [-1] * (len(seq) - 1) + [action_id]
        label += [-1] * (max_len - len(label))
        input_ids.append(seq_ids)
        labels.append(label)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_json(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    run_root = resolve_path(config_dir, cfg["paths"]["root"])

    dataset_dir = os.path.join(run_root, cfg["paths"]["dataset_dir"])
    report_dir = os.path.join(run_root, cfg["paths"]["report_dir"])
    os.makedirs(report_dir, exist_ok=True)

    eval_path = os.path.join(dataset_dir, cfg["dataset"]["eval_output"])
    if not os.path.exists(eval_path):
        raise FileNotFoundError("Missing eval dataset. Run gym_make_dataset.py first.")

    eval_rows = list(read_jsonl(eval_path))
    model_path = os.path.join(run_root, cfg["paths"]["model_dir"], "player_hrm.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Missing model. Run gym_train_player.py first.")

    payload = torch.load(model_path, map_location="cpu")
    vocab = payload["vocab"]
    config = payload["config"]

    model = HRM(
        networks=config["networks"],
        dim=config["dim"],
        num_tokens=len(vocab),
        reasoning_steps=config.get("reasoning_steps", 2),
        relative_period=config.get("relative_period", 2),
        causal=False,
        ignore_index=-1,
    )
    model.load_state_dict(payload["state_dict"])
    device = cfg.get("eval_config", {}).get("device", "cpu")
    model.to(device)
    model.eval()

    examples = build_examples(eval_rows, vocab)
    pad_id = vocab["<PAD>"]
    act_id = vocab["<ACT>"]

    batch_size = int(cfg.get("eval_config", {}).get("batch_size", 16))
    loader = DataLoader(
        EvalDataset(examples, vocab),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id, act_id),
    )

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            loss, _, pred = model(batch_inputs, labels=batch_labels)
            total_loss += loss.item()

            preds = pred.argmax(dim=-1)
            mask = batch_labels != -1
            if mask.any():
                correct += (preds[mask] == batch_labels[mask]).sum().item()
                total += mask.sum().item()

    avg_loss = total_loss / max(1, len(loader))
    acc = (correct / total) if total else 0.0

    metrics = {
        "run_id": cfg["run_id"],
        "eval_count": len(eval_rows),
        "eval_steps": total,
        "accuracy": round(acc, 6),
        "avg_loss": round(avg_loss, 6),
        "created_at": utc_now(),
    }

    metrics_path = os.path.join(report_dir, "metrics.json")
    write_json(metrics_path, metrics)

    manifest_path = os.path.join(run_root, "run_manifest.json")
    manifest = read_json(manifest_path)
    manifest["eval"] = {"created_at": utc_now(), "metrics": metrics_path}
    write_json(manifest_path, manifest)

    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
