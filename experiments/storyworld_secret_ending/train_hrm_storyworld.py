import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from HRM.hrm import HRM


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<ACT>"]


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


@dataclass
class Example:
    tokens: list
    action_token: str


class StoryworldDataset(Dataset):
    def __init__(self, examples, vocab):
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        token_ids = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in ex.tokens]
        action_id = self.vocab.get(ex.action_token, self.vocab["<UNK>"])
        return token_ids, action_id


def tokenize(text):
    return text.replace("\n", " ").split()


def build_examples(jsonl_path):
    examples = []
    action_tokens = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for step in record["steps"]:
                obs_text = serialize_obs(step["obs"])
                tokens = tokenize(obs_text)
                action_token = f"ACTION::{step['action']}"
                action_tokens.add(action_token)
                examples.append(Example(tokens=tokens, action_token=action_token))
    return examples, action_tokens


def build_vocab(examples, action_tokens):
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for ex in examples:
        for tok in ex.tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    for tok in sorted(action_tokens):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def filter_examples_for_vocab(examples, vocab):
    allowed = set(vocab.keys())
    return [ex for ex in examples if ex.action_token in allowed]


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


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path, obj):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def write_json(path, obj):
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class IOMonitor:
    def __init__(self, max_mb_s):
        self.max_mb_s = max_mb_s
        self.proc = psutil.Process() if psutil else None
        self.last_ts = None
        self.last_bytes = None

    def sample(self):
        if not self.proc or not self.max_mb_s or self.max_mb_s <= 0:
            return None, False
        now = time.time()
        counters = self.proc.io_counters()
        total_bytes = counters.read_bytes + counters.write_bytes
        if self.last_ts is None:
            self.last_ts = now
            self.last_bytes = total_bytes
            return 0.0, False
        dt = max(1e-6, now - self.last_ts)
        delta = total_bytes - self.last_bytes
        rate_mb_s = (delta / dt) / (1024 * 1024)
        self.last_ts = now
        self.last_bytes = total_bytes
        return rate_mb_s, rate_mb_s > self.max_mb_s


def save_checkpoint(path, model, optimizer, vocab, config, step, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "vocab": vocab,
        "config": config,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init_model", default=None)
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--checkpoint_every_steps", type=int, default=0)
    parser.add_argument("--checkpoint_every_seconds", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_runtime_seconds", type=int, default=0)
    parser.add_argument("--io_mb_s", type=float, default=0.0)
    parser.add_argument("--log_jsonl", default="")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resume_latest", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples, action_tokens = build_examples(args.data)
    if args.max_examples and args.max_examples > 0:
        examples = examples[: args.max_examples]

    if args.resume_latest and args.checkpoint_dir and os.path.isdir(args.checkpoint_dir):
        candidates = []
        for name in os.listdir(args.checkpoint_dir):
            if name.startswith("step-") and name.endswith(".pt"):
                parts = name.replace("step-", "").replace(".pt", "")
                try:
                    step = int(parts)
                    candidates.append((step, name))
                except ValueError:
                    continue
        if candidates:
            candidates.sort()
            latest = candidates[-1][1]
            args.resume_checkpoint = os.path.join(args.checkpoint_dir, latest)

    payload = None
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location=args.device)
        vocab = payload["vocab"]
        examples = filter_examples_for_vocab(examples, vocab)
    elif args.init_model:
        payload = torch.load(args.init_model, map_location=args.device)
        vocab = payload["vocab"]
        examples = filter_examples_for_vocab(examples, vocab)
    else:
        vocab = build_vocab(examples, action_tokens)

    pad_id = vocab["<PAD>"]
    act_id = vocab["<ACT>"]

    dataset = StoryworldDataset(examples, vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id, act_id),
    )

    model_dim = args.dim
    if payload:
        config = payload["config"]
        model_dim = config["dim"]
        model = HRM(
            networks=config["networks"],
            dim=model_dim,
            num_tokens=len(vocab),
            reasoning_steps=config.get("reasoning_steps", 2),
            relative_period=config.get("relative_period", 2),
            causal=False,
            ignore_index=-1,
        ).to(args.device)
        model.load_state_dict(payload["state_dict"])
        networks = config["networks"]
    else:
        networks = [
            {"dim": args.dim, "depth": args.depth, "heads": args.heads},
            {"dim": args.dim, "depth": args.depth, "heads": args.heads},
        ]

        model = HRM(
            networks=networks,
            dim=model_dim,
            num_tokens=len(vocab),
            reasoning_steps=2,
            relative_period=2,
            causal=False,
            ignore_index=-1,
        ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    resume_step = 0
    resume_epoch = 0

    if args.resume_checkpoint and payload:
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        resume_step = int(payload.get("step") or 0)
        resume_epoch = int(payload.get("epoch") or 0)
        append_jsonl(
            args.log_jsonl,
            {
                "ts": utc_now(),
                "event": "resume",
                "checkpoint": args.resume_checkpoint,
                "step": payload.get("step"),
                "epoch": payload.get("epoch"),
            },
        )

    model.train()
    start = time.time()
    last_ckpt = start
    steps_completed = resume_step
    abort_reason = None
    io_monitor = IOMonitor(args.io_mb_s)

    append_jsonl(
        args.log_jsonl,
        {
            "ts": utc_now(),
            "event": "start",
            "data": args.data,
            "out": args.out,
            "caps": {"io_mb_s": args.io_mb_s},
        },
    )
    if args.io_mb_s and psutil is None:
        append_jsonl(
            args.log_jsonl,
            {
                "ts": utc_now(),
                "event": "io_monitor_unavailable",
                "reason": "psutil_not_installed",
            },
        )

    for epoch in range(resume_epoch, args.epochs):
        total_loss = 0.0
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(args.device)
            batch_labels = batch_labels.to(args.device)
            optimizer.zero_grad()
            loss, _, _ = model(batch_inputs, labels=batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps_completed += 1

            now = time.time()
            if args.max_steps and steps_completed >= args.max_steps:
                abort_reason = "max_steps"
                break
            if args.max_runtime_seconds and (now - start) >= args.max_runtime_seconds:
                abort_reason = "timeout"
                break

            rate_mb_s, over = io_monitor.sample()
            if rate_mb_s is not None:
                append_jsonl(
                    args.log_jsonl,
                    {
                        "ts": utc_now(),
                        "event": "io_sample",
                        "step": steps_completed,
                        "io_mb_s": round(rate_mb_s, 3),
                    },
                )
                if over:
                    abort_reason = "io_cap_exceeded"
                    break

            do_ckpt_step = args.checkpoint_every_steps and steps_completed % args.checkpoint_every_steps == 0
            do_ckpt_time = args.checkpoint_every_seconds and (now - last_ckpt) >= args.checkpoint_every_seconds
            if args.checkpoint_dir and (do_ckpt_step or do_ckpt_time):
                ckpt_path = os.path.join(args.checkpoint_dir, f"step-{steps_completed}.pt")
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    vocab,
                    {
                        "dim": model_dim,
                        "networks": networks,
                        "reasoning_steps": 2,
                        "relative_period": 2,
                    },
                    steps_completed,
                    epoch,
                )
                append_jsonl(
                    args.log_jsonl,
                    {
                        "ts": utc_now(),
                        "event": "checkpoint",
                        "step": steps_completed,
                        "path": ckpt_path,
                    },
                )
                last_ckpt = now

        if abort_reason:
            break
        avg_loss = total_loss / max(1, len(loader))
        print(f"epoch={epoch} loss={avg_loss:.4f}")

    payload = {
        "vocab": vocab,
        "config": {
            "dim": model_dim,
            "networks": networks,
            "reasoning_steps": 2,
            "relative_period": 2,
        },
        "state_dict": model.state_dict(),
    }
    if not abort_reason:
        torch.save(payload, args.out)

    status = "aborted" if abort_reason else "completed"
    append_jsonl(
        args.log_jsonl,
        {
            "ts": utc_now(),
            "event": "end",
            "status": status,
            "abort_reason": abort_reason,
            "steps_completed": steps_completed,
        },
    )
    write_json(
        args.summary_json,
        {
            "status": status,
            "abort_reason": abort_reason,
            "steps_completed": steps_completed,
            "wall_seconds": round(time.time() - start, 3),
        },
    )

    if abort_reason:
        sys.exit(2)


if __name__ == "__main__":
    main()
