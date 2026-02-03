import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from HRM.hrm import HRM


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<ACT>"]


@dataclass
class Example:
    tokens: list
    action_token: str


def tokenize(text):
    return text.replace("\n", " ").split()


def serialize_input(state: str, tools: list) -> str:
    return f"STATE: {state}\nTOOLS: {', '.join(tools)}"


class ConductorDataset(Dataset):
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


def build_examples(jsonl_path):
    examples = []
    action_tokens = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = serialize_input(record["state"], record["tools"])
            tokens = tokenize(text)
            action_token = f"ACTION::{record['action']}"
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
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples, action_tokens = build_examples(args.data)
    if args.max_examples and args.max_examples > 0:
        examples = examples[: args.max_examples]
    vocab = build_vocab(examples, action_tokens)
    pad_id = vocab["<PAD>"]
    act_id = vocab["<ACT>"]

    dataset = ConductorDataset(examples, vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id, act_id),
    )

    networks = [
        {"dim": args.dim, "depth": args.depth, "heads": args.heads},
        {"dim": args.dim, "depth": args.depth, "heads": args.heads},
    ]

    model = HRM(
        networks=networks,
        dim=args.dim,
        num_tokens=len(vocab),
        reasoning_steps=2,
        relative_period=2,
        causal=False,
        ignore_index=-1,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(args.device)
            batch_labels = batch_labels.to(args.device)
            optimizer.zero_grad()
            loss, _, _ = model(batch_inputs, labels=batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        print(f"epoch={epoch} loss={avg_loss:.4f}")

    payload = {
        "vocab": vocab,
        "config": {
            "dim": args.dim,
            "networks": networks,
            "reasoning_steps": 2,
            "relative_period": 2,
        },
        "state_dict": model.state_dict(),
    }
    torch.save(payload, args.out)


if __name__ == "__main__":
    main()
