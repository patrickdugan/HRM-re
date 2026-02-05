import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from HRM.hrm import CombineHiddens
from x_transformers import Encoder

from .storyworld_state_dataset import StoryworldStateDataset, StoryworldStateSample


@dataclass
class Batch:
    state_t: torch.Tensor
    state_t_next: torch.Tensor
    operators: Optional[torch.Tensor]
    state_delta: Optional[torch.Tensor]
    encounter_acceptability: Optional[torch.Tensor]
    encounter_desirability: Optional[torch.Tensor]
    option_visibility: Optional[torch.Tensor]
    option_performability: Optional[torch.Tensor]
    reaction_desirability: Optional[torch.Tensor]
    ending_scores: Optional[torch.Tensor]
    ending_acceptability: Optional[torch.Tensor]
    ending_structural_distance: Optional[torch.Tensor]
    ending_metric_distance: Optional[torch.Tensor]
    world_grades: Optional[torch.Tensor]


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
    def __init__(self, max_mb_s, grace_seconds=5.0):
        self.max_mb_s = max_mb_s
        self.grace_seconds = grace_seconds
        self.proc = psutil.Process() if psutil else None
        self.last_ts = None
        self.last_bytes = None
        self.over_start = None
        self.peak_mb_s = 0.0

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
        self.peak_mb_s = max(self.peak_mb_s, rate_mb_s)
        if rate_mb_s > self.max_mb_s:
            if self.over_start is None:
                self.over_start = now
            elif (now - self.over_start) >= self.grace_seconds:
                return rate_mb_s, True
        else:
            self.over_start = None
        return rate_mb_s, False


class ResourceMonitor:
    def __init__(self):
        self.proc = psutil.Process() if psutil else None
        self.samples = 0
        self.ram_total = 0.0
        self.cpu_total = 0.0
        self.peak_ram = 0.0

    def sample(self):
        if not self.proc:
            return None
        mem = self.proc.memory_info().rss / (1024 * 1024)
        cpu = self.proc.cpu_percent(interval=None)
        self.samples += 1
        self.ram_total += mem
        self.cpu_total += cpu
        self.peak_ram = max(self.peak_ram, mem)
        return mem, cpu

    def summary(self):
        if self.samples == 0:
            return 0.0, 0.0, 0.0
        avg_ram = self.ram_total / self.samples
        avg_cpu = self.cpu_total / self.samples
        return self.peak_ram, avg_ram, avg_cpu


class HRMContinuous(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        state_dim: int,
        networks: List[dict],
        reasoning_steps: int = 2,
        relative_period: int = 2,
        num_ops: int = 0,
        num_endings: int = 0,
        num_grades: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim
        self.to_input = nn.Linear(state_dim, dim)
        self.reasoning_steps = reasoning_steps

        self.networks = nn.ModuleList([Encoder(**cfg) for cfg in networks])
        self.num_networks = len(self.networks)

        if not isinstance(relative_period, tuple):
            relative_period = (relative_period,) * (self.num_networks - 1)
        if len(relative_period) == (self.num_networks - 1):
            relative_period = (1, *relative_period)
        self.evaluate_networks_at = torch.tensor(relative_period).cumprod(dim=-1).tolist()
        self.lowest_steps_per_reasoning_step = self.evaluate_networks_at[-1]

        self.hidden_combiners = nn.ModuleList(
            [CombineHiddens(dim, self.num_networks + 1 - network_index) for network_index in range(self.num_networks)]
        )

        self.to_state = nn.Linear(dim, state_dim)
        self.to_ops = nn.Linear(dim, num_ops) if num_ops > 0 else None
        self.to_scripts = nn.Linear(dim, 5)
        self.to_endings = nn.Linear(dim, num_endings) if num_endings > 0 else None
        self.to_endings_accept = nn.Linear(dim, num_endings) if num_endings > 0 else None
        self.to_endings_struct = nn.Linear(dim, num_endings) if num_endings > 0 else None
        self.to_endings_metric = nn.Linear(dim, num_endings) if num_endings > 0 else None
        self.to_grades = nn.Linear(dim, num_grades) if num_grades > 0 else None

    def forward(self, state_vec: torch.Tensor):
        tokens = self.to_input(state_vec).unsqueeze(1)
        hiddens = torch.zeros_like(tokens)
        hiddens = hiddens.repeat(self.num_networks, 1, 1, 1)
        hiddens_dict = {index: hidden for index, hidden in enumerate(hiddens)}

        total_low_steps = self.reasoning_steps * self.lowest_steps_per_reasoning_step
        for index in range(total_low_steps):
            iteration = index + 1
            is_last_step = index == (total_low_steps - 1)
            context = torch.no_grad() if not is_last_step else torch.enable_grad()
            with context:
                for network_index, (network, hidden_combine, evaluate_network_at) in enumerate(
                    zip(self.networks, self.hidden_combiners, self.evaluate_networks_at)
                ):
                    if iteration % evaluate_network_at != 0:
                        continue
                    all_hiddens = (tokens, *hiddens_dict.values())
                    combined_input = hidden_combine(all_hiddens, network_index)
                    next_hidden = network(combined_input)
                    hiddens_dict[network_index] = next_hidden

        highest = hiddens_dict[self.num_networks - 1].squeeze(1)
        state_pred = self.to_state(highest)
        ops_pred = self.to_ops(highest) if self.to_ops else None
        scripts_pred = self.to_scripts(highest)
        endings_pred = self.to_endings(highest) if self.to_endings else None
        endings_accept = self.to_endings_accept(highest) if self.to_endings_accept else None
        endings_struct = self.to_endings_struct(highest) if self.to_endings_struct else None
        endings_metric = self.to_endings_metric(highest) if self.to_endings_metric else None
        grades_pred = self.to_grades(highest) if self.to_grades else None

        return (
            state_pred,
            ops_pred,
            scripts_pred,
            endings_pred,
            endings_accept,
            endings_struct,
            endings_metric,
            grades_pred,
        )


def collate_batch(samples: List[StoryworldStateSample]) -> Batch:
    state_t = torch.stack([s.state_t for s in samples], dim=0)
    state_t_next = torch.stack([s.state_t_next for s in samples], dim=0)
    operators = torch.stack([s.operators for s in samples], dim=0) if samples[0].operators is not None else None
    state_delta = (
        torch.stack([s.state_delta for s in samples], dim=0) if samples[0].state_delta is not None else None
    )

    def stack_optional(values: List[Optional[float]]) -> Optional[torch.Tensor]:
        if values and values[0] is not None:
            return torch.tensor(values, dtype=torch.float32)
        return None

    encounter_acceptability = stack_optional([s.encounter_acceptability for s in samples])
    encounter_desirability = stack_optional([s.encounter_desirability for s in samples])
    option_visibility = stack_optional([s.option_visibility for s in samples])
    option_performability = stack_optional([s.option_performability for s in samples])
    reaction_desirability = stack_optional([s.reaction_desirability for s in samples])

    ending_scores = (
        torch.stack([s.ending_scores for s in samples], dim=0) if samples[0].ending_scores is not None else None
    )
    ending_acceptability = (
        torch.stack([s.ending_acceptability for s in samples], dim=0)
        if samples[0].ending_acceptability is not None
        else None
    )
    ending_structural_distance = (
        torch.stack([s.ending_structural_distance for s in samples], dim=0)
        if samples[0].ending_structural_distance is not None
        else None
    )
    ending_metric_distance = (
        torch.stack([s.ending_metric_distance for s in samples], dim=0)
        if samples[0].ending_metric_distance is not None
        else None
    )
    world_grades = (
        torch.stack([s.world_grades for s in samples], dim=0) if samples[0].world_grades is not None else None
    )

    return Batch(
        state_t=state_t,
        state_t_next=state_t_next,
        operators=operators,
        state_delta=state_delta,
        encounter_acceptability=encounter_acceptability,
        encounter_desirability=encounter_desirability,
        option_visibility=option_visibility,
        option_performability=option_performability,
        reaction_desirability=reaction_desirability,
        ending_scores=ending_scores,
        ending_acceptability=ending_acceptability,
        ending_structural_distance=ending_structural_distance,
        ending_metric_distance=ending_metric_distance,
        world_grades=world_grades,
    )


def save_checkpoint(path, model, optimizer, config, step, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--checkpoint_every_steps", type=int, default=100)
    parser.add_argument("--checkpoint_every_seconds", type=int, default=60)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--max_runtime_seconds", type=int, default=28800)
    parser.add_argument("--io_mb_s", type=float, default=50.0)
    parser.add_argument("--io_grace_seconds", type=float, default=5.0)
    parser.add_argument("--log_jsonl", default="")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resume_latest", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = StoryworldStateDataset(
        inputs=[args.data],
        max_vars=1024,
        include_scripts=True,
        include_endings=True,
        include_grades=True,
        grades_mc_runs=0,
    )

    if args.max_examples and args.max_examples > 0:
        dataset.samples = dataset.samples[: args.max_examples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    networks = [
        {"dim": args.dim, "depth": args.depth, "heads": args.heads},
        {"dim": args.dim, "depth": args.depth, "heads": args.heads},
    ]

    model = HRMContinuous(
        dim=args.dim,
        state_dim=dataset.max_vars,
        networks=networks,
        reasoning_steps=2,
        relative_period=2,
        num_ops=len(dataset.op_basis),
        num_endings=len(dataset.ending_ids),
        num_grades=len(dataset.grade_keys),
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_fp16 = bool(args.fp16 and args.device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    resume_step = 0
    resume_epoch = 0
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location=args.device)
        model.load_state_dict(payload["state_dict"])
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
    io_monitor = IOMonitor(args.io_mb_s, grace_seconds=args.io_grace_seconds)
    res_monitor = ResourceMonitor()

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

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    loss_weights = {
        "state": 1.0,
        "ops": 0.5,
        "scripts": 0.2,
        "endings": 0.2,
        "grades": 0.2,
    }

    for epoch in range(resume_epoch, args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = Batch(**{k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.__dict__.items()})
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                (
                    state_pred,
                    ops_pred,
                    scripts_pred,
                    endings_pred,
                    endings_accept,
                    endings_struct,
                    endings_metric,
                    grades_pred,
                ) = model(batch.state_t)

                loss = mse(state_pred, batch.state_t_next) * loss_weights["state"]

                if ops_pred is not None and batch.operators is not None:
                    loss = loss + bce(ops_pred, batch.operators) * loss_weights["ops"]

                script_targets = [
                    batch.encounter_acceptability,
                    batch.encounter_desirability,
                    batch.option_visibility,
                    batch.option_performability,
                    batch.reaction_desirability,
                ]
                if all(t is not None for t in script_targets):
                    scripts_target = torch.stack(script_targets, dim=1)
                    loss = loss + mse(scripts_pred, scripts_target) * loss_weights["scripts"]

                if endings_pred is not None and batch.ending_scores is not None:
                    loss = loss + mse(endings_pred, batch.ending_scores) * loss_weights["endings"]
                if endings_accept is not None and batch.ending_acceptability is not None:
                    loss = loss + mse(endings_accept, batch.ending_acceptability) * loss_weights["endings"]
                if endings_struct is not None and batch.ending_structural_distance is not None:
                    loss = loss + mse(endings_struct, batch.ending_structural_distance) * loss_weights["endings"]
                if endings_metric is not None and batch.ending_metric_distance is not None:
                    loss = loss + mse(endings_metric, batch.ending_metric_distance) * loss_weights["endings"]

                if grades_pred is not None and batch.world_grades is not None:
                    loss = loss + mse(grades_pred, batch.world_grades) * loss_weights["grades"]

            if use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            steps_completed += 1

            now = time.time()
            res_monitor.sample()

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
                    {"ts": utc_now(), "event": "io_sample", "step": steps_completed, "io_mb_s": round(rate_mb_s, 3)},
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
                    {
                        "dim": args.dim,
                        "depth": args.depth,
                        "heads": args.heads,
                        "state_dim": dataset.max_vars,
                        "networks": networks,
                        "reasoning_steps": 2,
                        "relative_period": 2,
                        "op_basis": dataset.op_basis,
                        "ending_ids": dataset.ending_ids,
                        "grade_keys": dataset.grade_keys,
                    },
                    steps_completed,
                    epoch,
                )
                append_jsonl(
                    args.log_jsonl,
                    {"ts": utc_now(), "event": "checkpoint", "step": steps_completed, "path": ckpt_path},
                )
                last_ckpt = now

        if abort_reason:
            break
        avg_loss = total_loss / max(1, len(loader))
        print(f"epoch={epoch} loss={avg_loss:.4f}")

    payload = {
        "config": {
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "state_dim": dataset.max_vars,
            "networks": networks,
            "reasoning_steps": 2,
            "relative_period": 2,
            "op_basis": dataset.op_basis,
            "ending_ids": dataset.ending_ids,
            "grade_keys": dataset.grade_keys,
        },
        "state_dict": model.state_dict(),
    }
    if not abort_reason:
        torch.save(payload, args.out)

    peak_ram, avg_ram, avg_cpu = res_monitor.summary()
    status = "aborted" if abort_reason else "completed"
    append_jsonl(
        args.log_jsonl,
        {"ts": utc_now(), "event": "end", "status": status, "abort_reason": abort_reason, "steps_completed": steps_completed},
    )
    write_json(
        args.summary_json,
        {
            "status": status,
            "abort_reason": abort_reason,
            "steps_completed": steps_completed,
            "wall_seconds": round(time.time() - start, 3),
            "peak_ram_mb": round(peak_ram, 2),
            "avg_ram_mb": round(avg_ram, 2),
            "peak_io_mb_s": round(io_monitor.peak_mb_s, 3),
            "cpu_pct": round(avg_cpu, 2),
        },
    )

    if abort_reason:
        sys.exit(2)


if __name__ == "__main__":
    main()
