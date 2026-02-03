import argparse
import json
import os
import subprocess


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout or "") + (proc.stderr or ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyworld_dir", default=r"C:\projects\GPTStoryworld\storyworlds")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--mc_runs", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max_turns", type=int, default=25)
    parser.add_argument("--conductor_hrm", default=None)
    parser.add_argument("--player_hrm", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for entry in os.listdir(args.storyworld_dir):
        if not entry.endswith(".json"):
            continue
        sw_path = os.path.join(args.storyworld_dir, entry)
        base = os.path.splitext(entry)[0]
        bench_path = os.path.join(out_dir, f"{base}_builder.json")
        player_path = os.path.join(out_dir, f"{base}_player.json")

        cmd_bench = [
            "python", os.path.join(os.path.dirname(__file__), "benchmark_harness.py"),
            "--storyworld", sw_path,
            "--steps", str(args.steps),
            "--mc_runs", str(args.mc_runs),
            "--out", bench_path,
        ]
        if args.conductor_hrm:
            cmd_bench += ["--hrm_model", args.conductor_hrm, "--device", args.device]
        run(cmd_bench)

        cmd_player = [
            "python", os.path.join(os.path.dirname(__file__), "player_harness.py"),
            "--storyworld", sw_path,
            "--episodes", str(args.episodes),
            "--max_turns", str(args.max_turns),
            "--out", player_path,
        ]
        if args.player_hrm:
            cmd_player += ["--hrm_model", args.player_hrm, "--device", args.device]
        run(cmd_player)

        results.append({
            "storyworld": sw_path,
            "builder_report": bench_path,
            "player_report": player_path,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote batch index to {args.out}")


if __name__ == "__main__":
    main()
