#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-trainer_config.json}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

# Parse JSON with python to avoid jq dependency
read -r MAX_RAM_MB MAX_CPU_PCT <<<"$(python3 - <<PY
import json
cfg=json.load(open(r"$CONFIG_PATH"))
print(cfg.get("max_ram_mb", 2048), cfg.get("max_cpu_pct", 50))
PY
)"

# ulimit expects KB for -v
MAX_RAM_KB=$((MAX_RAM_MB * 1024))

RUN_CMD=(python3 train_stub.py --config "$CONFIG_PATH")

if command -v systemd-run >/dev/null 2>&1; then
  CPU_QUOTA="${MAX_CPU_PCT}%"
  MEM_MAX="${MAX_RAM_MB}M"
  echo "Running with systemd-run limits: MemoryMax=$MEM_MAX CPUQuota=$CPU_QUOTA"
  systemd-run --scope -p MemoryMax="$MEM_MAX" -p CPUQuota="$CPU_QUOTA" "${RUN_CMD[@]}"
else
  echo "Running with ulimit -v ${MAX_RAM_KB} KB (soft cap)"
  ulimit -v "$MAX_RAM_KB"
  "${RUN_CMD[@]}"
fi
