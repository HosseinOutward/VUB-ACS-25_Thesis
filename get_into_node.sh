#!/usr/bin/env bash
set -e

PARTITION="COOP"
NODELIST="ETROCOOP01"
CPUS_PER_TASK="4"
MEM="10000M"
GRES="gpu:4090:1"
TIME="4:00:00"

VENV_ACTIVATE="~/venv/bin/activate"

srun -p "$PARTITION" -w "$NODELIST" -c "$CPUS_PER_TASK" --mem="$MEM" --gres="$GRES" -t "$TIME" --pty bash -lc "
  source '$VENV_ACTIVATE'
  exec bash -l
"
