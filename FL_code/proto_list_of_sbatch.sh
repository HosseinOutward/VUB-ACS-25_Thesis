#!/usr/bin/env bash
set -euo pipefail

# queue_fl_jobs.sh
# Generates per-job sbatch scripts from a template and submits them.

# ---------- Config you might tweak ----------
PARTITION="COOP"
NODELIST="ETROCOOP02"
CPUS_PER_TASK="12"
MEM="45000M"
GRES="gpu:4090:1"
TIME="48:00:00"

VENV_ACTIVATE="$HOME/venv/bin/activate"
PYTHON_CMD="python run_fl.py"

# Where to write generated sbatch scripts
SBATCH_DIR="sbatch/generated"
# Where Slurm should write logs (matches your original structure)
LOG_DIR="sbatch/sbatch_log"

# ---------- Your tuple list (edit here) ----------
# Format: ("codec" port)
# identity, basic, ?_split_codec (2,3,...), debug_CancerWithBoundCalc
# non_wz_learned, cancer (_w_outlier, _basic_norm, _binary)
JOBS=(
#  "5_split_codec 29505"
#  "temporal_only 29560"
  "cancer 29580"
#  "marginal_cancer_binary 29605"
#  "cancer_basic_norm 29620"
#  "non_wz_learned_w_outlier 29640"
#  "marginal_cancer 29660"
)

# ---------- Safety checks ----------
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found in PATH."; exit 1; }
[[ -f "$VENV_ACTIVATE" ]] || { echo "ERROR: venv activate script not found at: $VENV_ACTIVATE"; exit 1; }

mkdir -p "$SBATCH_DIR" "$LOG_DIR"

# Optional: simple port sanity check (warn only)
is_port_in_use() {
  local p="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | awk '{print $4}' | grep -qE ":${p}$"
  else
    netstat -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${p}$"
  fi
}

echo "Generating + submitting ${#JOBS[@]} jobs..."
echo

for entry in "${JOBS[@]}"; do
  codec="$(awk '{print $1}' <<<"$entry")"
  port="$(awk '{print $2}' <<<"$entry")"

  if [[ -z "${codec}" || -z "${port}" ]]; then
    echo "Skipping malformed entry: '$entry'"
    continue
  fi

  if ! [[ "$port" =~ ^[0-9]+$ ]] || ((port < 1024 || port > 65535)); then
    echo "ERROR: invalid port '$port' for codec '$codec'"
    exit 1
  fi

  if is_port_in_use "$port"; then
    echo "WARNING: port $port appears to be in use on this machine (may still be fine on compute node)."
  fi

  ts="$(date +%Y%m%d_%H%M%S)"
  sbatch_file="${SBATCH_DIR}/run_${codec}_${port}_${ts}.sbatch"

  cat >"$sbatch_file" <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodelist=${NODELIST}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres=${GRES}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/run_%A.log
#SBATCH --error=${LOG_DIR}/error_%A.log
#SBATCH --job-name=${codec}

set -euo pipefail

source "${VENV_ACTIVATE}"
${PYTHON_CMD} --codec "${codec}" --master-port "${port}"
EOF

  chmod +x "$sbatch_file"

  echo "Submitting: $sbatch_file"
  job_out="$(sbatch "$sbatch_file")"
  echo "  -> ${job_out}"
  echo
done

echo "Done."
