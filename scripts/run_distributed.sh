#!/usr/bin/env bash
# Launch Agentic GRPO training on a 2 x A100/A800 node.
#
# Resource layout after CUDA_VISIBLE_DEVICES remapping:
#   - logical cuda:0 (physical GPU ${VLLM_PHYSICAL_GPU}) is reserved for vLLM rollout engine
#   - logical cuda:1 (physical GPU ${TRAIN_PHYSICAL_GPU}) is used by the GRPOTrainer/accelerate process
#
# Example:
#   bash scripts/run_distributed.sh --total-steps 100 --batch-size 4 --checkpoint-every 20
#
# You can also override defaults via env vars:
#   TOTAL_STEPS=200 BATCH_SIZE=2 CHECKPOINT_EVERY=25 bash scripts/run_distributed.sh


# 使用示例：


#  bash scripts/run_distributed.sh \
#    --total-steps 100 \
#    --batch-size 4 \
#    --checkpoint-every 20


# 也可以用环境变量：


#  TOTAL_STEPS=100 \
#  BATCH_SIZE=4 \
#  CHECKPOINT_EVERY=20 \
#  VLLM_PHYSICAL_GPU=0 \
#  TRAIN_PHYSICAL_GPU=1 \
#  bash scripts/run_distributed.sh


# 追加额外训练参数可用：


#  bash scripts/run_distributed.sh --total-steps 100 -- --override rollout.max_concurrency=2



set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_distributed.sh [options] [-- extra_train_args]

Options:
  --total-steps N        Override trainer.total_steps (default: env TOTAL_STEPS or 100)
  --batch-size N         Override rollout.batch_size and trainer.train_batch_size (default: env BATCH_SIZE or 4)
  --checkpoint-every N   Override checkpoint.save_every (default: env CHECKPOINT_EVERY or 50)
  --config PATH          Config path (default: configs/grpo_config.yaml)
  --vllm-gpu ID          Physical GPU id for vLLM engine (default: env VLLM_PHYSICAL_GPU or 0)
  --train-gpu ID         Physical GPU id for trainer process (default: env TRAIN_PHYSICAL_GPU or 1)
  -h, --help             Show this help

Any arguments after "--" are appended to scripts/train_agentic_grpo.py.
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-configs/grpo_config.yaml}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
VLLM_PHYSICAL_GPU="${VLLM_PHYSICAL_GPU:-0}"
TRAIN_PHYSICAL_GPU="${TRAIN_PHYSICAL_GPU:-1}"
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --total-steps)
      TOTAL_STEPS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --vllm-gpu)
      VLLM_PHYSICAL_GPU="$2"
      shift 2
      ;;
    --train-gpu)
      TRAIN_PHYSICAL_GPU="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ "${VLLM_PHYSICAL_GPU}" == "${TRAIN_PHYSICAL_GPU}" ]]; then
  echo "[ERROR] vLLM GPU and trainer GPU must be different physical devices." >&2
  echo "        Got --vllm-gpu ${VLLM_PHYSICAL_GPU} and --train-gpu ${TRAIN_PHYSICAL_GPU}." >&2
  exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERROR] accelerate is not available in PATH. Install it with: pip install accelerate" >&2
  exit 1
fi

if [[ ! -f "scripts/train_agentic_grpo.py" ]]; then
  echo "[ERROR] Training entrypoint not found: scripts/train_agentic_grpo.py" >&2
  exit 1
fi

# Remove stale Python bytecode before launch to avoid importing outdated modules.
echo "[INFO] Cleaning __pycache__ directories under ${PROJECT_ROOT} ..."
find "${PROJECT_ROOT}" -type d -name "__pycache__" -prune -exec rm -rf {} +

# Physical isolation: only the vLLM and trainer GPUs are exposed to this job.
# Inside the process they become cuda:0 and cuda:1 respectively.
export CUDA_VISIBLE_DEVICES="${VLLM_PHYSICAL_GPU},${TRAIN_PHYSICAL_GPU}"

# Keep CUDA allocator fragmentation lower on 40GB cards.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

OVERRIDES=(
  "trainer.total_steps=${TOTAL_STEPS}"
  "rollout.batch_size=${BATCH_SIZE}"
  "trainer.train_batch_size=${BATCH_SIZE}"
  "checkpoint.save_every=${CHECKPOINT_EVERY}"
  "trainer.gradient_accumulation_steps=4"
  "vllm.engine_args.device=cuda:0"
  "vllm.engine_args.gpu_memory_utilization=0.7"
  "vllm.engine_args.tensor_parallel_size=1"
  "trainer.device=cuda:1"
)

TRAIN_CMD=(
  accelerate launch
  --num_processes 1
  --num_machines 1
  --mixed_precision bf16
  scripts/train_agentic_grpo.py
  --config "${CONFIG_PATH}"
)

for override in "${OVERRIDES[@]}"; do
  TRAIN_CMD+=(--override "${override}")
done

TRAIN_CMD+=("${EXTRA_TRAIN_ARGS[@]}")

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] vLLM logical device: cuda:0 (physical GPU ${VLLM_PHYSICAL_GPU})"
echo "[INFO] Trainer logical device: cuda:1 (physical GPU ${TRAIN_PHYSICAL_GPU})"
echo "[INFO] total_steps=${TOTAL_STEPS}, batch_size=${BATCH_SIZE}, checkpoint_every=${CHECKPOINT_EVERY}"
echo "[INFO] vLLM gpu_memory_utilization=0.7, gradient_accumulation_steps=4"
echo "[INFO] Launch command:"
printf '  %q' "${TRAIN_CMD[@]}"
echo

exec "${TRAIN_CMD[@]}"
