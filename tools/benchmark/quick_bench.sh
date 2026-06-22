#!/bin/bash
# Quick-start benchmark script for Jetson Orin Nano
# Usage: ./tools/benchmark/quick_bench.sh [single-leaf|continuous] [image_dir]

set -e

# Default parameters
MODE="${1:-single-leaf}"
IMAGE_DIR="${2:-data/test_images}"
OUT_DIR="./bench_out_$(date +%Y%m%d_%H%M%S)"

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory not found: $IMAGE_DIR"
    echo "Usage: $0 [single-leaf|continuous] [image_dir]"
    exit 1
fi

# Check if conda env is activated
if [[ "$CONDA_DEFAULT_ENV" != *"mmdet310"* ]]; then
    echo "Activating mmdet310 environment..."
    eval "$(conda shell.bash hook)"
    conda activate /home/jetson/miniforge3/envs/mmdet310
fi

echo "========================================="
echo "Jetson Orin Nano Benchmark Quick Start"
echo "========================================="
echo "Mode: $MODE"
echo "Images: $IMAGE_DIR"
echo "Output: $OUT_DIR"
echo "========================================="
echo ""

# Stage-1 config
STAGE1_CONFIG="2025-11-17-01/my_config/mask2former.py"
STAGE1_CKPT="2025-11-17-01/work_dirs/mask2former/iter_5000.pth"

# Stage-2 config
STAGE2_CONFIG="2025-11-17-02/configs/my_model_configs/deeplabv3plus_all.py"
STAGE2_CKPT="2025-11-17-02/work_dirs/deeplabv3plus_all/iter_10000.pth"

# Check if checkpoints exist
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "Warning: Stage-1 checkpoint not found: $STAGE1_CKPT"
    echo "Please update STAGE1_CKPT in this script or provide the correct path."
fi

if [ ! -f "$STAGE2_CKPT" ]; then
    echo "Warning: Stage-2 checkpoint not found: $STAGE2_CKPT"
    echo "Please update STAGE2_CKPT in this script or provide the correct path."
fi

# Run benchmark
if [ "$MODE" = "single-leaf" ]; then
    echo "Running single-leaf benchmark (5 images, 1 warmup)..."
    python tools/benchmark/run_benchmark.py \
        --images "$IMAGE_DIR" \
        --output "$OUT_DIR" \
        --num-images 5 \
        --warmup 1 \
        --stage1-config "$STAGE1_CONFIG" \
        --stage1-checkpoint "$STAGE1_CKPT" \
        --stage2-config "$STAGE2_CONFIG" \
        --stage2-checkpoint "$STAGE2_CKPT" \
        --device cuda:0 \
        --score-thr 0.3

elif [ "$MODE" = "continuous" ]; then
    echo "Running continuous benchmark (5 minutes)..."
    python tools/benchmark/run_benchmark.py \
        --images "$IMAGE_DIR" \
        --out_dir "$OUT_DIR" \
        --mode continuous \
        --continuous_minutes 5 \
        --warmup 10 \
        --stage1_config "$STAGE1_CONFIG" \
        --stage1_checkpoint "$STAGE1_CKPT" \
        --stage2_config "$STAGE2_CONFIG" \
        --stage2_checkpoint "$STAGE2_CKPT" \
        --device cuda:0 \
        --seed 0 \
        --deterministic \
        --power_interval_ms 1000

else
    echo "Error: Invalid mode. Use 'single-leaf' or 'continuous'"
    exit 1
fi

echo ""
echo "========================================="
echo "Benchmark complete!"
echo "Results saved to: $OUT_DIR"
echo "========================================="
echo ""
echo "Output files:"
echo "  - $OUT_DIR/summary.json    (complete statistics)"
echo "  - $OUT_DIR/latency.csv     (per-image latencies)"
echo "  - $OUT_DIR/tegrastats.log  (raw power/temp data)"
echo ""
