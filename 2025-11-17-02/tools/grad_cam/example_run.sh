#!/bin/bash
# Example script to run Grad-CAM visualization on sample images

# Configuration
PROJECT_ROOT="/home/jetson/RB_REC_TOOL/2025-11-17-02"
CONFIG_DIR="${PROJECT_ROOT}/configs/my_model_configs"
WORK_DIR="${PROJECT_ROOT}/work_dirs"

# Model configs (in order: baseline, tr, sae, tr+sae)
CONFIGS=(
    "${CONFIG_DIR}/deeplabv3plus.py"
    "${CONFIG_DIR}/deeplabv3plus_tr.py"
    "${CONFIG_DIR}/deeplabv3plus_sae.py"
    "${CONFIG_DIR}/deeplabv3plus_all.py"
)

# Checkpoints (adjust paths as needed)
CHECKPOINTS=(
    "${WORK_DIR}/deeplabv3plus/best_mIoU_iter_10000.pth"
    "${WORK_DIR}/deeplabv3plus_tr/best_mIoU_iter_10000.pth"
    "${WORK_DIR}/deeplabv3plus_sae/best_mIoU_iter_10000.pth"
    "${WORK_DIR}/deeplabv3plus_all/best_mIoU_iter_10000.pth"
)

# Check if checkpoint paths need adjustment
echo "=== Grad-CAM Visualization Example ==="
echo ""
echo "Checking checkpoint files..."
for i in "${!CHECKPOINTS[@]}"; do
    if [ ! -f "${CHECKPOINTS[$i]}" ]; then
        echo "⚠️  Warning: Checkpoint not found: ${CHECKPOINTS[$i]}"
        echo "   Please update CHECKPOINTS array in this script with correct paths."
    else
        echo "✓ Found: ${CHECKPOINTS[$i]}"
    fi
done
echo ""

# Example 1: Single image with T1 target
echo "=== Example 1: Single Image (T1 - Global Lesion Mean) ==="
echo "This example assumes you have a test image."
echo ""
echo "Command:"
echo "python ${PROJECT_ROOT}/tools/grad_cam/run_grad_cam.py \\"
echo "    --configs ${CONFIGS[@]} \\"
echo "    --checkpoints ${CHECKPOINTS[@]} \\"
echo "    --input /path/to/your/test_image.jpg \\"
echo "    --target T1 \\"
echo "    --out-dir ${PROJECT_ROOT}/results/grad_cam_t1 \\"
echo "    --device cuda:0 \\"
echo "    --lesion-idx 1"
echo ""

# Example 2: Multiple images with T3 target
echo "=== Example 2: Multiple Images (T3 - Boundary-Weighted) ==="
echo "This example requires GT masks in a separate directory."
echo ""
echo "Command:"
echo "python ${PROJECT_ROOT}/tools/grad_cam/run_grad_cam.py \\"
echo "    --configs ${CONFIGS[@]} \\"
echo "    --checkpoints ${CHECKPOINTS[@]} \\"
echo "    --input /path/to/image_list.txt \\"
echo "    --target T3 \\"
echo "    --out-dir ${PROJECT_ROOT}/results/grad_cam_t3 \\"
echo "    --gt-dir /path/to/gt_masks \\"
echo "    --band-width 3 \\"
echo "    --device cuda:0 \\"
echo "    --lesion-idx 1"
echo ""

# Uncomment below to run Example 1 (update paths first!)
# python ${PROJECT_ROOT}/tools/grad_cam/run_grad_cam.py \
#     --configs "${CONFIGS[@]}" \
#     --checkpoints "${CHECKPOINTS[@]}" \
#     --input /path/to/your/test_image.jpg \
#     --target T1 \
#     --out-dir ${PROJECT_ROOT}/results/grad_cam_t1 \
#     --device cuda:0 \
#     --lesion-idx 1

echo "=== Instructions ==="
echo "1. Update checkpoint paths in CHECKPOINTS array if needed"
echo "2. Prepare your test image(s)"
echo "3. For T3 target, prepare GT mask directory"
echo "4. Uncomment the example command at the end of this script and update paths"
echo "5. Run: bash tools/grad_cam/example_run.sh"
echo ""
