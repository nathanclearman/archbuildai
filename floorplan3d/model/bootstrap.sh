#!/usr/bin/env bash
# FloorPlan3D training bootstrap.
#
# Run this ONCE on a fresh Lambda or RunPod H100 pod to go from zero to
# "training in progress" in a single command:
#
#   curl -sSL https://raw.githubusercontent.com/nathanclearman/archbuildai/claude/debug-detection-issues-1zk7j/floorplan3d/model/bootstrap.sh | bash
#
# Or, after `git clone`:
#
#   bash floorplan3d/model/bootstrap.sh
#
# Expects: Ubuntu 22.04, CUDA 12.x, Python 3.10+, a mounted persistent
# volume at /workspace (Lambda and RunPod both do this by default).

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/nathanclearman/archbuildai.git}"
BRANCH="${BRANCH:-claude/debug-detection-issues-1zk7j}"
# Use /workspace if a persistent filesystem is mounted there; otherwise
# fall back to the user's home directory on the instance's local NVMe.
if [ -z "${WORKDIR:-}" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then
        WORKDIR=/workspace
    else
        WORKDIR="$HOME"
    fi
fi
SYNTH_COUNT="${SYNTH_COUNT:-10000}"
EPOCHS="${EPOCHS:-2}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

echo "=== FloorPlan3D bootstrap ==="
echo "repo:    $REPO_URL"
echo "branch:  $BRANCH"
echo "workdir: $WORKDIR"
echo "synth:   $SYNTH_COUNT samples"
echo "epochs:  $EPOCHS"
echo "base:    $BASE_MODEL"
echo

# --- 1. System deps -----------------------------------------------------
# Lambda instances run as the `ubuntu` user; RunPod pods run as root.
# Detect which and prefix apt-get with sudo only when needed.
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "--- installing system packages"
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    git git-lfs wget unzip jq ca-certificates > /dev/null
git lfs install --skip-smudge > /dev/null

# --- 2. Clone --------------------------------------------------------------
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ ! -d archbuildai ]; then
    echo "--- cloning repo"
    git clone "$REPO_URL"
fi
cd archbuildai
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH" || true

# --- 3. Python deps --------------------------------------------------------
echo "--- installing Python packages (this takes a few minutes)"
pip install --upgrade pip > /dev/null
pip install -r floorplan3d/model/requirements.txt

# --- 4. Download CubiCasa5k (~2GB) ----------------------------------------
cd floorplan3d/model
mkdir -p data
if [ ! -d data/cubicasa5k ]; then
    echo "--- downloading CubiCasa5k (~2GB, 3-10 min)"
    wget -q --show-progress \
        -O data/cubicasa5k.zip \
        https://zenodo.org/records/2613548/files/cubicasa5k.zip
    echo "--- unzipping"
    unzip -q data/cubicasa5k.zip -d data/cubicasa5k
    rm data/cubicasa5k.zip
else
    echo "--- CubiCasa5k already present, skipping download"
fi

# --- 5. Generate US-style synthetic data ----------------------------------
if [ ! -d data/synthetic ] || [ "$(ls -1 data/synthetic 2>/dev/null | wc -l)" -lt "$((SYNTH_COUNT / 2))" ]; then
    echo "--- generating $SYNTH_COUNT synthetic US plans (~5-15 min on CPU)"
    python synthesize.py --out data/synthetic --count "$SYNTH_COUNT"
else
    echo "--- synthetic set already present, skipping"
fi

# --- 6. Training ----------------------------------------------------------
if [ "$SKIP_TRAIN" = "1" ]; then
    echo "--- SKIP_TRAIN=1, stopping before training"
    echo
    echo "Run manually with:"
    echo "  cd $WORKDIR/archbuildai/floorplan3d/model"
    echo "  python train.py --cubicasa data/cubicasa5k --synthetic data/synthetic --out weights --epochs $EPOCHS"
    exit 0
fi

mkdir -p weights
echo "--- starting training (~20 GPU-hours on 1x H100)"
python train.py \
    --base "$BASE_MODEL" \
    --cubicasa data/cubicasa5k \
    --synthetic data/synthetic \
    --out weights \
    --epochs "$EPOCHS" \
    --batch-size 1 \
    --grad-accum 16 \
    --lr 2e-5

echo
echo "=== done ==="
echo
echo "Adapter + processor saved to:"
echo "  $WORKDIR/archbuildai/floorplan3d/model/weights/"
echo
echo "To copy back to your desktop, from your desktop run:"
echo "  scp -r ubuntu@<pod-ip>:$WORKDIR/archbuildai/floorplan3d/model/weights ./floorplan3d/model/"
