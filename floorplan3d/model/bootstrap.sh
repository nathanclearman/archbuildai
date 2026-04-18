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
SYNTH_COUNT="${SYNTH_COUNT:-15000}"  # matches the 7B QLoRA corpus size target
EPOCHS="${EPOCHS:-2}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# Ubuntu 22.04 ships `python3` but not a `python` symlink by default.
# Prefer python3 so cloud images without the python-is-python3 package
# still work. Users can override with PYTHON=/path/to/venv/python for
# a specific env.
PYTHON="${PYTHON:-python3}"

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
    git git-lfs wget curl unzip jq ca-certificates > /dev/null
git lfs install --skip-smudge > /dev/null

# --- 2. Clone --------------------------------------------------------------
# fetch + checkout + pull must succeed. Previously `git pull ... || true`
# masked network failures so a stale checkout silently trained on a
# stale branch head — a debug nightmare on a $60/run pod. Fast-forward-
# only makes "already up-to-date" succeed without a merge prompt; a
# real divergence fails loudly.
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ ! -d archbuildai ]; then
    echo "--- cloning repo"
    git clone "$REPO_URL"
fi
cd archbuildai
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

# --- 3. Python deps --------------------------------------------------------
echo "--- installing Python packages (this takes a few minutes)"
$PYTHON -m pip install --upgrade pip > /dev/null
$PYTHON -m pip install -r floorplan3d/model/requirements.txt

# --- 4. Download CubiCasa5k (~2GB) ----------------------------------------
# Zenodo publishes the md5 for every file inside the record's JSON metadata.
# We pull the expected hash before the big download, verify the archive
# afterwards, and remove+exit on mismatch. Without this, a corrupted or
# MITMed archive silently unzips a partial dataset — the loader then drops
# malformed samples with a [skip] print and training proceeds on a
# fraction of the intended corpus.
CUBICASA_RECORD_ID="${CUBICASA_RECORD_ID:-2613548}"
CUBICASA_FILE="${CUBICASA_FILE:-cubicasa5k.zip}"

cd floorplan3d/model
mkdir -p data
if [ ! -d data/cubicasa5k ]; then
    echo "--- resolving expected md5 from Zenodo record $CUBICASA_RECORD_ID"
    EXPECTED_MD5=$(
        curl -sSL --fail \
            "https://zenodo.org/api/records/${CUBICASA_RECORD_ID}" \
        | jq -r \
            --arg f "$CUBICASA_FILE" \
            '.files[] | select(.key == $f) | .checksum' \
        | sed 's/^md5://'
    )
    if [ -z "$EXPECTED_MD5" ] || [ "$EXPECTED_MD5" = "null" ]; then
        echo "ERROR: could not resolve md5 for $CUBICASA_FILE from Zenodo" >&2
        echo "       record $CUBICASA_RECORD_ID. If the record was deleted" >&2
        echo "       or the file renamed, update CUBICASA_RECORD_ID /" >&2
        echo "       CUBICASA_FILE before rerunning." >&2
        exit 1
    fi
    echo "    expected md5: $EXPECTED_MD5"

    echo "--- downloading CubiCasa5k (~2GB, 3-10 min)"
    wget -q --show-progress \
        -O data/cubicasa5k.zip \
        "https://zenodo.org/records/${CUBICASA_RECORD_ID}/files/${CUBICASA_FILE}"

    echo "--- verifying md5"
    ACTUAL_MD5=$(md5sum data/cubicasa5k.zip | awk '{print $1}')
    if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
        echo "ERROR: md5 mismatch on data/cubicasa5k.zip" >&2
        echo "    expected: $EXPECTED_MD5" >&2
        echo "    got:      $ACTUAL_MD5" >&2
        # Remove the bad archive so a rerun doesn't see it as "already
        # present" and unzip corrupted data.
        rm -f data/cubicasa5k.zip
        exit 1
    fi
    echo "    md5 OK"

    echo "--- unzipping"
    unzip -q data/cubicasa5k.zip -d data/cubicasa5k
    rm data/cubicasa5k.zip
else
    echo "--- CubiCasa5k already present, skipping download"
fi

# --- 5. Generate US-style synthetic data ----------------------------------
# Each sample produces two files (.png + .json), so a complete corpus
# has exactly 2 * SYNTH_COUNT files. We require >= 95% of that before
# skipping regeneration: a partial run (process killed, disk full) that
# left e.g. 20000 files out of an expected 30000 would previously pass
# the old SYNTH_COUNT/2 threshold (7500) and silently train on a
# truncated corpus. 95% gives a tiny slack for fs noise without
# accepting a genuinely incomplete set.
SYNTH_TARGET_FILES=$((2 * SYNTH_COUNT))
SYNTH_MIN_FILES=$(((SYNTH_TARGET_FILES * 95) / 100))
if [ ! -d data/synthetic ] || [ "$(ls -1 data/synthetic 2>/dev/null | wc -l)" -lt "$SYNTH_MIN_FILES" ]; then
    echo "--- generating $SYNTH_COUNT synthetic US plans (~5-15 min on CPU)"
    $PYTHON synthesize.py --out data/synthetic --count "$SYNTH_COUNT"
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
$PYTHON train.py \
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
# Lambda pods run as `ubuntu`, RunPod pods as `root`. Print the user
# that invoked the script rather than hardcoding — users who
# copy-paste a hardcoded `ubuntu@` on RunPod hit a silent auth failure.
echo "To copy back to your desktop, from your desktop run:"
echo "  scp -r $(id -un)@<pod-ip>:$WORKDIR/archbuildai/floorplan3d/model/weights ./floorplan3d/model/"
