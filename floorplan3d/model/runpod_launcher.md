# RunPod / Lambda training launcher

End-to-end recipe to fine-tune the floor plan VLM on a single H100 (80GB)
for roughly **$50–80 of compute**.

## 1. Launch the pod

- Provider: RunPod or Lambda (RunPod recommended for lower cost + persistent volumes)
- GPU: **1× H100 80GB SXM or PCIe**
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  (or any CUDA 12.x + PyTorch 2.3+ image)
- Disk: **150 GB** persistent volume (datasets + checkpoints)
- Region: cheapest available — training is not latency-sensitive

## 2. Clone and install

```bash
cd /workspace
git clone https://github.com/nathanclearman/archbuildai.git
cd archbuildai
pip install --upgrade pip
pip install -r floorplan3d/model/requirements.txt
```

## 3. Download datasets

```bash
cd floorplan3d/model

# CubiCasa5k (~2 GB)
mkdir -p data
wget -O data/cubicasa5k.zip \
  https://zenodo.org/records/2613548/files/cubicasa5k.zip
unzip -q data/cubicasa5k.zip -d data/cubicasa5k
rm data/cubicasa5k.zip

# Synthetic augmentation (CPU, ~10 min for 5k samples)
python synthesize.py --out data/synthetic --count 5000
```

ResPlan and CFP are optional — add them later if you want to push accuracy
further.

## 4. Train

```bash
# 7B profile, ~20 GPU-hours at $2.49/hr ≈ $50
python train.py \
  --base Qwen/Qwen2.5-VL-7B-Instruct \
  --cubicasa data/cubicasa5k \
  --synthetic data/synthetic \
  --out weights \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 16 \
  --lr 2e-5
```

Checkpoints save to `weights/` every 500 steps. Training can be resumed
with `--resume weights/checkpoint-<n>`.

## 5. Smoke test inference

```bash
python inference.py \
  --image /path/to/any_floor_plan.png \
  --weights weights \
  --output json | jq
```

If this prints valid floor plan JSON, training worked.

## 6. Pull the adapter home

From your local machine:

```bash
# Adapter is small — just the LoRA weights (~200 MB for r=32).
scp -r runpod:/workspace/archbuildai/floorplan3d/model/weights ./floorplan3d/model/
```

Then kill the pod. You pay only for training time, not idle time.

## 7. Local use

Point `LocalModelClient` at the weights directory (the default already
does: `floorplan3d/model/weights/`) and run the Blender add-on. Inference
on your M4 Max via MLX will be added in a follow-up — for now inference
runs via the same Qwen2.5-VL Transformers path used for training, which
works on both CUDA and MPS.

## Cost checkpoints

| Stage                           | GPU-hours | Cost @ $2.49/hr |
|---------------------------------|-----------|------------------|
| Data prep (CubiCasa + synth)    | 0.5       | $1               |
| 7B QLoRA, 2 epochs              | 18–22     | $45–55           |
| Eval + smoke tests              | 1–2       | $3–5             |
| **Total**                       | ~22       | **~$55**         |

Budget $100 to cover one reroll.
