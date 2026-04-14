# FloorPlan3D — Windows Quick Start

Everything you need on your Windows desktop to go from zero to a trained
model on Lambda. Total time on your desktop: ~10 minutes.

## 1. Install prerequisites

Open **PowerShell** (not Command Prompt) and run:

```powershell
# Git (skip if already installed)
winget install --id Git.Git -e

# OpenSSH client — usually already present on Windows 11
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH*'

# If the above shows OpenSSH.Client as NotPresent:
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
```

Close and reopen PowerShell after installing Git.

## 2. Clone the repo

```powershell
cd ~\Documents       # or wherever you keep code
git clone https://github.com/nathanclearman/archbuildai.git
cd archbuildai
git checkout claude/debug-detection-issues-IiUC7
```

## 3. Generate an SSH key (if you don't have one)

```powershell
ssh-keygen -t ed25519 -C "floorplan3d"
# press Enter through all prompts — no passphrase needed
cat ~\.ssh\id_ed25519.pub
```

Copy the printed line. You'll paste it into Lambda's SSH Keys page in the
next step.

## 4. Launch training on Lambda

Follow [`floorplan3d/model/lambda_launcher.md`](floorplan3d/model/lambda_launcher.md)
for the full walkthrough. Short version:

1. [Sign up at Lambda Labs](https://lambdalabs.com/service/gpu-cloud), add
   ~$75 of credit, paste your SSH key.
2. Launch `gpu_1x_h100_pcie` with a 200 GB persistent filesystem at
   `/workspace`.
3. SSH in from PowerShell:
   ```powershell
   ssh ubuntu@<your-pod-ip>
   ```
4. On the pod, run:
   ```bash
   tmux new -s train
   curl -sSL https://raw.githubusercontent.com/nathanclearman/archbuildai/claude/debug-detection-issues-IiUC7/floorplan3d/model/bootstrap.sh | bash
   ```
5. Ctrl+B then D to detach. The pod trains for ~20–25 hours.

## 5. Pull the weights back

When training finishes (~$50–65 of compute), from PowerShell on your desktop:

```powershell
cd ~\Documents\archbuildai
scp -r ubuntu@<your-pod-ip>:/workspace/archbuildai/floorplan3d/model/weights floorplan3d\model\
```

## 6. Terminate the Lambda pod

Go to the Lambda dashboard and click **Terminate** on the instance. You
are billed per minute until you do this. The persistent filesystem is a
separate small charge ($40/month for 200 GB) that you can also delete if
you're not planning another run soon.

## 7. Test the trained model locally

You don't have a GPU on Windows and the Qwen2.5-VL inference path in this
repo currently requires CUDA or Apple MPS. Two options:

- **Option A:** Rent a second (cheap) Lambda pod for inference only, or
  keep the training pod running briefly to validate before terminating.
- **Option B:** If you have an Apple Silicon Mac, we can add an MLX
  inference path that runs the model locally there. Ask me to scaffold it.
- **Option C:** Point the Blender add-on at the pod's IP via the
  `LocalModelClient(python_bin=...)` hook — turns your pod into a
  temporary inference server.

---

## File map (what's what)

| Path                                     | Purpose |
|------------------------------------------|---------|
| `floorplan3d/blender_addon/`             | Blender add-on (UI, geometry generation) |
| `floorplan3d/model/bootstrap.sh`         | One-command GPU-box setup + training |
| `floorplan3d/model/lambda_launcher.md`   | Lambda Labs walkthrough |
| `floorplan3d/model/runpod_launcher.md`   | RunPod alternative |
| `floorplan3d/model/schema.py`            | Canonical floor plan JSON schema |
| `floorplan3d/model/cv_walls.py`          | OpenCV wall extractor (no model needed) |
| `floorplan3d/model/synthesize.py`        | US-style synthetic plan generator |
| `floorplan3d/model/dataset.py`           | CubiCasa5k + synthetic loader |
| `floorplan3d/model/train.py`             | Qwen2.5-VL-7B QLoRA fine-tune |
| `floorplan3d/model/inference.py`         | Inference entry point |
| `floorplan3d/model/claude_refiner.py`    | Optional Claude Opus refinement pass |
| `floorplan3d/tests/`                     | Unit tests (run: `python -m unittest discover floorplan3d/tests`) |

## Troubleshooting

**`ssh` says "Permission denied (publickey)"** — the key on Lambda doesn't
match the one on your desktop. Re-upload the `.pub` file, wait 1 min, retry.

**`scp` is slow** — try `-C` for compression or use `rsync -az` instead.
The adapter is ~200 MB so it should transfer in under a minute even on
slow links.

**Training OOMs** — reduce `--batch-size` to 1 (already default) and
`--grad-accum` to 8 in the bootstrap invocation, or switch to the 3B base
model via `BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct bash bootstrap.sh`.

**The bootstrap fails on `pip install`** — some Lambda images come with an
older pip. Run `pip install --upgrade pip setuptools wheel` manually, then
re-run the bootstrap.
