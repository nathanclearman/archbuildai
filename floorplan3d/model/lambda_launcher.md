# Lambda Labs training launcher

End-to-end recipe to fine-tune the floor plan VLM on Lambda Labs GPU Cloud
in one command. **Expected total cost: $50–$70** for a single training run.

---

## 1. One-time Lambda account setup

1. Sign up at [lambdalabs.com](https://lambdalabs.com/service/gpu-cloud).
2. Add a payment method and ~$75 of credit.
3. Go to **SSH Keys**. Either:
   - Upload your existing public key (from `C:\Users\YOU\.ssh\id_ed25519.pub` on Windows), or
   - Generate a new key in Lambda's UI and download the `.pem` file.

   On Windows PowerShell, if you don't have an SSH key yet:
   ```powershell
   ssh-keygen -t ed25519 -C "floorplan3d"
   # accept defaults, no passphrase needed for this
   cat ~/.ssh/id_ed25519.pub  # upload this to Lambda
   ```

## 2. Launch an instance

1. Go to **Instances → Launch instance**.
2. Choose:
   - **Region:** any (cost is flat across regions at time of writing)
   - **Instance type:** `gpu_1x_h100_pcie` (1× H100 80GB, ~$2.49/hr) — or
     `gpu_1x_h100_sxm5` if PCIe is sold out
   - **Filesystem:** **skip this for a one-shot run.** Lambda instances
     include several hundred GB of local NVMe, which is plenty for the
     ~20 GB this project needs. Only attach a persistent filesystem if
     you plan to stop/restart the pod across sessions — otherwise you'll
     pay $40/mo for storage you don't need. The bootstrap script
     auto-detects and falls back to `$HOME` if `/workspace` isn't mounted.
   - **SSH key:** pick the one you set up in step 1
3. Click **Launch**. The pod will be ready in 1–3 minutes. Copy the public
   IP from the dashboard.

## 3. Start training

SSH into the pod from Windows PowerShell:

```powershell
ssh ubuntu@<your-pod-ip>
```

On the pod, run the bootstrap in one line:

```bash
curl -sSL https://raw.githubusercontent.com/nathanclearman/archbuildai/claude/debug-detection-issues-1zk7j/floorplan3d/model/bootstrap.sh | bash
```

That script will:
1. Install system packages (git, wget, unzip, jq)
2. Clone this repo at the right branch
3. Install all Python ML dependencies
4. Download CubiCasa5k (~2 GB)
5. Generate 15,000 US-style synthetic floor plans
6. Start training on the attached H100

**Expected wall-clock:** 20–30 hours for training, plus ~15 min of setup.

## 4. Monitor training

From PowerShell, re-SSH in any time to check progress:

```powershell
ssh ubuntu@<your-pod-ip>
tmux attach    # if you started in tmux, otherwise tail the log file
```

Pro tip: wrap the bootstrap in `tmux` so you can close your laptop without
interrupting training:

```bash
tmux new -s train
curl -sSL https://raw.githubusercontent.com/nathanclearman/archbuildai/claude/debug-detection-issues-1zk7j/floorplan3d/model/bootstrap.sh | bash
# Ctrl+B then D to detach; come back later with `tmux attach -t train`
```

## 5. Pull the adapter back to your Windows desktop

When training finishes, from PowerShell on your desktop:

```powershell
cd C:\path\to\archbuildai
scp -r ubuntu@<your-pod-ip>:~/archbuildai/floorplan3d/model/weights floorplan3d\model\
```

Then commit and push the weights from your local repo (or store them
elsewhere — LoRA adapters are ~200 MB).

## 6. Terminate the instance

**IMPORTANT — you are billed by the minute.** After you've scp'd the
weights, go to the Lambda dashboard and click **Terminate**. The
persistent filesystem is cheap ($0.20/GB/month) so you can keep it around
for a second training run without re-downloading the dataset.

---

## Cost breakdown

| Stage                        | GPU-hours | Cost @ $2.49/hr |
|------------------------------|-----------|------------------|
| Bootstrap (deps, dataset, synth) | 0.5  | $1.25            |
| QLoRA training, 2 epochs, 7B | 20–24     | $50–$60          |
| Smoke tests + buffer         | 2         | $5               |
| **Total**                    | ~25       | **$55–$65**      |

Persistent filesystem (200 GB × $0.20/mo) adds $40/month if you leave it —
delete it after you're done if you don't plan another run soon.

## Customizing the bootstrap

Environment variables you can set before running bootstrap.sh:

| Var            | Default                             | What it does |
|----------------|-------------------------------------|--------------|
| `BRANCH`       | `claude/debug-detection-issues-1zk7j` | Git branch to check out |
| `SYNTH_COUNT`  | `15000`                             | Number of synthetic plans |
| `EPOCHS`       | `2`                                 | Training epochs |
| `BASE_MODEL`   | `Qwen/Qwen2.5-VL-7B-Instruct`       | Base model (try `Qwen/Qwen2.5-VL-3B-Instruct` for a cheaper smoke test) |
| `SKIP_TRAIN`   | `0`                                 | Set to `1` to prep data only, skip training |

Example — dry run, no training:
```bash
SKIP_TRAIN=1 bash <(curl -sSL .../bootstrap.sh)
```
