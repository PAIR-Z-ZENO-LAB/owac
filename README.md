# OWAC: Optimizing RAW Representations for Robot Manipulation Across Lighting Conditions

[[`Release`](https://github.com/PAIR-Z-ZENO-LAB/owac/releases/tag/paper-v1.0.1)] [[`Encoder`](https://huggingface.co/Jeong-USYD/owac-j5-raw-encoder)] [[`Dataset`](https://huggingface.co/datasets/ToferFish/RawVLA-Bench)] [[`Sim Guide`](docs/paper-reproduction.md)] [[`Real-Robot Guide`](docs/real-robot-reproduction.md)] [[`BibTeX`](#citing-owac)]

PyTorch implementation, pretrained weights, training pipelines, and evaluation tools for **OWAC** (*Optimizing RAW Representations for Robot Manipulation Across Lighting Conditions*).

OWAC learns visuomotor manipulation policies directly from linear RAW sensor measurements without rendering intermediate RGB images. The repository covers self-supervised RAW encoder pretraining, joint language-conditioned policy training (`OWAC-Regression` and `OWAC-Flow`), LIBERO closed-loop simulation under fixed and switching illumination, and native 10-bit BGGR Bayer deployment on a physical robot.

<div align="center">
  <img src="docs/assets/teaser.png" alt="OWAC Overview" width="100%" />
  <br />
  <em>Overview of OWAC RAW encoder pretraining, joint policy adaptation, LIBERO evaluation across lighting regimes, and real-robot deployment.</em>
</div>

## Pretrained Models

<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>LIBERO Fixed Mean</th>
      <th>LIBERO All-5</th>
      <th>Switching</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>OWAC RAW Encoder (ViT-S/16)</td>
      <td align="right">22.2M</td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center"><a href="https://huggingface.co/Jeong-USYD/owac-j5-raw-encoder/tree/b042fdd6cd5066c8871fbaf19dc84a906acc9000">Hugging Face</a></td>
    </tr>
    <tr>
      <td>OWAC-Regression (10k EMA)</td>
      <td align="right">44.0M</td>
      <td align="right">82.00%</td>
      <td align="right">53.10%</td>
      <td align="right">78.09%</td>
      <td align="center"><a href="https://github.com/PAIR-Z-ZENO-LAB/owac/releases/download/paper-v1.0.1/owac-regression-10000.safetensors">safetensors</a></td>
    </tr>
    <tr>
      <td>OWAC-Flow (20k EMA)</td>
      <td align="right">48.9M</td>
      <td align="right">73.82%</td>
      <td align="right">37.50%</td>
      <td align="right">71.65%</td>
      <td align="center"><a href="https://github.com/PAIR-Z-ZENO-LAB/owac/releases/download/paper-v1.0.1/owac-flow-20000.safetensors">safetensors</a></td>
    </tr>
  </tbody>
</table>

All release checksums (`SHA256SUMS`), licenses, and verification notes are published in [`paper-v1.0.1`](https://github.com/PAIR-Z-ZENO-LAB/owac/releases/tag/paper-v1.0.1). Archived episode logs for all 40,000 LIBERO evaluation trials are included in [`research/experiments/results/paper-v1.0`](research/experiments/results/paper-v1.0).

## Dependencies & Installation

| Use Case | Requirements |
| :--- | :--- |
| **Offline Policy Inference & Scoring** | Linux, Git, `curl`, [`uv`](https://docs.astral.sh/uv/), Python 3.12 (`--extra paper`; CPU supported) |
| **Encoder & Policy Training** | Python 3.12, PyTorch 2.9.1 (CUDA / BF16 GPUs), `paper`, `encoder`, `raw-ssl`, `visualization`, `teleop-mcap` extras |
| **Closed-Loop LIBERO Simulation** | Separate Python 3.11 environment (`.venv-sim`), EGL/OpenGL drivers, pinned [RawVLA](https://github.com/ShuhongLL/RawVLA) & LIBERO assets |
| **Physical Robot Deployment** | Ubuntu 24.04, ROS 2 Jazzy, CUDA Python 3.12 (`real-robot` extra), NERO arm stack (`agilx-teleop-collection`) |

### Core environment (CPU inference & scoring)

```bash
git clone --branch paper-v1.0.1 https://github.com/PAIR-Z-ZENO-LAB/owac.git
cd owac
uv sync --python 3.12 --locked --extra paper
```

### CUDA environment (training & real robot)

```bash
uv venv --python 3.12 --seed .venv-cuda
.venv-cuda/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv-cuda/bin/python -m pip install -e '.[paper,encoder,raw-ssl,visualization,teleop-mcap]'
```

Add `real-robot` (`-e '.[real-robot]'`) on the robot host for ROS 2 / MCAP hardware deployment.

## Deployment & Inference

### Run a Released Policy

Download the released EMA checkpoints and verify checksums:

```bash
mkdir -p outputs/models
base=https://github.com/PAIR-Z-ZENO-LAB/owac/releases/download/paper-v1.0.1
for file in owac-regression-10000.safetensors owac-flow-20000.safetensors SHA256SUMS
do
  curl -fL --retry 3 "$base/$file" -o "outputs/models/$file"
done
(cd outputs/models && sha256sum -c SHA256SUMS)
```

Run offline CPU inference (the tokenizer and normalization statistics are bundled):

```bash
# Deterministic synthetic input (prints 12 × 7 delta action chunk)
.venv/bin/python -m owac.paper.infer \
  --checkpoint outputs/models/owac-regression-10000.safetensors

# Recorded RawVLA-Bench trajectory
dataset=https://huggingface.co/datasets/ToferFish/RawVLA-Bench/resolve/8586c594fff9ce38580e29745f074e82e96e74f7
sample=data/libero/raw/libero_10_no_noops_1.0.0_lerobot/episode_000001/variant_00.npz
curl -fL --retry 3 "$dataset/$sample" -o outputs/example.npz
.venv/bin/python -m owac.paper.infer \
  --episode outputs/example.npz --frame 0 \
  --checkpoint outputs/models/owac-flow-20000.safetensors \
  --output outputs/actions.npy
```

To extract standalone visual features (`384`-D CLS + `16×16×384` spatial patch descriptors) using the public RAW encoder:

```bash
.venv/bin/python -m owac.paper.encode \
  --episode outputs/example.npz \
  --encoder <PATH/TO/encoder.pt> \
  --output outputs/encoder-features.npz
```

### Real-Robot Deployment (Native 10-bit BGGR Bayer)

The repository provides the ROS 2 RAW10 camera bridge ([`hardware/nero/`](hardware/nero)), MCAP dataset converter (`owac.raw.teleop_mcap`), local vision/policy adaptation tools (`owac.paper.teleop`), and live robot inference consoles (`scripts/fruit-infer`, `scripts/bowl-infer`, `scripts/pouring-infer`). Physical demonstrations and field weights are private; see **[`docs/real-robot-reproduction.md`](docs/real-robot-reproduction.md)** for full hardware setup, local training, and robot deployment steps.

## Training

Training uses a single node with two BF16-capable NVIDIA GPUs (`torch.distributed.run`) and the [RawVLA-Bench](https://huggingface.co/datasets/ToferFish/RawVLA-Bench) dataset.

### 1. Prepare data and initialization assets

```bash
gpu="$PWD/.venv-cuda/bin/python"
"$gpu" -m owac.paper.download_data --dataset outputs/rawvla-bench
"$gpu" -m owac.paper.prepare --dataset outputs/rawvla-bench --work outputs/paper
```

Use `--domain libero` if training action policies directly from the public pretrained RAW encoder. Asset download commands and SHA-256 verification for the pretrained RAW encoder and `google/t5-efficient-tiny` are listed in [`docs/paper-reproduction.md`](docs/paper-reproduction.md#train-the-raw-encoder-and-both-action-policies).

### 2. Pretrain the RAW encoder (three-stage SSL)

Run the three sequential pretraining stages (Base RAW SSL → Exposure Consistency → Relations & Motion) using the prepared configs in `outputs/paper/`:

```bash
"$gpu" -m torch.distributed.run --standalone --nproc-per-node=2 --module \
  owac.training.raw_ssl_train \
  --config <PATH/TO/STAGE_CONFIG.json> \
  --output <PATH/TO/OUTPUT_DIR> --no-tracking
```

### 3. Train action policies and export Safetensors

```bash
for pair in regression/10000 flow/20000
do
  head=${pair%/*}
  step=${pair#*/}
  "$gpu" -m torch.distributed.run --standalone --nproc-per-node=2 --module \
    owac.training.compact_train \
    --config "outputs/paper/$head.json" \
    --output "outputs/paper/$head" --stop-step "$step" --offline
  "$gpu" -m owac.paper.checkpoint \
    "outputs/paper/$head/policy.pt" "outputs/paper/$head.safetensors"
done
```

For exact multi-stage training commands, resuming (`--resume`), and paired ablation configs ([`configs/paper/ablations`](configs/paper/ablations)), see **[`docs/paper-reproduction.md`](docs/paper-reproduction.md)**.

## Evaluation

### Re-score archived LIBERO rollouts (no simulator needed)

```bash
mkdir -p outputs/paper
manifest=experiments/simulation/libero/manifests/libero_manifest_50_init_states_10000_rollouts.json
curl -fL "https://raw.githubusercontent.com/ShuhongLL/RawVLA/ba9c56b04da2c4931b70d8844710f5928d43775b/$manifest" \
  -o outputs/paper/libero-manifest.json
for head in regression flow
do
  d=research/experiments/results/paper-v1.0/$head
  .venv/bin/python -m owac.paper.score \
    --manifest outputs/paper/libero-manifest.json \
    --fixed "$d/fixed" --switching "$d/switching"
done
```

### Closed-loop LIBERO simulation

After installing the simulator environment (`.venv-sim`) and pinned RawVLA checkout per [`docs/paper-reproduction.md#run-native-libero`](docs/paper-reproduction.md#run-native-libero), run fixed or switching lighting rollouts:

```bash
.venv/bin/python -m owac.paper.evaluate \
  --rawvla "$PWD/external/RawVLA" \
  --sim-python "$PWD/.venv-sim/bin/python" \
  --checkpoint outputs/models/owac-regression-10000.safetensors \
  --mode fixed --device cuda --output outputs/eval/regression-fixed
```

Use `--mode switching` for random lighting transitions and `--shards N --shard-index I` for multi-process parallel evaluation.

## License

- **Code:** OWAC-authored source code is released under the [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Policy Weights:** OWAC contributions to the released checkpoints are released under [CC BY-NC 4.0](licenses/CC-BY-NC-4.0.txt).
- **Third-Party Components:** Upstream modules, tokenizers, datasets, and simulators retain their original licenses; see [`licenses/README.md`](licenses/README.md).

## Citing OWAC

If you use this repository or its released checkpoints, please cite:

```bibtex
@software{owac2026,
  author  = {{OWAC contributors}},
  title   = {OWAC: Optimizing RAW Representations for Robot Manipulation Across Lighting Conditions},
  year    = {2026},
  version = {1.0.1},
  url     = {https://github.com/PAIR-Z-ZENO-LAB/owac}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.
