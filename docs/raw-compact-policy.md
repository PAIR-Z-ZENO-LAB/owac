# C1 training and evaluation

This guide describes the entry points used by the released C1 experiment. The implementation and result summaries are public; the original J5 initialization, policy checkpoints, complete dataset inventory, and assembled simulator runtime remain in the experiment archive. The checked-in configuration records that environment's paths and must be adapted before use on another machine.

## Environments

The core development environment is installed with `uv sync --locked`. It is separate from the GPU policy runtime and the LIBERO simulation runtime. In particular, the repository's optional `encoder` extra pins a CPU PyTorch build for earlier local numerical work; it is not the CUDA environment used for C1.

| Runtime | Recorded configuration |
|---|---|
| Policy training and inference | Python 3.12.3, PyTorch 2.12.1+cu132, Transformers 4.57.0, SwanLab 0.10.3 |
| Compute | Two RTX 5090 GPUs with 32 GiB each; BF16 autocast |
| Simulation | Separate Python 3.11 environment with the archived LIBERO/robosuite, RawVLA adapters, EGL libraries, and OpenPI client |

Install OWAC into an already configured policy environment without replacing that environment's CUDA packages:

```bash
python -m pip install --no-deps -e .
```

The core dependency lock does not reconstruct the external simulator, drivers, checkpoints, or full policy runtime. Runtime provenance and the actual executed configuration are recorded in the [experiment log](../research/experiments/raw_compact_c1_20261104.md) and [release notes](releases/v1.0.md).

## Data and external assets

Use the LIBERO subset of [RawVLA-Bench](https://huggingface.co/datasets/ToferFish/RawVLA-Bench). The released run uses the original source-level split: 1,421 training sources, 177 validation sources, and 173 held-out sources. It trains on the five lighting domains with 211,114 training frames.

Copy the configuration into a local directory before setting paths:

```bash
mkdir -p configs/local
cp configs/experiments/raw_compact_c1.json configs/local/raw_compact_c1.json
```

| Configuration field | Required asset |
|---|---|
| `assets.j5` | Original J5 RAW-only step-12818 encoder checkpoint for initialization |
| `assets.tokenizer` | Local T5 tokenizer files matching the released configuration |
| `data.inventory` | Dataset inventory with local trajectory paths and the original source splits |
| `simulation.author_root` | RawVLA checkout containing its LIBERO evaluation adapters |
| `simulation.openpi_root` | OpenPI source and `openpi-client` package |
| `simulation.legacy_root` | Archived runtime containing simulator overlays, LIBERO source/config, and EGL libraries |
| `simulation.python` | Interpreter for that simulation runtime |
| `swanlab.id`, `swanlab.name` | A separate experiment identity for a new run |

The dataset inventory contract is implemented in [`load_inventory`](../src/owac/raw/all_bench_distillation.py), and C1's observation/action loader is [`CompactFrames`](../src/owac/training/compact_data.py). Training reads `agentview_raw_uint8`, `wrist_raw_uint8`, robot state, action, and task-description fields. It does not read paired RGB members. Robot state is taken from the seven joint angles and two finger positions at the same pre-action time.

Prepare state normalization, action quantiles, and task-language mappings from training sources only:

```bash
python -m owac.training.compact_data \
  --config configs/local/raw_compact_c1.json \
  --output outputs/c1-reproduction/config.json
```

The input dataset and J5 checkpoint must already exist; this command does not download them or recreate the original inventory. Replacing the initialization or split changes the experiment.

## Training

The trainer uses two distributed workers. The recorded global batch is 256: 64 examples per GPU, two GPUs, and two gradient-accumulation steps. The schedule is 80,000 optimizer updates, with learning rate 5e-5, 1,000 warmup updates, and EMA decay 0.999.

To run training and offline validation without cloud logging in the configured GPU environment:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m owac.training.compact_train \
  --config outputs/c1-reproduction/config.json \
  --output outputs/c1-reproduction/training \
  --offline
```

This command trains the policy; it does not perform the staged closed-loop checkpoint selection used for the reported score. Full checkpoints preserve model, EMA, optimizer, per-rank RNG, configuration, and the step used to resume the sample schedule. To continue a run, add `--resume outputs/c1-reproduction/training/latest.pt` with the same world size and configuration.

## Closed-loop evaluation

The rollout entry point starts two inference services and eight simulator clients. It requires the configured native simulator, policy checkpoint, tokenizer, EGL setup, and logging integration. Camera conventions, action execution, task horizons, and native success predicates follow the experiment adapters.

The v1.0 logger targets the original OWAC project/workspace and reads its credential from `/root/.config/owac/swanlab-api-key`. Before using the controller or rollout CLI in another environment, configure [`init_run`](../src/owac/runtime/compact_swanlab.py) for your own account and credential source, and use a new run identity. Rollout publishing resumes an existing training run; the training-only `--offline` flag does not disable logging for this entry point.

With those prerequisites configured, the recorded final protocol is:

```bash
python -m owac.runtime.compact_rollouts \
  --config outputs/c1-reproduction/config.json \
  --checkpoint outputs/c1-reproduction/training/step-020000.pt \
  --output outputs/c1-reproduction/independent-final \
  --episodes 10 \
  --initial-offset 20 \
  --lighting Normal Low ExtremeLow Over ExtremeOver \
  --evaluation-kind independent-final
```

The 20k checkpoint is the selection from the published run. A new run should choose its checkpoint using its own development results before final evaluation.

The complete staged training/evaluation controller is:

```bash
python -m owac.runtime.compact_experiment \
  --config outputs/c1-reproduction/config.json \
  --output outputs/c1-reproduction
```

Run the controller for the staged workflow instead of launching the training-only command concurrently. It resumes completed stages and uses the following protocols:

| Stage | Protocol | Role |
|---|---|---|
| 1k updates | Normal, 40 tasks × initial state 5 = 40 episodes | Early screening; excluded from checkpoint ranking |
| 5k / 10k / 20k / 40k / 60k / 80k | Normal, 40 tasks × initial states 5–9 = 200 episodes each | Development selection |
| Selected checkpoint | Five lights, 40 tasks × initial states 20–29 = 2,000 episodes | Final evaluation |

## Reading the outputs

| Output | Contents |
|---|---|
| `training/metrics.jsonl` | Training objectives and offline validation |
| `training/step-*.pt` | Complete training checkpoints |
| `development-*/summary.json` | Successes and denominators by suite/lighting |
| `development-*/episodes.jsonl` | Native outcomes, task IDs, initial states, seeds, and lighting parameters |
| `selection.json` | Checkpoint selected using the common development protocol |
| `independent-final/` | Final summary, per-episode records, and sharded videos/trajectories |
| `*.exit.json` | Native process exit codes and executed commands |

For v1.0, select **step 20,000 EMA**. The final `training/latest.pt` and `training/policy.pt` are from step 80,000 and are not the selected policy. The separately exported `policy-selected-020000-ema.pt` contains inference weights and configuration; use the original complete checkpoint to resume optimization.

SwanLab places the development curve under `success/*`, screening under `success/screening/*`, and final results under `success/independent-final/*`. The x-axis is the checkpoint's training step: the final 2,000-episode score appears at 20k, even though the training schedule reached 80k.

See the [full result report](../research/experiments/raw_compact_c1_results_20261104.md) for all 40 task scores, lighting comparisons, failure examples, and the distinction between offline action error and closed-loop success.
