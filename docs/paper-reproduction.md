# Reproduce the simulation experiments

Run commands from a checkout of `paper-v1.0.1`. The two selected policies and the training encoder are public. CPU inference requires only the locked `paper` environment and one policy file. [Quick start and checkpoint downloads](../README.md#run-a-released-policy)

## Recalculate the paper metrics

The four compressed journals contain all 40,000 formal trial outcomes. The published journals retain the checkpoint SHA-256 of the original PyTorch export, resolved from its evaluation run configuration. The downloadable Safetensors files have different file hashes and identical tensor values. Each switching row retains its schedule hash and observed switch count. No simulator installation is needed to score these journals.

```bash
mkdir -p outputs/paper
manifest_path=experiments/simulation/libero/manifests/libero_manifest_50_init_states_10000_rollouts.json
curl -fL "https://raw.githubusercontent.com/ShuhongLL/RawVLA/ba9c56b04da2c4931b70d8844710f5928d43775b/$manifest_path" \
  -o outputs/paper/libero-manifest.json
for head in regression flow
do
  d=research/experiments/results/paper-v1.0/$head
  .venv/bin/python -m owac.paper.score --manifest outputs/paper/libero-manifest.json \
    --fixed "$d/fixed" --switching "$d/switching"
done
```

The score uses the common step budget for each task and checks episode identities, initial states, lighting schedules and policy identity. The switching confidence interval resamples physical starts within task. The archived fixed scores are 8,200/10,000 and 7,382/10,000 for regression and flow. Corresponding switching scores are 7,809/10,000 and 7,165/10,000. [Archived summaries and per-episode records](../research/experiments/results/paper-v1.0)

## Run native LIBERO

The simulator and policy use separate Python environments. The renderer needs Linux, working EGL/OpenGL drivers and the native LIBERO assets and initial states. The following creates the simulator environment with a CPU PyTorch wheel. The policy may use CPU or CUDA independently. Upstream submodules other than LIBERO are optional for this path.

```bash
git clone --no-checkout https://github.com/ShuhongLL/RawVLA.git external/RawVLA
git -C external/RawVLA checkout ba9c56b04da2c4931b70d8844710f5928d43775b
git -C external/RawVLA submodule update --init third_party/LIBERO
raw="$PWD/external/RawVLA"
uv venv --python 3.11 .venv-sim
uv pip install --python .venv-sim/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
uv pip install --python .venv-sim/bin/python -r configs/paper/simulator-requirements.txt
export LIBERO_CONFIG_PATH="$PWD/outputs/paper/libero-config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$raw/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
.venv-sim/bin/python "$raw/scripts/setup/configure_libero.py" --datasets "$PWD/outputs/paper/libero-datasets"
.venv/bin/python -m owac.paper.evaluate --rawvla "$raw" --sim-python "$PWD/.venv-sim/bin/python" \
  --checkpoint outputs/models/owac-regression-10000.safetensors \
  --mode fixed --limit 1 --device cpu --output outputs/eval/regression-fixed-smoke
```

Use `--mode switching` for the random renderer intervention every ten native actions. Use a fresh output directory and omit `--limit` to evaluate all 10,000 trials per mode. A rerun of the same command resumes its journal. To parallelize, run `--shards N --shard-index I` in separate processes with a different output subdirectory for each index from 0 through N−1. The scorer accepts their parent directory. For the paper-scale runs use the CUDA policy interpreter and `--device cuda`. The simulator uses a local pipe, so workers require neither a network policy server nor an account.

## Train the RAW encoder and both action policies

The simulated encoder trains on LIBERO and RoboTwin trajectories. Action supervision uses the 1,421 LIBERO training trajectories. Fixed train/validation/test counts across both domains are 1,923/240/240. The action recipes include the original training-only normalization statistics, task order, seeds, learning-rate schedules and batch sizes. The complete dataset is large. A reader using the released J5 encoder can download the LIBERO subset alone.

```bash
uv venv --python 3.12 --seed .venv-cuda
gpu="$PWD/.venv-cuda/bin/python"
"$gpu" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
"$gpu" -m pip install -e '.[paper,encoder,raw-ssl,visualization,teleop-mcap]'
"$gpu" -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())'
work="$PWD/outputs/paper"
data="$PWD/outputs/rawvla-bench"
"$gpu" -m owac.paper.download_data --dataset "$data"
"$gpu" -m owac.paper.prepare --dataset "$data" --work "$work"
.venv-cuda/bin/hf download Jeong-USYD/owac-j5-raw-encoder encoder.pt \
  --revision b042fdd6cd5066c8871fbaf19dc84a906acc9000 --local-dir "$work/assets/j5"
.venv-cuda/bin/hf download google/t5-efficient-tiny config.json pytorch_model.bin \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  --revision 3441d7e8bf3f89841f366d39452b95200416e4a9 --local-dir "$work/assets/t5"
```

Use `--domain libero` on the data download for action-only training. Verify the downloaded training assets from the public revisions.

```bash
(cd "$work/assets" && sha256sum -c <<'SUMS'
c081b1ea6023c3ef075caa8b457d31e437058371b353b4b5c59c69a641fc0cae  j5/encoder.pt
b840cd5afdcc806b8175fed5a8800a5aa8be1beb60aab8ab7f650728b122dac2  t5/pytorch_model.bin
SUMS
)
```

The public J5 export keeps only model configuration metadata and all 212 original encoder tensors. Both selected policies jointly train the inherited RAW encoder and the pretrained Google T5 language encoder. [Dataset revision](https://huggingface.co/datasets/ToferFish/RawVLA-Bench/tree/8586c594fff9ce38580e29745f074e82e96e74f7) · [J5 weights](https://huggingface.co/Jeong-USYD/owac-j5-raw-encoder/tree/b042fdd6cd5066c8871fbaf19dc84a906acc9000) · [T5 weights](https://huggingface.co/google/t5-efficient-tiny/tree/3441d7e8bf3f89841f366d39452b95200416e4a9)

To run the visual encoder independently, use the sample from the quick start. J5 returns one 384-dimensional global descriptor and a 16 × 16 grid of 384-dimensional patch descriptors.

```bash
.venv/bin/python -m owac.paper.encode --episode outputs/example.npz \
  --encoder "$work/assets/j5/encoder.pt" --output outputs/encoder-features.npz
```

The following commands target a single node with two BF16-capable NVIDIA GPUs. The encoder global batch is 192. The policy global batch is 256, with microbatch 64 per GPU and gradient accumulation. Each stage writes its own optimizer, EMA and RNG state. The `--no-tracking` and `--offline` switches write metrics locally. To resume an interrupted stage, use the same configuration and add `--resume PATH/TO/latest.pt`. Exact floating-point trajectories depend on hardware and framework versions.

```bash
dist=( "$gpu" -m torch.distributed.run --standalone --nproc-per-node=2 --module )
for stage in j3 j4
do
  "${dist[@]}" owac.training.raw_ssl_train --config "$work/$stage.json" \
    --output "$work/$stage" --no-tracking
done
"${dist[@]}" owac.training.raw_ssl_train --config "$work/j5.json" \
  --output "$work/j5" --steps 12818 --no-tracking
```

J3 trains from random initialization for 100,000 updates. J4 starts from the J3 teacher and trains for 30,000 updates. J5 adds spatial and temporal relations with a frozen J3 reference and stops at update 12,818 of its 30,000-step schedule. The explicit J5 stopping budget skips optional visual diagnostic panels. To use the newly trained encoder for action training, copy `$work/j5/encoder.pt` to `$work/assets/j5/encoder.pt` in a separate experiment directory or set `assets.j5` in the generated action recipes.

```bash
for pair in regression/10000 flow/20000
do
  head=${pair%/*}
  step=${pair#*/}
  "${dist[@]}" owac.training.compact_train --config "$work/$head.json" \
    --output "$work/$head" --stop-step "$step" --offline
  "$gpu" -m owac.paper.checkpoint "$work/$head/policy.pt" "$work/$head.safetensors"
done
```

Both heads retain the original 80,000-update learning-rate schedule when stopped at their selected checkpoints. Run the resulting Safetensors file through the same inference, evaluation and scoring commands. Release verification compared every tensor with each selected training export and all 84 output values on frame zero of the example trajectory. On an RTX 5090, all three encoder stages ran consecutively for one update each with a reduced batch, and both action training entry points completed an update, held-out validation, checkpoint write and EMA export. The checks used PyTorch 2.9.1 on CPU and 2.12.1+cu132 on GPU with Transformers 4.57.0. The full training budgets and 40,000 new rollouts were outside that release check.

## Reproduce the paired ablation

The ablation table uses initial index zero in each of the 40 tasks and five fixed lighting regimes. The selected checkpoints were chosen retrospectively at different action budgets. The full OWAC reference uses the released regression policy at 10,000 updates. The three controls use 4,000 updates for linear patch with the full objective, 4,000 for OWAC response with base SSL, and 3,000 for linear patch with base SSL. [Paired outcomes](../research/experiments/results/paper-v1.0/ablation-episodes.jsonl.gz) · [Control configurations](../configs/paper/ablations)

With the default training paths from above, set `arm` to `patch-full`, `owac-standard` or `patch-standard`. Each arm has an independently initialized visual encoder. The zero loss weights in the base-SSL arm preserve its matched schedule.

```bash
arm=patch-full
dst="$work/ablations/$arm"
for stage in base exposure relations
do
  extra=()
  if [ "$stage" = relations ]; then extra=(--steps 12818); fi
  "${dist[@]}" owac.training.raw_ssl_train --config "configs/paper/ablations/$arm-$stage.json" \
    --output "$dst/$stage" --no-tracking "${extra[@]}"
done
step=4000
if [ "$arm" = patch-standard ]; then step=3000; fi
"${dist[@]}" owac.training.compact_train --config "configs/paper/ablations/$arm-action.json" \
  --output "$dst/action" --stop-step "$step" --offline
"$gpu" -m owac.paper.checkpoint "$dst/action/policy.pt" "$dst/action.safetensors"
"$gpu" -m owac.paper.evaluate --checkpoint "$dst/action.safetensors" --rawvla "$raw" \
  --sim-python "$PWD/.venv-sim/bin/python" --mode fixed --initial-index 0 --device cuda \
  --output "$dst/paired-200"
```
