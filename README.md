# OWAC

**Optimizing RAW Representations for Robot Manipulation Across Lighting Conditions**

OWAC learns a visual representation directly from RAW sensor measurements. The encoder is pretrained with exposure changes, spatial relations and temporal correspondences. At policy training time, it is optimized jointly with a language encoder and an action head. The two released LIBERO policies share this design. OWAC-Regression predicts a 12-step action chunk directly. OWAC-Flow samples the chunk in ten integration steps.

## Run a released policy

Start with Linux, Git, curl and [uv](https://docs.astral.sh/uv/). The example runs on CPU and downloads both selected EMA checkpoints. The tokenizer and normalization statistics are included. Inference runs offline after setup.

```bash
git clone --branch paper-v1.0 https://github.com/PAIR-Z-ZENO-LAB/owac.git
cd owac
uv sync --python 3.12 --locked --extra paper
mkdir -p outputs/models
base=https://github.com/PAIR-Z-ZENO-LAB/owac/releases/download/paper-v1.0
for file in owac-regression-10000.safetensors owac-flow-20000.safetensors SHA256SUMS
do
  curl -fL --retry 3 "$base/$file" -o "outputs/models/$file"
done
(cd outputs/models && sha256sum -c SHA256SUMS)
.venv/bin/python -m owac.paper.infer \
  --checkpoint outputs/models/owac-regression-10000.safetensors
```

The example uses a deterministic synthetic input and prints 12 rows of seven LIBERO delta commands. To try a recorded demonstration, download one public trajectory and pass it to either policy.

```bash
dataset=https://huggingface.co/datasets/ToferFish/RawVLA-Bench/resolve/8586c594fff9ce38580e29745f074e82e96e74f7
sample=data/libero/raw/libero_10_no_noops_1.0.0_lerobot/episode_000001/variant_00.npz
curl -fL --retry 3 "$dataset/$sample" -o outputs/example.npz
.venv/bin/python -m owac.paper.infer --episode outputs/example.npz --frame 0 \
  --checkpoint outputs/models/owac-flow-20000.safetensors --output outputs/actions.npy
```

## Paper results

Each policy was evaluated on 40 LIBERO tasks, with 50 initial states per task. Each state has five fixed lighting trials and five switching trials. All-5 is the fraction of physical initial states that succeed under every fixed lighting regime. Switching drop is measured from the same states under fixed Normal lighting.

| Selected policy | Fixed mean | Fixed All-5 | Fixed Normal | Switching | Switching drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| OWAC-Regression, 10,000 updates | 82.00% | 53.10% | 85.80% | 78.09% | 7.71 pp |
| OWAC-Flow, 20,000 updates | 73.82% | 37.50% | 74.90% | 71.65% | 3.25 pp |

The release includes the 40,000 scored episode records behind these rows and the separate 200-rollout-per-policy ablation panel. The main policies consume two upright, three-channel pseudo-RAW views and nine proprioceptive values. The physical platform uses two packed 10-bit Bayer streams, seven follower joint positions and a gripper opening in metres. It has its own training and deployment pipeline.

| Reader task | Entry point |
| --- | --- |
| Download policies and verify hashes | [GitHub Release](https://github.com/PAIR-Z-ZENO-LAB/owac/releases/tag/paper-v1.0) |
| Recalculate the paper table, train the encoder and policies, run closed-loop LIBERO, reproduce the ablations | [Simulation reproduction guide](docs/paper-reproduction.md) |
| Train on locally collected MCAP data and deploy fruit, bowl or pouring policies | [Physical robot guide](docs/real-robot-reproduction.md) |
| Inspect run configurations, scoring data and attribution | [Recipes](configs/paper), [episode records](research/experiments/results/paper-v1.0), [licensing and credits](licenses/README.md) |

## Citation and use

Cite the software using the repository's citation metadata and specify the release tag and checkpoint in any reproduction. The manuscript title is *OWAC: Optimizing RAW Representations for Robot Manipulation Across Lighting Conditions*. [Citation metadata](CITATION.cff)

OWAC-authored source is available for noncommercial use under the PolyForm Noncommercial License 1.0.0. The OWAC contributions to the two released policy checkpoints are available under CC BY-NC 4.0. Third-party components retain their respective licenses. [Code license](LICENSE) · [Model terms and third-party notices](licenses/README.md)
