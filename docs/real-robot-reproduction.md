# Native Bayer training and NERO deployment

The real-robot pipeline uses two synchronized OWAC board cameras and the right NERO follower. Camera order is U2 then U3. Each frame contains 640 × 480 active BGGR10 samples with a 1,024-byte packed row stride. The policy consumes the original integer samples and eight proprioceptive values. Targets and commands use seven joint angles in radians and one gripper opening in metres. The physical demonstrations, trained hardware checkpoints, camera annotations and fitted field profiles are private. The instructions below use recordings, weights and calibration from the reader's installation.

## Train and replay from local MCAP

Use the CUDA environment from the simulation guide, including the `teleop-mcap` and `real-robot` extras. Install the latter with `.venv-cuda/bin/python -m pip install -e ".[real-robot]"`. Store each recording under `DATA/subset/episode_NNN/episode/*.mcap`. Record both RAW image topics, the follower state and the executed follower command. The converter pairs camera frames and uses feedback available at the observation time. Keep entire trajectories together when splitting.

| Stream | ROS topic |
| --- | --- |
| U2, U3 native RAW10 | `/teleop_camera/u2/raw10`, `/teleop_camera/u3/raw10` |
| Right follower state | `/slave_right/joint_states_single` |
| Executed joint command | `/slave_right/joint_cmd_mux/output` |

```bash
gpu="$PWD/.venv-cuda/bin/python"
real="$PWD/outputs/real/local"
paper="$PWD/outputs/paper"
"$gpu" -m owac.raw.teleop_mcap --data "$PWD/recordings" --output "$real/aligned" \
  --validation episode_005 episode_010
"$gpu" -m owac.paper.teleop vision --inventory "$real/aligned/inventory.json" --work "$real" \
  --encoder "$paper/assets/j5/encoder.pt" --anchor "$paper/j3/encoder.pt"
"$gpu" -m torch.distributed.run --standalone --nproc-per-node=1 --module \
  owac.training.raw_ssl_train --config "$real/vision.json" --output "$real/vision" --no-tracking
"$gpu" -m owac.paper.teleop action --inventory "$real/aligned/inventory.json" --work "$real" \
  --encoder "$real/vision/encoder.pt" --language "$paper/assets/t5" \
  --instruction "put the banana in the basket"
"$gpu" -m owac.training.teleop_policy_data --config "$real/action.json"
"$gpu" -m torch.distributed.run --standalone --nproc-per-node=1 --module \
  owac.training.teleop_policy_train --config "$real/action.json" --output "$real/action" --no-tracking
"$gpu" -m owac.training.teleop_policy_eval --checkpoint "$real/action/policy-best.pt" \
  --inventory "$real/aligned/inventory.json" --episode fruit01/episode_005 --frame 0
```

Replace the episode IDs and instruction with the task actually recorded. A fully qualified held-out ID such as `pouring01/episode_009` selects that trajectory only. J3 is the frozen reference from the first visual pretraining stage. The simulation guide gives the J3 training command and public J5 and T5 download commands. The example produces a locally trained regression policy, and replay prints its 12 × 8 command chunk. [Visual pretraining](paper-reproduction.md#train-the-raw-encoder-and-both-action-policies)

## Set up the hardware interfaces

The field host used Ubuntu 24.04, ROS 2 Jazzy, system Python 3.12 and the NERO arm packages. The repository includes the deployed ROS message and the OWAC-board capture bridge. From the OWAC checkout, install them into a clean checkout of the public arm stack.

```bash
export OWAC_COLLECTION="$PWD/external/agilx-teleop-collection"
git clone https://github.com/PAIR-Z-ZENO-LAB/agilx-teleop-collection.git "$OWAC_COLLECTION"
git -C "$OWAC_COLLECTION" checkout 479b88d00a08b5dcf339d0b745540178a6086a17
msg="$OWAC_COLLECTION/agx_arm_ros/src/agx_arm_msgs"
cp hardware/nero/OwacRawFrame.msg "$msg/msg/"
cp hardware/nero/owac_raw.py hardware/nero/owac_raw_camera_ros2.py "$OWAC_COLLECTION/scripts/"
if ! grep -q 'msg/OwacRawFrame.msg' "$msg/CMakeLists.txt"; then
  sed -i '/"msg\/GripperStatus.msg"/a\  "msg/OwacRawFrame.msg"' "$msg/CMakeLists.txt"
fi
(cd "$OWAC_COLLECTION" && ./scripts/build_ros2.sh)
source "$OWAC_COLLECTION/scripts/env/pure_teleop_env.sh"
ros2 interface show agx_arm_msgs/msg/OwacRawFrame
```

Set up `config/local.env` and the right follower CAN interface as documented by the arm stack. Start the bridge with the address of your OWSF-compatible board, then check that both camera topics carry data. The console can also start the bridge from `$OWAC_COLLECTION/config/recording-right-owac.yaml` with the fields `sources.owac.host`, `port` and `fps`. [Arm setup and ROS dependencies](https://github.com/PAIR-Z-ZENO-LAB/agilx-teleop-collection/tree/479b88d00a08b5dcf339d0b745540178a6086a17)

```bash
/usr/bin/python3 "$OWAC_COLLECTION/scripts/owac_raw_camera_ros2.py" --host "$OWAC_HOST" --port 5560 --fps 30
```

## Deploy a locally trained model

On the robot host, use a CUDA-enabled Python 3.12 environment compatible with the installed Jazzy Python bindings. Install this checkout and its paper, encoder, raw-ssl and both real-robot extras in it. Set `OWAC_PYTHON` to that interpreter. A one-policy catalog allows a new installation to use its own step number and model file.

```bash
export OWAC_PYTHON="$PWD/.venv-cuda/bin/python"
export OWAC_FRUIT_HOME="$PWD/outputs/real/deployment"
mkdir -p "$OWAC_FRUIT_HOME/candidates/local"
cp "$real/action/policy-best.pt" "$OWAC_FRUIT_HOME/candidates/local/policy.pt"
"$OWAC_PYTHON" - <<'PY'
import json, os, torch
from pathlib import Path
dest = Path(os.environ["OWAC_FRUIT_HOME"]) / "candidates"
saved = torch.load(dest / "local/policy.pt", map_location="cpu", weights_only=False, mmap=True)
row = dict(id="local", label="Local regression policy", step=saved["step"], rtc=False)
(dest / "catalog.json").write_text(json.dumps(dict(include_originals=False, candidates=[row])))
PY
scripts/fruit-infer --list
scripts/fruit-infer --check --policy local --frames 12
```

The second command previews predictions using live observations. For a supervised run, define a reachable reset position for your robot in `$OWAC_FRUIT_HOME/reset-pose.json` using `joints_deg` (seven values) and `gripper_mm`. Verify with `scripts/fruit-infer --check-reset --policy local`. Then launch `scripts/fruit-infer --policy local` in the robot terminal. The operator selects the controller and enables execution from the console.

## Field configurations

| Task | Selected field policy | Code and fitting inputs |
| --- | --- | --- |
| Fruit placement | 800-step final placement model with the task readouts | [Field catalog](../configs/runtime/fruit_final_catalog.json), [placement training](../src/owac/training/fruit_placement.py), [fruit fitting and packaging](../scripts/analysis) |
| Bowl stacking | 9,500-step temporal model, four-command execution and a fitted two-grasp readout | [Field catalog](../configs/runtime/bowl_catalog.json), [temporal training](../scripts/experiments/train_bowl_temporal.py), [two-grasp fitting](../scripts/experiments/fit_bowl_flow.py) |
| Pouring | 6,000-step field model with feedback-based departure. The alternate proactive policy uses 12,200 steps | [Field catalog](../configs/runtime/pouring_field_catalog.json), [proactive catalog](../configs/runtime/pouring_proactive_catalog.json), [feedback training](../scripts/experiments/train_pouring_feedback.py), [departure fitting](../scripts/experiments/fit_pouring_departure.py) |

The catalogs specify the expected relative filenames, steps, execution horizons and fitted profiles. Training and fitting scripts expose their inputs via `--help`. For the bowl pipeline, provide marked training frames for the wrist bracket, run the release and smoothing fits, and use the resulting `bowl-postprocess.json` in the fit directory. The trace-based validation takes locally recorded feedback and commands. For pouring, the field preparation script takes the locally fitted departure profile via `--departure`. These task-specific paths require training data and field calibration from the corresponding setup. The published simulation weights use a seven-dimensional LIBERO action space.
