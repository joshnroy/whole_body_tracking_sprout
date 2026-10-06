# BeyondMimic Motion Tracking Code

[![mjlab](https://img.shields.io/badge/mjlab-%E2%89%A51.6.0-silver)](https://github.com/mujocolab/mjlab)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Overview

BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.

This repository has two implementations of the same tasks:

- **mjlab** ([`source/whole_body_tracking_mjlab`](source/whole_body_tracking_mjlab)): runs on
  [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp). It needs neither Isaac Sim nor the robot description
  download, supports the Unitree G1 and the Sprout humanoid, and adds CPU MuJoCo sim-to-sim evaluation and
  velocity-commanded motion matching. The sections below up to [Isaac Lab Version](#isaac-lab-version) cover it.
- **Isaac Lab** ([`source/whole_body_tracking`](source/whole_body_tracking)): the original implementation. See
  [Isaac Lab Version](#isaac-lab-version).

The observations, domain randomization, rewards and terminations are described in [MDP.md](MDP.md).

For sim-to-real deployment, please refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation (uv)

You need Linux and [uv](https://docs.astral.sh/uv/). Training needs an NVIDIA GPU with CUDA. Sim2sim and motion
matching run on the CPU.

1. Install uv, if you don't have it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:

   ```bash
   git clone git@github.com:Kartikgar/whole_body_tracking_sprout.git
   cd whole_body_tracking_sprout
   ```

3. Create the environment. `uv sync` reads [`.python-version`](.python-version) (Python 3.12) and
   [`uv.lock`](uv.lock), downloads that Python if needed, creates `.venv`, and installs mjlab and the
   `whole_body_tracking_mjlab` package in editable mode:

   ```bash
   uv sync
   ```

   The commands in this README run through `uv run`, which uses `.venv` without activating it and re-syncs it if
   `uv.lock` changed. You can also `source .venv/bin/activate` and drop the `uv run` prefix.

4. Check the install. This should list the `Tracking-Flat-*` tasks:

   ```bash
   uv run wbt-list-envs
   ```

5. **Optional: W&B.** W&B is only needed to share motions through its registry, to train or play with
   `--registry-name` / `--wandb-run-path`, or to see runs on its dashboard. Without it, use local files and
   TensorBoard (see [Training](#training)). To use it:

   ```bash
   uv run wandb login
   export WANDB_ENTITY={your-organization}   # your organization, not your personal username
   ```

6. **Sprout only.** The Sprout model is not distributed with this repository. Unzip it to `~/sprout-model-v0.1.4`
   (the default) or to `source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/assets/sprout` (gitignored), or set
   `SPROUT_MODEL_DIR` to the folder that contains `sprout.xml`. G1 uses mjlab's built-in model and needs nothing extra.

### Tasks

| Task | Description |
|---|---|
| `Tracking-Flat-{G1,Sprout}-v0` | Full BeyondMimic MDP |
| `Tracking-Flat-{G1,Sprout}-Wo-State-Estimation-v0` | No anchor position or base linear velocity in the policy observations |
| `Tracking-Flat-{G1,Sprout}-Low-Freq-v0` | Policy at 25 Hz instead of 50 Hz |
| `Tracking-Flat-{G1,Sprout}-Wo-State-Estimation-No-DR-v0` | No state estimation and no domain randomization |

The G1 tasks have the same rewards, terminations, observations, randomization and PPO settings as the Isaac Lab
configs. See [MDP.md](MDP.md) for details.

## Motion Preprocessing

Reference motions must be retargeted to the robot and use generalized coordinates only. `wbt-csv-to-npz` adds the body
poses, velocities and accelerations by forward kinematics:

```bash
# G1: Unitree-format csv (e.g. the retargeted LAFAN1 dataset)
uv run wbt-csv-to-npz --robot g1 --input-file {motion_name}.csv --input-fps 30 --output-file {motion_name}.npz

# Sprout: GMR pickles are read directly
uv run wbt-csv-to-npz --robot sprout --input-file {motion_name}.pkl --output-file {motion_name}.npz
```

Add `--registry-upload {motion_name}` to also upload the npz to the W&B "Motions" registry. To set that registry up, open
Registry under Core in W&B and create a collection named "Motions" with artifact type "All Types". You can also skip the
registry and train from the local npz.

Motion sources (please follow the original licenses):

- Unitree-retargeted LAFAN1 Dataset is available
  on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
- Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
- Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
- Balance motions are from [HuB](https://hub-robot.github.io/)
- Sprout motions can be retargeted with [GMR](https://github.com/YanjieZe/GMR)

## Training

```bash
# From a local npz, logging to TensorBoard (no W&B account needed)
uv run wbt-train Tracking-Flat-Sprout-Wo-State-Estimation-v0 --env.commands.motion.motion-file {motion_name}.npz \
  --agent.logger tensorboard --agent.run-name {run_name}

# From a local npz, logging to W&B
uv run wbt-train Tracking-Flat-Sprout-Wo-State-Estimation-v0 --env.commands.motion.motion-file {motion_name}.npz \
  --agent.wandb-project {project_name} --agent.run-name {run_name}

# From the W&B registry
uv run wbt-train Tracking-Flat-G1-v0 --registry-name {your-organization}-org/wandb-registry-motions/{motion_name} \
  --agent.wandb-project {project_name} --agent.run-name {run_name}
```

Checkpoints go to `logs/rsl_rl/{g1_flat,sprout_flat}/{run}/`, every 500 iterations. Next to each checkpoint, training
exports `{run}.onnx`, which has the reference motion and the observation layout in its metadata. Sim2sim and motion
matching use this file.

`wbt-train`, `wbt-play` and `wbt-list-envs` wrap mjlab's `train`, `play` and `list-envs`, so they take the same flags
(see `--help`). Any env or agent setting can be overridden on the command line, for example `--env.scene.num-envs 2048`
or `--agent.max-iterations 10000`. Unlike mjlab's `train`, `wbt-train` records training videos by default
(`--video-length` steps every `--video-interval` env steps, saved under the run's `videos/train` and uploaded to W&B when
logging to it). Pass `--video False` to turn it off. With TensorBoard, view the curves with
`uv run tensorboard --logdir logs/rsl_rl`.

To watch a trained policy in mjlab:

```bash
# From a local checkpoint
uv run wbt-play Tracking-Flat-G1-v0 --checkpoint-file logs/rsl_rl/g1_flat/{run}/model_{iteration}.pt \
  --motion-file {motion_name}.npz

# From a W&B run (downloads the checkpoint and the motion)
uv run wbt-play Tracking-Flat-G1-v0 --wandb-run-path {wandb-run-path}
```

The W&B run path is shown in the run overview. It has the form `{your-organization}/{project_name}/` followed by an
8-character run ID. It is not the run name.

## Sim2sim (CPU MuJoCo)

`wbt-sim2sim` runs an exported `{run}.onnx` in plain CPU MuJoCo with onnxruntime instead of MuJoCo-Warp. It builds the
observations and joint targets from the ONNX metadata, the way a deployment controller would. The robot model, actuator
laws (including Sprout's DC motors), default pose and action scale come from the task.

```bash
uv run wbt-sim2sim Tracking-Flat-Sprout-Wo-State-Estimation-v0 \
  --onnx-file logs/rsl_rl/sprout_flat/{run}/{run}.onnx --output-file metrics.json
```

The robot starts at the reference with no randomization and runs once through the motion, or until a training
termination fires. The script reports whether it succeeded, plus the same tracking metrics as mjlab's `evaluate`
(`mpkpe`, `r_mpkpe`, joint and end-effector errors).

| Flag | Effect |
|---|---|
| `--viewer True` | Show the rollout in the MuJoCo viewer in real time, with the reference bodies as green spheres |
| `--video-file out.mp4` | Render the rollout offscreen (set `MUJOCO_GL=egl` on a headless machine) |
| `--output-file metrics.json` | Write the metrics to JSON |
| `--trajectory-file traj.npz` | Write per-step robot state, reference and actions |
| `--start-frame`, `--num-frames` | Run part of the motion |
| `--dr True` | Apply all of training's randomization: startup friction, CoM and encoder bias, the reset perturbation, observation noise, sampled delays and pushes |
| `--push-scale 0.5` | Add training's pushes, with the velocity range scaled |
| `--actuator-delay`, `--obs-delay` | Fixed delays (physics steps / policy steps) for terms trained with a random delay. Default: the middle of the trained range, rounded down (Sprout: 3 physics steps, no observation delay) |
| `--stop-on-termination False` | Keep going after a termination, to watch the whole motion |
| `--seed` | Seed for pushes and `--dr` |

## Motion Matching (velocity commands)

`wbt-motion-match` steers a tracking policy with forward/backward speed and yaw rate commands. It does motion matching
over the npz the policy was trained on, for example a single walking clip. Every 0.2 s, or when the command changes, it
jumps to the clip frame that best fits two things: the command, through the upcoming root trajectory at 0.4, 0.8 and
1.2 s, and the current pose, through the feet and root velocity. The jump is blended away with inertialization.

The generated reference drives the exported policy in CPU MuJoCo, as in `wbt-sim2sim` (no randomization). Without
`--onnx-file` the reference plays kinematically.

```bash
# Keyboard control in the viewer: up/down sets forward speed, left/right sets yaw rate, space stops.
uv run wbt-motion-match Tracking-Flat-Sprout-Wo-State-Estimation-v0 --motion-file walk1_subject1.npz \
  --onnx-file logs/rsl_rl/sprout_flat/{run}/{run}.onnx --viewer True

# Scripted: space-separated "time,forward_speed,yaw_rate" entries (s, m/s, rad/s), each held until the next.
# Prints commanded vs reference vs robot velocity for each command.
uv run wbt-motion-match Tracking-Flat-Sprout-Wo-State-Estimation-v0 --motion-file walk1_subject1.npz \
  --onnx-file logs/rsl_rl/sprout_flat/{run}/{run}.onnx --commands "0,0,0 3,0.5,0 8,0.4,0.8 13,-0.3,0 18,0,0" \
  --video-file out.mp4 --output-file matched.npz
```

| Flag | Effect |
|---|---|
| `--commands` | Command schedule. Without it, `--viewer True` takes commands from the keyboard |
| `--duration` | Rollout length in seconds. Default: 3 s past the last scheduled command |
| `--output-file matched.npz` | Write the generated reference in the training npz format, plus the per-frame command. You can train on it |
| `--metrics-file` | Write the per-command velocity metrics to JSON |
| `--matcher.max-forward-speed 1.0`, `--matcher.max-backward-speed 0.3`, `--matcher.max-yaw-rate 1.5` | Commands are clipped to these, which should match what the clip covers |
| `--foot-body-names` | Feet for the matching features. Default: bodies matching `(left\|right)_(foot\|ankle_roll)_link` |
| `--actuator-delay`, `--obs-delay`, `--viewer`, `--video-file`, `--start-frame` | As in `wbt-sim2sim` |

## Sprout

The Sprout tasks run the same MDP on the Sprout humanoid (27 joints). Sprout's joints are driven by the vendor's DC motor
models ([`robots/sprout_motors.py`](source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/robots/sprout_motors.py)):
kp 32.5 / kd 1 on every joint, with per-motor effort limits, torque-speed curves and armature. The armature values are
vendor guesses.

Joint commands reach the motors 2–5 physics steps (10–25 ms) late, with the delay resampled per environment on each reset
and every 4 s. The policy's sensor observations (anchor pose, IMU, joint position and velocity) are randomly 0 or 1 policy
steps (0–20 ms) old. The action scale is a fixed 0.25 rad instead of BeyondMimic's `0.25 * effort / stiffness`, which
would leave the 3.5 Nm neck-pitch, wrist-pitch and gripper motors only 0.027 rad. Pushes use half of G1's velocity range.

## Differences from the Isaac Lab Version

- Base velocities come from the robot's IMU sensors instead of the pelvis root state.
- Joint default-position randomization becomes mjlab's `encoder_bias` (a ±0.01 rad calibration error).
- Friction is randomized on the feet only, in the range (0.3, 1.2). The other MuJoCo collision geoms use frictionless
  contacts.
- `undesired_contacts` uses a MuJoCo contact sensor on every body except the end effectors (G1 ankles and wrists;
  Sprout feet and grippers).

There is also a reproduction of BeyondMimic inside mjlab itself; see
[its tracking config](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py). This
repository reuses mjlab's port of the motion command and MDP terms and its Unitree G1 MJCF model.

## Isaac Lab Version

The original implementation in [`source/whole_body_tracking`](source/whole_body_tracking) is not part of the uv
environment. It is installed into an Isaac Lab environment.

### Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory).

- Pull the robot description files from GCS

```bash
# Enter the repository
cd whole_body_tracking
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

### Motion Preprocessing & Registry Setup

Set up the W&B "Motions" registry as described in [Motion Preprocessing](#motion-preprocessing), then convert
retargeted motions:

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

This will automatically upload the processed motion file to the WandB registry with output name {motion_name}.

- Test if the WandB registry works properly by replaying the motion in Isaac Sim:

```bash
python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
```

- Debugging
    - Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    - If /tmp folder is not accessible, modify csv_to_npz.py L319 & L326 to a temporary folder of your choice.

### Policy Training

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

## Code Structure

### mjlab

- **`source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/tasks/tracking/tracking_env_cfg.py`**
  The robot-agnostic MDP config, and the play / no-state-estimation / no-DR / low-frequency variants.

- **`source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/tasks/tracking/config/{g1,sprout}`**
  Per-robot env configs (`flat_env_cfg.py`), PPO configs (`rl_cfg.py`) and task registration.

- **`source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/tasks/tracking/mdp`**
  Re-exports mjlab's tracking MDP terms and adds `undesired_contacts`.

- **`source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/robots`**
  The Sprout model loader and DC motor models.

- **`source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/motion_matching.py`**
  The motion matcher used by `wbt-motion-match`.

- **`source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/scripts`**
  The `wbt-*` command-line tools.

### Isaac Lab

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.
