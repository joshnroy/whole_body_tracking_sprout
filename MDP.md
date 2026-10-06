# Tracking MDP (mjlab)

This page describes the observations, domain randomization, rewards and terminations of the mjlab tracking tasks in
[`source/whole_body_tracking_mjlab`](source/whole_body_tracking_mjlab). The shared config is
[`tracking_env_cfg.py`](source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/tasks/tracking/tracking_env_cfg.py).
The robot configs ([`g1/flat_env_cfg.py`](source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/tasks/tracking/config/g1/flat_env_cfg.py),
[`sprout/flat_env_cfg.py`](source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/tasks/tracking/config/sprout/flat_env_cfg.py))
fill in body names, the action scale and the contact sensor. The term implementations come from mjlab
(`mjlab.tasks.tracking.mdp`), except `undesired_contacts`, which is in
[`mdp/rewards.py`](source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/tasks/tracking/mdp/rewards.py).

## Timing and action

| Setting | Value |
|---|---|
| Physics timestep | 0.005 s (200 Hz) |
| Decimation | 4, so the policy runs at 50 Hz (`Low-Freq` tasks: 8, 25 Hz) |
| Episode length | 10 s |
| Parallel envs | 4096 |
| Action | Joint position targets for every actuated joint: `target = default_joint_pos + scale * action` |
| Action scale | G1: mjlab's `G1_ACTION_SCALE` (BeyondMimic's `0.25 * effort_limit / stiffness` per joint). Sprout: a fixed 0.25 rad |

## Reference motion and anchor

The motion command (`MotionCommand`) plays a reference motion npz. The robot's pose is compared to the reference
through an **anchor body**, `torso_link` on both robots, and a list of tracked bodies (14 on G1, 15 on Sprout).

- **Relative body poses.** Before the reference bodies are compared to the robot's, they are moved so that the
  reference anchor lies on the robot's anchor in x/y and heading (yaw). The reference anchor height and roll/pitch are
  kept. The body rewards and the end-effector termination use these relative poses, so they do not penalize drift in
  global position or heading.
- **Reset state.** On reset the robot is put at a reference frame, with the root pose perturbed by up to ±5 cm in x/y,
  ±1 cm in z, ±0.1 rad in roll/pitch and ±0.2 rad in yaw. The root velocity is perturbed by up to ±0.5 m/s in x/y,
  ±0.2 m/s in z, ±0.52 rad/s in roll/pitch and ±0.78 rad/s in yaw. Each joint position gets up to ±0.1 rad of noise.
- **Adaptive sampling.** The start frame is not uniform. The motion is split into 1 s bins, and each bin is sampled in
  proportion to how often episodes failed in it, plus a uniform floor (`adaptive_uniform_ratio = 0.1`). The failure
  counts are an exponential moving average (`adaptive_alpha = 0.001`). Play mode always starts from frame 0.

## Observations

The actor and critic each take one concatenated vector.

| Term | Actor | Critic | Actor noise (uniform ±) | Description |
|---|:-:|:-:|---|---|
| `command` | ✓ | ✓ | – | Reference joint positions and velocities at the current frame |
| `motion_anchor_pos_b` | ✓ * | ✓ | 0.25 m | Reference anchor position in the robot anchor frame |
| `motion_anchor_ori_b` | ✓ | ✓ | 0.05 | Reference anchor orientation in the robot anchor frame (first two columns of the rotation matrix, 6 values) |
| `body_pos` | | ✓ | – | Robot tracked-body positions in the robot anchor frame |
| `body_ori` | | ✓ | – | Robot tracked-body orientations in the robot anchor frame (6 values each) |
| `base_lin_vel` | ✓ * | ✓ | 0.5 m/s | IMU linear velocity (`robot/imu_lin_vel`) |
| `base_ang_vel` | ✓ | ✓ | 0.2 rad/s | IMU angular velocity (`robot/imu_ang_vel`) |
| `joint_pos` | ✓ | ✓ | 0.01 rad | Joint positions minus the default pose. The actor reads them with the encoder bias added |
| `joint_vel` | ✓ | ✓ | 0.5 rad/s | Joint velocities |
| `actions` | ✓ | ✓ | – | The previous action (raw policy output) |

\* Removed in the `Wo-State-Estimation` tasks, because on hardware they need a base position and linear velocity
estimator.

The critic gets no noise. Both groups use running observation normalization in the PPO networks.

**Sprout observation delay.** On Sprout, each actor term that comes from a sensor or the state estimator
(`motion_anchor_pos_b`, `motion_anchor_ori_b`, `base_lin_vel`, `base_ang_vel`, `joint_pos`, `joint_vel`) is randomly
0 or 1 policy steps (0–20 ms) old. The delay is resampled every step. The command and last action are never delayed.

## Domain randomization

| Term | When | What |
|---|---|---|
| `physics_material` | Startup | Foot friction set to a value in [0.3, 1.2], shared by all foot geoms of an env. G1: `(left\|right)_foot[1-7]_collision`. Sprout: `(left\|right)_foot_link_collision_[01]`. The other collision geoms are frictionless (`condim=1`) |
| `add_joint_default_pos` | Startup | Encoder bias of ±0.01 rad per joint (calibration error). The actor's `joint_pos` observation includes it |
| `base_com` | Startup | `torso_link` centre of mass shifted by ±2.5 cm in x and ±5 cm in y and z |
| `push_robot` | Every 1–3 s | Random offset added to the root velocity: ±0.5 m/s in x/y, ±0.2 m/s in z, ±0.52 rad/s in roll/pitch, ±0.78 rad/s in yaw. Sprout uses half these ranges |
| Observation noise | Every step | Uniform noise on the actor terms (see the observation table) |
| Actuator delay (Sprout) | Each reset, then every 4 s | Joint commands reach the motors 2–5 physics steps (10–25 ms) late |
| Observation delay (Sprout) | Every step | 0–1 policy steps (see above) |

Sprout's joints use the vendor's DC motor models (kp 32.5, kd 1, with per-motor effort limits, torque-speed curves and
armature), defined in
[`robots/sprout_motors.py`](source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/robots/sprout_motors.py).

The `No-DR` tasks drop friction, encoder-bias and CoM randomization, pushes and observation noise. Sprout's actuator and
observation delays and the reset-state perturbation stay on. Play mode turns off noise and pushes and starts from the
unperturbed reference.

## Rewards

The tracking terms are Gaussian kernels `exp(-error / std²)`. "Mean over bodies" means the squared error is averaged
over the tracked bodies before the kernel.

| Term | Weight | std | Error |
|---|---:|---:|---|
| `motion_global_anchor_pos` | 0.5 | 0.3 | Squared distance between the reference and robot anchor positions (world frame) |
| `motion_global_anchor_ori` | 0.5 | 0.4 | Squared rotation angle between the reference and robot anchor orientations |
| `motion_body_pos` | 1.0 | 0.3 | Squared position error of the relative reference bodies, mean over bodies |
| `motion_body_ori` | 1.0 | 0.4 | Squared rotation angle of the relative reference bodies, mean over bodies |
| `motion_body_lin_vel` | 1.0 | 1.0 | Squared linear velocity error (world frame), mean over bodies |
| `motion_body_ang_vel` | 1.0 | 3.14 | Squared angular velocity error (world frame), mean over bodies |
| `action_rate_l2` | −0.1 | – | `‖a_t − a_{t−1}‖²` (`Low-Freq`: weight −0.05) |
| `joint_limit` | −10.0 | – | Sum over joints of how far each joint is past its soft limit (90% of the joint range) |
| `undesired_contacts` | −0.1 | – | Number of bodies with a net contact force above 1 N at any physics substep of the policy step. Every body counts except G1's ankles and wrists, and Sprout's feet and gripper links |

## Terminations

| Term | Condition |
|---|---|
| `time_out` | Episode reaches 10 s. Counted as a timeout, not a failure, so PPO bootstraps the value |
| `anchor_pos` | Anchor height differs from the reference by more than 0.25 m |
| `anchor_ori` | The z component of gravity in the anchor frame differs from the reference's by more than 0.8. This measures tilt only, not heading |
| `ee_body_pos` | Any end effector's height differs from the relative reference by more than 0.25 m. G1: ankles and wrists. Sprout: feet and lower gripper links |

The failure terminations also drive adaptive sampling, so start frames near hard parts of the motion are picked more
often.

## Task variants

| Task suffix | Change |
|---|---|
| `Tracking-Flat-{G1,Sprout}-v0` | Full MDP above |
| `-Wo-State-Estimation-v0` | Actor loses `motion_anchor_pos_b` and `base_lin_vel` |
| `-Low-Freq-v0` | Policy at 25 Hz (decimation 8), `action_rate_l2` weight halved, contact history covers 8 substeps. PPO uses 12 steps per env and `γ = 0.99²`, `λ = 0.95²` |
| `-Wo-State-Estimation-No-DR-v0` | No state estimation, and no friction, encoder-bias or CoM randomization, pushes or observation noise |

## PPO

All tasks share BeyondMimic's settings: actor and critic MLPs of (512, 256, 128) with ELU, initial action std 1.0,
24 steps per env, 5 epochs, 4 mini-batches, learning rate 1e-3 with an adaptive KL schedule (target 0.01), clip 0.2,
entropy coefficient 0.005, `γ = 0.99`, `λ = 0.95`, 30 000 iterations and a checkpoint every 500. See
[`config/g1/rl_cfg.py`](source/whole_body_tracking_mjlab/whole_body_tracking_mjlab/tasks/tracking/config/g1/rl_cfg.py).
