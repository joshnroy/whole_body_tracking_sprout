"""Sim-to-sim evaluation of an exported tracking policy in CPU MuJoCo.

Training runs on MuJoCo-Warp through mjlab's managers. This script runs the exported ``policy.onnx`` (with its bundled
reference motion) in plain CPU MuJoCo with onnxruntime, building observations and actuator commands from the ONNX
metadata the way a deployment controller would. Only the MuJoCo model (robot MJCF, actuators, terrain, solver options)
and the actuator laws come from the task config, because the ONNX metadata does not describe torque-level actuators
such as Sprout's DC motors (its ``joint_stiffness`` is the motor gain of 1). The default joint positions and action
scale also come from the task config: the metadata rounds them to 3 decimals. They are checked against the metadata.

The rollout matches ``wbt-play``: the robot starts at the reference state of ``--start-frame`` with no randomization,
no pushes and no observation noise. It runs once through the motion, or until a training termination fires.
``--push-scale`` adds training's ``push_robot`` disturbance (a random root velocity kick every 1-3 s), scaled.
``--dr`` applies all of training's randomization instead: startup model randomization (friction, COM, encoder bias),
the reset-state perturbation, observation noise, the sampled actuator and observation delays, and pushes.

Metrics follow ``mjlab.tasks.tracking.scripts.evaluate``: each post-step robot state is scored against the reference
frame the policy was commanded with, which is how the reward and terminations score it in training.
"""

from __future__ import annotations

import json
import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import time
import tyro
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mjlab
from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg, IdealPdActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr as mjlab_dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.string import resolve_matching_names
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.utils.string import resolve_expr

import whole_body_tracking_mjlab  # noqa: F401
import whole_body_tracking_mjlab.tasks.tracking.mdp as mdp

ENTITY = "robot"


@dataclass(frozen=True)
class Sim2SimConfig:
    onnx_file: str
    """Exported policy (``logs/rsl_rl/<experiment>/<run>/<run>.onnx``). The reference motion is read from it."""
    start_frame: int = 0
    """Motion frame to start from."""
    num_frames: int | None = None
    """Number of policy steps to run. Defaults to the rest of the motion."""
    dr: bool = False
    """Apply training's domain randomization: foot friction, torso COM offset and encoder bias sampled at startup,
    the perturbed reset state, observation noise, actuator and observation delays resampled on training's schedule,
    and pushes (``--push-scale`` defaults to 1)."""
    actuator_delay: int | None = None
    """Fixed command delay in physics steps for actuators trained with one. Defaults to the middle of the trained
    range, rounded down. Not allowed with ``--dr``, which samples it."""
    obs_delay: int | None = None
    """Fixed delay in policy steps for observation terms trained with one. Defaults to the middle of the trained
    range, rounded down. Not allowed with ``--dr``, which samples it."""
    push_scale: float | None = None
    """Push the robot like training's ``push_robot`` event (a random root velocity kick at random intervals), with
    the sampled velocity range scaled by this factor. 0 turns pushes off; 1 is the training range. Defaults to 1 with
    ``--dr``, else 0."""
    seed: int = 0
    """Seed for pushes and, with ``--dr``, every other random draw."""
    stop_on_termination: bool = True
    """End the rollout when a training termination fires. Turn off to watch the whole motion."""
    viewer: bool = False
    """Show the rollout in the MuJoCo viewer in real time. The reference bodies, re-anchored to the robot's position and
    heading, are drawn as green spheres."""
    video_file: str | None = None
    """Render the rollout to this mp4 (offscreen; set ``MUJOCO_GL=egl`` on a headless machine)."""
    output_file: str | None = None
    """Write the metrics to this JSON file."""
    trajectory_file: str | None = None
    """Write per-step robot state, reference and actions to this npz file."""


##
# Quaternion helpers (w, x, y, z), matching mjlab.utils.lab_api.math.
##


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0]) / np.sum(q * q, axis=-1, keepdims=True)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_inv(q), v)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ]
    )


def yaw_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zeros = np.zeros_like(yaw)
    return np.stack([np.cos(yaw / 2), zeros, zeros, np.sin(yaw / 2)], axis=-1)


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    diff = quat_mul(quat_inv(q1), q2)
    return 2.0 * np.arctan2(np.linalg.norm(diff[..., 1:], axis=-1), np.abs(diff[..., 0]))


def _se3_ranges(ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    """(6, 2) ranges for x, y, z, roll, pitch, yaw; missing keys are (0, 0)."""
    return np.array([ranges.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")], dtype=np.float64)


##
# Policy.
##


def _parse_metadata_value(value: str) -> list | str:
    if "," not in value and ";" not in value:
        return value
    entries = []
    for entry in value.split(","):
        parts = [_to_number(p) for p in entry.split(";")]
        entries.append(parts if len(parts) > 1 else parts[0])
    return entries


def _to_number(s: str) -> float | str:
    try:
        return float(s)
    except ValueError:
        return s


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


class OnnxPolicy:
    """The exported policy plus the reference motion it bundles, indexed by ``time_step``."""

    def __init__(self, path: str):
        model = onnx.load(path)
        self.metadata = {p.key: _parse_metadata_value(p.value) for p in model.metadata_props}
        motion_buffer = next(i for i in model.graph.initializer if i.name.split(".")[0] == "joint_pos")
        self.num_frames = int(motion_buffer.dims[0])
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.obs_dim = self.session.get_inputs()[0].shape[1]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.joint_names: list[str] = _as_list(self.metadata["joint_names"])
        self.body_names: list[str] = _as_list(self.metadata["body_names"])
        self.anchor_body_name: str = self.metadata["anchor_body_name"]
        self.observation_names: list[str] = _as_list(self.metadata["observation_names"])
        n = len(self.observation_names)
        self.obs_scale = _as_list(self.metadata.get("observation_terms_scale", [1.0] * n))
        self.obs_clip = _as_list(self.metadata.get("observation_terms_clip", [[-np.inf, np.inf]] * n))
        history = _as_list(self.metadata.get("observation_terms_history_length", [0.0] * n))
        if any(h > 1 for h in history):
            raise NotImplementedError("Observation history is not supported.")

    def __call__(self, obs: np.ndarray | None, time_step: int) -> dict[str, np.ndarray]:
        """Run the policy on ``obs`` (zeros if None, to read only the reference) at motion frame ``time_step``."""
        if obs is None:
            obs = np.zeros(self.obs_dim)
        outputs = self.session.run(
            None,
            {
                "obs": obs.astype(np.float32)[None],
                "time_step": np.array([[time_step]], dtype=np.float32),
            },
        )
        return {name: out[0].astype(np.float64) for name, out in zip(self.output_names, outputs)}


##
# Actuators.
##


class Actuators:
    """Joint-space actuator laws of the task's robot, applied to CPU MuJoCo ``ctrl``.

    MuJoCo position actuators (G1) take the joint target as ``ctrl``. Ideal PD and DC motor actuators (Sprout) are
    ``<motor>`` actuators whose torque mjlab computes in torch; they are recomputed here every physics step.
    """

    def __init__(
        self,
        env_cfg: ManagerBasedRlEnvCfg,
        model: mujoco.MjModel,
        joint_names: list[str],
        delay: int | None,
        rng: np.random.Generator | None = None,
    ):
        """``delay`` fixes the command delay of delayed actuators; with ``rng`` it is sampled like training instead."""
        n = len(joint_names)
        joint_ids = [model.joint(f"{ENTITY}/{name}").id for name in joint_names]
        self.qpos_adr = model.jnt_qposadr[joint_ids]
        self.dof_adr = model.jnt_dofadr[joint_ids]
        self.ctrl_ids = np.full(n, -1)
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] in joint_ids:
                self.ctrl_ids[joint_ids.index(model.actuator_trnid[a, 0])] = a
        missing = [joint_names[i] for i in np.flatnonzero(self.ctrl_ids < 0)]
        if missing:
            raise ValueError(f"Joints without actuators: {missing}")

        self.is_torque = np.zeros(n, dtype=bool)
        self.is_dc = np.zeros(n, dtype=bool)
        self.kp, self.kd = np.zeros(n), np.zeros(n)
        self.effort_limit, self.saturation_effort, self.velocity_limit = np.full(n, np.inf), np.ones(n), np.ones(n)
        self.lag = np.zeros(n, dtype=int)
        self.rng = rng
        # (joint ids, min lag, max lag, update period) per actuator config; each has its own DelayBuffer in mjlab.
        self.delay_groups: list[tuple[list[int], int, int, int]] = []
        self._step = 0
        matched = np.zeros(n, dtype=bool)
        for cfg in env_cfg.scene.entities[ENTITY].articulation.actuators:
            ids, _ = resolve_matching_names(cfg.target_names_expr, joint_names)
            ids = [i for i in ids if not matched[i]]
            matched[ids] = True
            if isinstance(cfg, BuiltinPositionActuatorCfg):
                pass
            elif isinstance(cfg, IdealPdActuatorCfg):
                self.is_torque[ids] = True
                self.kp[ids], self.kd[ids], self.effort_limit[ids] = cfg.stiffness, cfg.damping, cfg.effort_limit
                if isinstance(cfg, DcMotorActuatorCfg):
                    self.is_dc[ids] = True
                    self.saturation_effort[ids], self.velocity_limit[ids] = cfg.saturation_effort, cfg.velocity_limit
            else:
                raise NotImplementedError(f"{type(cfg).__name__} is not supported.")
            if cfg.delay_max_lag > 0:
                self.delay_groups.append((ids, cfg.delay_min_lag, cfg.delay_max_lag, cfg.delay_update_period))
                self.lag[ids] = delay if delay is not None else (cfg.delay_min_lag + cfg.delay_max_lag) // 2
        max_lag = max((g[2] for g in self.delay_groups), default=0) if rng is not None else int(self.lag.max())
        self._history: deque[np.ndarray] = deque(maxlen=max_lag + 1)

    def reset(self) -> None:
        self._history.clear()
        self._step = 0

    def _update_lags(self) -> None:
        """``DelayBuffer._update_lags`` without a phase offset: resample at reset and every ``update_period`` steps."""
        for ids, min_lag, max_lag, period in self.delay_groups:
            if period == 0 or self._step % period == 0:
                self.lag[ids] = self.rng.integers(min_lag, max_lag + 1)
        self._step += 1

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray) -> None:
        """Write ``ctrl`` for one physics step. ``target`` is the policy's joint position target."""
        if not self._history:
            self._history.extend([target] * self._history.maxlen)
        else:
            self._history.append(target)
        if self.rng is not None:
            self._update_lags()
        # history[-1] is the newest target; each joint reads the one ``lag`` physics steps old.
        delayed = np.array([self._history[-1 - lag][j] for j, lag in enumerate(self.lag)])

        ctrl = delayed.copy()
        q, qd = data.qpos[self.qpos_adr], data.qvel[self.dof_adr]
        torque = self.kp * (delayed - q) - self.kd * qd
        torque = np.clip(torque, -self.effort_limit, self.effort_limit)
        # mjlab.actuator.dc_actuator.dc_motor_clip
        vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self.saturation_effort)
        vel = np.clip(qd, -vel_at_effort_lim, vel_at_effort_lim)
        max_effort = np.minimum(self.saturation_effort * (1.0 - vel / self.velocity_limit), self.effort_limit)
        min_effort = np.maximum(self.saturation_effort * (-1.0 - vel / self.velocity_limit), -self.effort_limit)
        torque = np.where(self.is_dc, np.clip(torque, min_effort, max_effort), torque)
        ctrl[self.is_torque] = torque[self.is_torque]
        data.ctrl[self.ctrl_ids] = ctrl


##
# Rollout.
##


class Sim2Sim:
    def __init__(self, task_id: str, cfg: Sim2SimConfig):
        if cfg.dr and (cfg.actuator_delay is not None or cfg.obs_delay is not None):
            raise ValueError("--dr samples the delays; drop --actuator-delay and --obs-delay.")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # The training config holds the randomization that the play config removes.
        self.train_cfg = load_env_cfg(task_id, play=False)
        self.env_cfg = load_env_cfg(task_id, play=True)
        self.env_cfg.scene.num_envs = 1
        self.model = Scene(self.env_cfg.scene, device="cpu").compile()
        self.env_cfg.sim.mujoco.apply(self.model)
        self.data = mujoco.MjData(self.model)
        self.decimation = self.env_cfg.decimation
        self.step_dt = self.model.opt.timestep * self.decimation

        self.policy = OnnxPolicy(cfg.onnx_file)
        p = self.policy
        motion_cmd = self.env_cfg.commands["motion"]
        assert isinstance(motion_cmd, mdp.MotionCommandCfg)
        if list(motion_cmd.body_names) != p.body_names or motion_cmd.anchor_body_name != p.anchor_body_name:
            raise ValueError(f"{cfg.onnx_file} tracks different bodies than task {task_id}.")
        actor_terms = self.env_cfg.observations["actor"].terms
        if list(actor_terms) != p.observation_names:
            raise ValueError(
                f"{cfg.onnx_file} observes {p.observation_names}, task {task_id} observes {list(actor_terms)}."
            )
        self.actor_terms = actor_terms

        self.actuators = Actuators(
            self.env_cfg, self.model, p.joint_names, cfg.actuator_delay, rng=self.rng if cfg.dr else None
        )
        self.qpos_adr, self.dof_adr = self.actuators.qpos_adr, self.actuators.dof_adr

        # Exact default pose (the entity's init_state keyframe) and action scale, checked against the rounded metadata.
        self.default_joint_pos = self.model.key("init_state").qpos[self.qpos_adr].copy()
        action_cfg = self.env_cfg.actions["joint_pos"]
        assert isinstance(action_cfg, JointPositionActionCfg) and action_cfg.use_default_offset
        scale = action_cfg.scale
        self.action_scale = np.array(
            resolve_expr(scale, tuple(p.joint_names), 1.0) if isinstance(scale, dict) else scale
        )
        for name, value in (("default_joint_pos", self.default_joint_pos), ("action_scale", self.action_scale)):
            if not np.allclose(value, np.array(p.metadata[name], dtype=np.float64), atol=1e-3):
                raise ValueError(f"{name} of {cfg.onnx_file} does not match task {task_id}.")
        free_joint = next(
            j
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and self.model.joint(j).name.startswith(f"{ENTITY}/")
        )
        self.root_qpos_adr = self.model.jnt_qposadr[free_joint]
        self.root_dof_adr = self.model.jnt_dofadr[free_joint]
        self.body_ids = np.array([self.model.body(f"{ENTITY}/{name}").id for name in p.body_names])
        self.anchor_index = p.body_names.index(p.anchor_body_name)

        # Soft joint limits used to clip the reset state, as in MotionCommand._write_reference_state_to_sim.
        factor = self.env_cfg.scene.entities[ENTITY].articulation.soft_joint_pos_limit_factor
        joint_ids = [self.model.joint(f"{ENTITY}/{name}").id for name in p.joint_names]
        lo, hi = self.model.jnt_range[joint_ids].T
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * factor
        limited = self.model.jnt_limited[joint_ids].astype(bool)
        self.soft_lo = np.where(limited, mid - half, -np.inf)
        self.soft_hi = np.where(limited, mid + half, np.inf)

        # Observation delay per term, in policy steps: fixed, or with --dr resampled every step like training's
        # DelayBuffer (update_period 0).
        delayed_terms = {name: term for name, term in actor_terms.items() if term.delay_max_lag > 0}
        for term in delayed_terms.values():
            assert term.delay_update_period == 0 and term.delay_hold_prob == 0.0
        self.obs_lag_range = {name: (term.delay_min_lag, term.delay_max_lag) for name, term in delayed_terms.items()}
        self.obs_lag = {
            name: cfg.obs_delay if cfg.obs_delay is not None else (lo + hi) // 2
            for name, (lo, hi) in self.obs_lag_range.items()
        }
        self._obs_history: dict[str, deque[np.ndarray]] = {
            name: deque(maxlen=(hi if cfg.dr else self.obs_lag[name]) + 1)
            for name, (lo, hi) in self.obs_lag_range.items()
        }

        # Observation noise, applied (before the delay) only with --dr, as training's actor group corrupts.
        self.obs_noise: dict[str, tuple[float, float]] = {}
        train_actor = self.train_cfg.observations["actor"]
        if cfg.dr and train_actor.enable_corruption:
            for name, term in train_actor.terms.items():
                if term.noise is None:
                    continue
                if not isinstance(term.noise, UniformNoiseCfg) or term.noise.operation != "add":
                    raise NotImplementedError(f"Noise {term.noise} on {name} is not supported.")
                self.obs_noise[name] = (term.noise.n_min, term.noise.n_max)

        terms = self.env_cfg.terminations
        self.ee_body_names = tuple(terms["ee_body_pos"].params["body_names"])
        self.ee_indices = [p.body_names.index(name) for name in self.ee_body_names]
        self.terminations = {name: term for name, term in terms.items() if not term.time_out}
        known = {mdp.bad_anchor_pos_z_only, mdp.bad_anchor_ori, mdp.bad_motion_body_pos_z_only}
        unknown = [name for name, term in self.terminations.items() if term.func not in known]
        if unknown:
            raise NotImplementedError(f"Terminations {unknown} are not supported.")

        self.last_action = np.zeros(len(p.joint_names))

        # The training push (removed from the play config), applied at the same interval with a scaled range.
        push = self.train_cfg.events["push_robot"]
        assert push.func is mdp.push_by_setting_velocity and push.interval_range_s is not None
        self.push_interval_s = push.interval_range_s
        self.push_velocity_range = _se3_ranges(push.params["velocity_range"])

        self.encoder_bias = np.zeros(len(p.joint_names))
        self.dr_samples: dict[str, list | float] = {}
        if cfg.dr:
            self._randomize_startup()

    def _randomize_startup(self) -> None:
        """Training's ``startup`` events, applied to this model once."""
        local = lambda name: name.removeprefix(f"{ENTITY}/")  # noqa: E731
        nominal_ipos = self.model.body_ipos.copy()
        for name, term in self.train_cfg.events.items():
            if term.mode == "interval" and term.func is mdp.push_by_setting_velocity:
                continue
            if term.mode != "startup":
                raise NotImplementedError(f"Event {name} ({term.mode}) is not supported.")
            params = dict(term.params)
            asset_cfg = params.pop("asset_cfg")
            if params.pop("distribution", "uniform") != "uniform" or params.pop("axes", None) is not None:
                raise NotImplementedError(f"Event {name}: only uniform sampling on default axes is supported.")
            if term.func is mjlab_dr.geom_friction:
                geom_names = [local(self.model.geom(i).name) for i in range(self.model.ngeom)]
                ids, _ = resolve_matching_names(asset_cfg.geom_names, geom_names)
                ids = [i for i in ids if self.model.geom(i).name.startswith(f"{ENTITY}/")]
                lo, hi = params["ranges"]
                n = 1 if params.get("shared_random", False) else len(ids)
                values = self._combine(
                    self.model.geom_friction[ids, 0], self.rng.uniform(lo, hi, n), params["operation"]
                )
                self.model.geom_friction[ids, 0] = values
                self.dr_samples["foot_friction"] = values.tolist()
            elif term.func is mjlab_dr.body_com_offset:
                ids = [self.model.body(f"{ENTITY}/{body}").id for body in asset_cfg.body_names]
                ranges = params["ranges"]
                assert not params.get("shared_random", False)
                for axis, (lo, hi) in ranges.items():
                    sample = self.rng.uniform(lo, hi, len(ids))
                    self.model.body_ipos[ids, axis] = self._combine(
                        self.model.body_ipos[ids, axis], sample, params["operation"]
                    )
                self.dr_samples["com_offset"] = (self.model.body_ipos[ids] - nominal_ipos[ids]).tolist()
            elif term.func is mjlab_dr.encoder_bias:
                assert asset_cfg.joint_names is None, "Encoder bias on a joint subset is not supported."
                lo, hi = params["bias_range"]
                self.encoder_bias = self.rng.uniform(lo, hi, len(self.encoder_bias))
                self.dr_samples["encoder_bias_abs_max"] = float(np.abs(self.encoder_bias).max())
            else:
                raise NotImplementedError(f"Event {name} ({term.func.__name__}) is not supported.")
        # body_ipos changes need derived constants recomputed, as mjlab's set_const.
        mujoco.mj_setConst(self.model, self.data)

    @staticmethod
    def _combine(base: np.ndarray, sample: np.ndarray, operation: str) -> np.ndarray:
        if operation == "abs":
            return np.broadcast_to(sample, base.shape).copy()
        if operation == "add":
            return base + sample
        if operation == "scale":
            return base * sample
        raise NotImplementedError(f"Operation {operation} is not supported.")

    # Robot state.

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    @property
    def body_pos_w(self) -> np.ndarray:
        return self.data.xpos[self.body_ids].copy()

    @property
    def body_quat_w(self) -> np.ndarray:
        return self.data.xquat[self.body_ids].copy()

    def reset(self, frame: int) -> None:
        """Put the robot in the reference state of ``frame``, like ``MotionCommand.reset_to_frame``."""
        self.reset_to(self.policy(None, frame))

    def reset_to(self, ref: dict[str, np.ndarray]) -> None:
        """Put the robot in the reference state ``ref`` (policy joint and body order)."""
        mujoco.mj_resetData(self.model, self.data)
        root_pos, root_quat = ref["body_pos_w"][0].copy(), ref["body_quat_w"][0].copy()
        root_vel = np.concatenate([ref["body_lin_vel_w"][0], ref["body_ang_vel_w"][0]])
        joint_pos = ref["joint_pos"].copy()
        if self.cfg.dr:
            # MotionCommand._resample_command's perturbation of the reference state.
            motion_cmd = self.train_cfg.commands["motion"]
            pose = self.rng.uniform(*_se3_ranges(motion_cmd.pose_range).T)
            root_pos += pose[:3]
            root_quat = quat_mul(quat_from_euler_xyz(*pose[3:]), root_quat)
            root_vel += self.rng.uniform(*_se3_ranges(motion_cmd.velocity_range).T)
            joint_pos += self.rng.uniform(*motion_cmd.joint_position_range, len(joint_pos))
        self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = root_pos
        self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = root_quat
        # Free joint qvel holds world-frame linear and body-frame angular velocity.
        self.data.qvel[self.root_dof_adr : self.root_dof_adr + 3] = root_vel[:3]
        self.data.qvel[self.root_dof_adr + 3 : self.root_dof_adr + 6] = quat_apply_inverse(root_quat, root_vel[3:])
        self.data.qpos[self.qpos_adr] = np.clip(joint_pos, self.soft_lo, self.soft_hi)
        self.data.qvel[self.dof_adr] = ref["joint_vel"]
        mujoco.mj_forward(self.model, self.data)
        self.actuators.reset()
        for history in self._obs_history.values():
            history.clear()
        self.last_action[:] = 0.0

    def observation(self, ref: dict[str, np.ndarray]) -> np.ndarray:
        p = self.policy
        robot_anchor_pos = self.data.xpos[self.body_ids[self.anchor_index]]
        robot_anchor_quat = self.data.xquat[self.body_ids[self.anchor_index]]
        ref_anchor_pos = ref["body_pos_w"][self.anchor_index]
        ref_anchor_quat = ref["body_quat_w"][self.anchor_index]

        parts = []
        for i, name in enumerate(p.observation_names):
            term = self.actor_terms[name]
            if term.func is mdp.generated_commands:
                value = np.concatenate([ref["joint_pos"], ref["joint_vel"]])
            elif term.func is mdp.motion_anchor_pos_b:
                value = quat_apply_inverse(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
            elif term.func is mdp.motion_anchor_ori_b:
                rel = quat_mul(quat_inv(robot_anchor_quat), ref_anchor_quat)
                value = matrix_from_quat(rel)[:, :2].reshape(-1)
            elif term.func is mdp.builtin_sensor:
                sensor = self.model.sensor(term.params["sensor_name"])
                value = self.data.sensordata[sensor.adr[0] : sensor.adr[0] + sensor.dim[0]].copy()
            elif term.func is mdp.joint_pos_rel:
                bias = self.encoder_bias if term.params.get("biased", False) else 0.0
                value = self.joint_pos + bias - self.default_joint_pos
            elif term.func is mdp.joint_vel_rel:
                value = self.joint_vel
            elif term.func is mdp.last_action:
                value = self.last_action.copy()
            else:
                raise NotImplementedError(f"Observation term {name} ({term.func.__name__}) is not supported.")

            # Same order as ObservationManager.compute_group: noise, clip, scale, then delay.
            if name in self.obs_noise:
                value = value + self.rng.uniform(*self.obs_noise[name], value.shape)
            lo, hi = p.obs_clip[i] if isinstance(p.obs_clip[i], list) else (-np.inf, np.inf)
            value = np.clip(value, lo, hi) * np.asarray(p.obs_scale[i])

            if name in self._obs_history:
                history = self._obs_history[name]
                if not history:
                    history.extend([value] * history.maxlen)
                else:
                    history.append(value)
                if self.cfg.dr:
                    self.obs_lag[name] = int(
                        self.rng.integers(self.obs_lag_range[name][0], self.obs_lag_range[name][1] + 1)
                    )
                value = history[-1 - self.obs_lag[name]]
            parts.append(value)
        obs = np.concatenate(parts)
        if obs.shape[0] != p.obs_dim:
            raise ValueError(f"Built a {obs.shape[0]}-dim observation, the policy takes {p.obs_dim}.")
        return obs

    def relative_reference(self, ref: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Reference body poses re-anchored to the robot's anchor xy and yaw (``update_relative_body_poses``)."""
        ref_anchor_pos = ref["body_pos_w"][self.anchor_index]
        ref_anchor_quat = ref["body_quat_w"][self.anchor_index]
        robot_anchor_pos = self.data.xpos[self.body_ids[self.anchor_index]]
        robot_anchor_quat = self.data.xquat[self.body_ids[self.anchor_index]]
        delta_pos = robot_anchor_pos.copy()
        delta_pos[2] = ref_anchor_pos[2]
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))
        n = len(self.body_ids)
        body_quat_rel = quat_mul(np.tile(delta_ori, (n, 1)), ref["body_quat_w"])
        body_pos_rel = delta_pos + quat_apply(np.tile(delta_ori, (n, 1)), ref["body_pos_w"] - ref_anchor_pos)
        return body_pos_rel, body_quat_rel

    def evaluate(self, ref: dict[str, np.ndarray]) -> tuple[dict[str, float], list[str]]:
        """Tracking errors of the current robot state against ``ref``, and the terminations that fire."""
        body_pos, body_quat = self.body_pos_w, self.body_quat_w
        body_pos_rel, body_quat_rel = self.relative_reference(ref)
        a, ee = self.anchor_index, self.ee_indices
        errors = {
            "mpkpe": np.linalg.norm(ref["body_pos_w"] - body_pos, axis=-1).mean(),
            "r_mpkpe": np.linalg.norm(body_pos_rel - body_pos, axis=-1).mean(),
            "joint_pos_error": np.sqrt(np.mean((ref["joint_pos"] - self.joint_pos) ** 2)),
            "joint_vel_error": np.sqrt(np.mean((ref["joint_vel"] - self.joint_vel) ** 2)),
            "ee_pos_error": np.linalg.norm(body_pos_rel[ee] - body_pos[ee], axis=-1).mean(),
            "ee_ori_error": quat_error_magnitude(body_quat_rel[ee], body_quat[ee]).mean(),
            "anchor_pos_error": np.linalg.norm(ref["body_pos_w"][a] - body_pos[a]),
            "anchor_ori_error": quat_error_magnitude(ref["body_quat_w"][a], body_quat[a]),
        }

        fired = []
        gravity = np.array([0.0, 0.0, -1.0])
        for name, term in self.terminations.items():
            threshold = term.params["threshold"]
            if term.func is mdp.bad_anchor_pos_z_only:
                bad = abs(ref["body_pos_w"][a, 2] - body_pos[a, 2]) > threshold
            elif term.func is mdp.bad_anchor_ori:
                ref_gravity_z = quat_apply_inverse(ref["body_quat_w"][a], gravity)[2]
                robot_gravity_z = quat_apply_inverse(body_quat[a], gravity)[2]
                bad = abs(ref_gravity_z - robot_gravity_z) > threshold
            else:
                ids = [self.policy.body_names.index(n) for n in term.params["body_names"]]
                bad = np.any(np.abs(body_pos_rel[ids, 2] - body_pos[ids, 2]) > threshold)
            if bad:
                fired.append(name)
        return {k: float(v) for k, v in errors.items()}, fired

    def sample_push_interval(self) -> float:
        return float(self.rng.uniform(*self.push_interval_s))

    def push(self, scale: float) -> np.ndarray:
        """Add a random world-frame (linear, angular) velocity to the root, like ``push_by_setting_velocity``."""
        lo, hi = self.push_velocity_range.T * scale
        delta = self.rng.uniform(lo, hi)
        quat = self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7]
        qvel = self.data.qvel[self.root_dof_adr : self.root_dof_adr + 6]
        qvel[:3] += delta[:3]
        # Free joint angular velocity is in the body frame.
        qvel[3:] = quat_apply_inverse(quat, quat_apply(quat, qvel[3:]) + delta[3:])
        mujoco.mj_forward(self.model, self.data)
        return delta

    def step(self, obs: np.ndarray, time_step: int) -> dict[str, np.ndarray]:
        """Run the policy on ``obs`` and simulate one policy step. Returns the policy outputs for ``time_step``."""
        out = self.policy(obs, time_step)
        action = out["actions"]
        # JointPositionAction.apply_actions subtracts the encoder bias from the target.
        target = self.default_joint_pos + self.action_scale * action - self.encoder_bias
        for _ in range(self.decimation):
            self.actuators.apply(self.model, self.data, target)
            mujoco.mj_step(self.model, self.data)
        # mjlab refreshes derived quantities (xpos, sensors) once before computing observations.
        mujoco.mj_forward(self.model, self.data)
        self.last_action = action
        return out


def mpjpe_metrics(body_pos: np.ndarray, ref_body_pos: np.ndarray) -> dict[str, float]:
    """Pose-sequence metrics of motion-tracking papers (PHC, OmniH2O, ASAP), in mm, over all tracked bodies.

    ``mpjpe_g`` is the global position error; ``mpjpe_l`` the error after subtracting each pose's root (first body)
    position, which removes global drift but keeps heading error. ``vel_error`` (mm/frame) and ``acc_error``
    (mm/frame^2) compare first and second finite differences of the body positions, one frame per policy step.

    Args:
      body_pos: Robot body positions, (T, B, 3).
      ref_body_pos: Reference body positions, (T, B, 3).
    """
    metrics = {
        "mpjpe_g": np.linalg.norm(body_pos - ref_body_pos, axis=-1).mean() * 1000,
        "mpjpe_l": (
            np.linalg.norm((body_pos - body_pos[:, :1]) - (ref_body_pos - ref_body_pos[:, :1]), axis=-1).mean() * 1000
        ),
    }
    if len(body_pos) > 1:
        vel = np.diff(body_pos, axis=0) - np.diff(ref_body_pos, axis=0)
        metrics["vel_error"] = np.linalg.norm(vel, axis=-1).mean() * 1000
    if len(body_pos) > 2:
        acc = np.diff(body_pos, n=2, axis=0) - np.diff(ref_body_pos, n=2, axis=0)
        metrics["acc_error"] = np.linalg.norm(acc, axis=-1).mean() * 1000
    return {k: float(v) for k, v in metrics.items()}


##
# Visualization.
##


def _add_reference_markers(scene: mujoco.MjvScene, body_pos: np.ndarray) -> None:
    for pos in body_pos:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.03, 0.0, 0.0]),
            pos=pos,
            mat=np.eye(3).reshape(-1),
            rgba=np.array([0.2, 0.9, 0.2, 0.6], dtype=np.float32),
        )
        scene.ngeom += 1


def _add_push_arrow(scene: mujoco.MjvScene, origin: np.ndarray, lin_vel: np.ndarray) -> None:
    """Blue arrow from the anchor along the push's linear velocity, 1 m per m/s."""
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(lin_vel) < 1e-6:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).reshape(-1),
        rgba=np.array([0.1, 0.4, 1.0, 1.0], dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.04, origin, origin + lin_vel)
    scene.ngeom += 1


def _configure_camera(cam: mujoco.MjvCamera, sim: Sim2Sim) -> None:
    viewer = sim.env_cfg.viewer
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = int(sim.body_ids[sim.anchor_index])
    cam.distance = viewer.distance
    cam.azimuth = viewer.azimuth
    cam.elevation = viewer.elevation


##
# Main.
##


def run_sim2sim(task_id: str, cfg: Sim2SimConfig) -> dict:
    sim = Sim2Sim(task_id, cfg)
    p = sim.policy
    if not 0 <= cfg.start_frame < p.num_frames:
        raise ValueError(f"--start-frame must be in [0, {p.num_frames}).")
    end_frame = p.num_frames if cfg.num_frames is None else min(p.num_frames, cfg.start_frame + cfg.num_frames)

    print(f"[INFO] Task {task_id}, policy {cfg.onnx_file}")
    print(
        f"[INFO] Motion frames {cfg.start_frame}-{end_frame - 1} of {p.num_frames}, policy dt {sim.step_dt:.3f} s, "
        f"physics dt {sim.model.opt.timestep:.3f} s"
    )
    lags = sorted(set(sim.actuators.lag.tolist()))
    push_scale = cfg.push_scale if cfg.push_scale is not None else (1.0 if cfg.dr else 0.0)
    if cfg.dr:
        groups = sorted({(lo, hi, period) for _, lo, hi, period in sim.actuators.delay_groups})
        print(
            f"[INFO] Training DR (seed {cfg.seed}): {sim.dr_samples}, obs noise on {list(sim.obs_noise)}, actuator"
            f" delay {groups} (min, max, resample period) physics steps, observation delay {sim.obs_lag_range} policy"
            " steps"
        )
    else:
        print(f"[INFO] Actuator delay {lags} physics steps, observation delay {sim.obs_lag or 'none'} policy steps")
    if push_scale > 0:
        print(
            f"[INFO] Pushes every {sim.push_interval_s[0]:g}-{sim.push_interval_s[1]:g} s, velocity range x"
            f"{push_scale:g} of training (seed {cfg.seed})"
        )

    viewer = None
    if cfg.viewer:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False)
        _configure_camera(viewer.cam, sim)
    renderer, camera, video = None, None, None
    if cfg.video_file:
        import mediapy

        renderer = mujoco.Renderer(sim.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        _configure_camera(camera, sim)
        Path(cfg.video_file).parent.mkdir(parents=True, exist_ok=True)
        # Frames are encoded as they are rendered; a full motion would not fit in memory.
        video = mediapy.VideoWriter(cfg.video_file, shape=(480, 640), fps=1.0 / sim.step_dt)
        video.__enter__()

    sim.reset(cfg.start_frame)
    ref = sim.policy(None, cfg.start_frame)
    sums: dict[str, float] = {}
    body_pos_seq, ref_body_pos_seq = [], []
    log_keys = ("time_step", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "action", "ref_joint_pos")
    log: dict[str, list] = {k: [] for k in (*log_keys, "ref_body_pos_w")}
    termination, steps = None, 0
    pushes: list[dict] = []
    push_time_left = sim.sample_push_interval()
    arrow, arrow_steps = np.zeros(3), 0
    wall_start = time.perf_counter()

    for time_step in range(cfg.start_frame, end_frame):
        step_start = time.perf_counter()
        obs = sim.observation(ref)
        # The returned reference is the frame this step is commanded with; score the post-step state against it.
        ref = sim.step(obs, time_step)
        errors, fired = sim.evaluate(ref)
        body_pos_seq.append(sim.body_pos_w)
        ref_body_pos_seq.append(ref["body_pos_w"])
        steps += 1
        for k, v in errors.items():
            sums[k] = sums.get(k, 0.0) + v

        if cfg.trajectory_file:
            log["time_step"].append(time_step)
            log["joint_pos"].append(sim.joint_pos)
            log["joint_vel"].append(sim.joint_vel)
            log["body_pos_w"].append(sim.body_pos_w)
            log["body_quat_w"].append(sim.body_quat_w)
            log["action"].append(ref["actions"])
            log["ref_joint_pos"].append(ref["joint_pos"])
            log["ref_body_pos_w"].append(ref["body_pos_w"])

        # Interval events run after the step, before the next observation.
        if push_scale > 0:
            push_time_left -= sim.step_dt
            if push_time_left <= 1e-6:
                delta = sim.push(push_scale)
                pushes.append(
                    {"time_s": steps * sim.step_dt, "lin_vel": delta[:3].tolist(), "ang_vel": delta[3:].tolist()}
                )
                push_time_left = sim.sample_push_interval()
                arrow, arrow_steps = delta[:3], int(round(0.5 / sim.step_dt))

        # The next observation uses the next frame, as MotionCommand advances after each step.
        ref = sim.policy(None, min(time_step + 1, p.num_frames - 1))

        markers = sim.relative_reference(ref)[0]
        if renderer is not None:
            renderer.update_scene(sim.data, camera)
            _add_reference_markers(renderer.scene, markers)
            if arrow_steps > 0:
                _add_push_arrow(renderer.scene, sim.data.xpos[sim.body_ids[sim.anchor_index]], arrow)
            video.add_image(renderer.render())
        if viewer is not None:
            if not viewer.is_running():
                break
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                _add_reference_markers(viewer.user_scn, markers)
                if arrow_steps > 0:
                    _add_push_arrow(viewer.user_scn, sim.data.xpos[sim.body_ids[sim.anchor_index]], arrow)
            viewer.sync()
            time.sleep(max(0.0, sim.step_dt - (time.perf_counter() - step_start)))

        arrow_steps -= 1

        if fired and termination is None:
            termination = {"frame": time_step, "time_s": steps * sim.step_dt, "terms": fired}
            print(f"[INFO] Terminated at frame {time_step} ({steps * sim.step_dt:.2f} s): {', '.join(fired)}")
            if cfg.stop_on_termination:
                break

    wall_time = time.perf_counter() - wall_start
    if viewer is not None:
        viewer.close()

    metrics = {
        "task": task_id,
        "onnx_file": cfg.onnx_file,
        "success": termination is None and steps == end_frame - cfg.start_frame,
        "steps": steps,
        "duration_s": steps * sim.step_dt,
        "termination": termination,
        "dr": cfg.dr,
        "push_scale": push_scale,
        "num_pushes": len(pushes),
        **{k: v / max(steps, 1) for k, v in sums.items()},
        **(mpjpe_metrics(np.array(body_pos_seq), np.array(ref_body_pos_seq)) if steps else {}),
        "realtime_factor": steps * sim.step_dt / wall_time,
    }

    print("\n" + "=" * 50)
    print("Sim2Sim Results (CPU MuJoCo)")
    print("=" * 50)
    for name, value in metrics.items():
        if name in ("task", "onnx_file"):
            continue
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    print("=" * 50)

    if cfg.output_file:
        path = Path(cfg.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**metrics, "dr_samples": sim.dr_samples, "pushes": pushes}, indent=2))
        print(f"[INFO] Metrics saved to {path}")
    if cfg.trajectory_file:
        path = Path(cfg.trajectory_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        push_log = {
            "push_time_s": np.array([push["time_s"] for push in pushes]),
            "push_vel": np.array([push["lin_vel"] + push["ang_vel"] for push in pushes]).reshape(-1, 6),
        }
        np.savez(path, dt=sim.step_dt, joint_names=p.joint_names, body_names=p.body_names, **log, **push_log)
        print(f"[INFO] Trajectory saved to {path}")
    if renderer is not None:
        video.__exit__(None, None, None)
        renderer.close()
        print(f"[INFO] Video saved to {cfg.video_file}")

    return metrics


def main():
    tracking_tasks = [t for t in list_tasks() if "Tracking" in t]
    task_id, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(tracking_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(Sim2SimConfig, args=remaining_args, prog=f"wbt-sim2sim {task_id}", config=mjlab.TYRO_FLAGS)
    run_sim2sim(task_id, cfg)


if __name__ == "__main__":
    main()
