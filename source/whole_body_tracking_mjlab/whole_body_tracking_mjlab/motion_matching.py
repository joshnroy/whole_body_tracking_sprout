"""Motion matching over a single reference clip, steered by forward/backward speed and yaw rate commands.

The database is one training npz (``wbt-csv-to-npz`` output, e.g. the walking clip a tracking policy was trained on),
so every generated frame is a frame of that clip, which the policy has learned to track. The matcher follows Holden's
"Code vs Data Driven Displacement" / "Learned Motion Matching" (without the learned parts):

- Each frame has a feature vector: the root (floating-base body) trajectory 0.4, 0.8 and 1.2 s ahead (position and
  heading), the foot positions and velocities and the root velocity, all in the root's heading frame.
- A critically damped command model predicts the trajectory the commands ask for. Every ``search_interval`` frames,
  when the command changes, or at the end of the clip, the frame whose features are closest to (current pose, desired
  trajectory) is found and playback jumps there.
- Jumps are hidden by inertialization: the difference between the old and new pose decays to zero with a critically
  damped spring, so the output pose and velocity stay continuous.
- The root moves by integrating the clip's per-frame root velocity in its heading frame, so playback without jumps
  reproduces the clip exactly.

Body poses and velocities come from forward kinematics on the robot model, in the same layout as the training npz.
"""

from __future__ import annotations

import mujoco
import numpy as np
from dataclasses import dataclass

LN2 = float(np.log(2.0))

##
# Rotation helpers (quaternions are w, x, y, z).
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


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    zeros = np.zeros_like(yaw)
    return np.stack([np.cos(yaw / 2), zeros, zeros, np.sin(yaw / 2)], axis=-1)


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    scale = np.where(s > 1e-9, angle / np.maximum(s, 1e-12), 2.0)
    return q[..., 1:] * scale


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    scale = np.where(angle > 1e-9, np.sin(angle / 2) / np.maximum(angle, 1e-12), 0.5)
    return np.concatenate([np.cos(angle / 2), r * scale], axis=-1)


def rotate2d(angle: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 2D vectors ``v`` (..., 2) by ``angle`` (...)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)


def decay_spring(x: np.ndarray, v: np.ndarray, halflife: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact step of a critically damped spring pulling offset ``x`` (velocity ``v``) to zero."""
    y = 2.0 * LN2 / halflife
    j1 = v + x * y
    eydt = np.exp(-y * dt)
    return eydt * (x + j1 * dt), eydt * (v - j1 * y * dt)


##
# Matcher.
##


@dataclass(frozen=True)
class MotionMatcherCfg:
    trajectory_times: tuple[float, ...] = (0.4, 0.8, 1.2)
    """Future times (s) of the trajectory features."""
    search_interval: float = 0.2
    """Time (s) between searches. A command change or the end of the clip also triggers a search."""
    min_jump: float = 0.4
    """Frames within this time (s) of the current frame are not considered as jump targets."""
    blend_halflife: float = 0.1
    """Half-life (s) of the inertialization offset after a jump."""
    command_halflife: float = 0.25
    """Half-life (s) of the predicted velocity and yaw rate converging to the command."""
    weight_foot_pos: float = 0.75
    weight_foot_vel: float = 1.0
    weight_root_vel: float = 1.0
    weight_trajectory_pos: float = 1.0
    weight_trajectory_dir: float = 1.5
    max_forward_speed: float = 1.0
    """Commands are clipped to the range the clip covers."""
    max_backward_speed: float = 0.3
    max_yaw_rate: float = 1.5


class MotionMatcher:
    """Generates a reference motion frame by frame from (forward speed, yaw rate) commands.

    ``reset`` and ``step`` return the reference frame as a dict with ``joint_pos`` and ``joint_vel`` (J,) in the npz's
    joint order and ``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w`` (B, 3 or 4) in its body
    order, which is the robot entity's (the first body is the floating base).
    """

    def __init__(
        self,
        motion_file: str,
        model: mujoco.MjModel,
        foot_body_names: tuple[str, str],
        cfg: MotionMatcherCfg = MotionMatcherCfg(),
        entity: str = "robot",
    ):
        self.cfg = cfg
        self.model = model
        self.data = mujoco.MjData(model)
        motion = np.load(motion_file)
        self.fps = float(motion["fps"][0])
        self.dt = 1.0 / self.fps
        self.joint_pos = motion["joint_pos"].astype(np.float64)
        self.joint_vel = motion["joint_vel"].astype(np.float64)
        body_pos = motion["body_pos_w"].astype(np.float64)
        body_quat = motion["body_quat_w"].astype(np.float64)
        body_lin_vel = motion["body_lin_vel_w"].astype(np.float64)
        self.num_frames = len(self.joint_pos)

        # The npz stores every body and joint of the entity, in model order.
        prefix = f"{entity}/"
        self.body_names = [
            model.body(i).name[len(prefix) :] for i in range(model.nbody) if model.body(i).name.startswith(prefix)
        ]
        self.body_ids = np.array([model.body(prefix + name).id for name in self.body_names])
        joints = [
            j
            for j in range(model.njnt)
            if model.joint(j).name.startswith(prefix) and model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.joint_names = [model.joint(j).name[len(prefix) :] for j in joints]
        self.qpos_adr = model.jnt_qposadr[joints]
        self.dof_adr = model.jnt_dofadr[joints]
        free = next(
            j
            for j in range(model.njnt)
            if model.joint(j).name.startswith(prefix) and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        )
        self.root_qpos_adr = model.jnt_qposadr[free]
        self.root_dof_adr = model.jnt_dofadr[free]
        if body_pos.shape[1] != len(self.body_names) or self.joint_pos.shape[1] != len(self.joint_names):
            raise ValueError(
                f"{motion_file} has {body_pos.shape[1]} bodies and {self.joint_pos.shape[1]} joints; the model has"
                f" {len(self.body_names)} and {len(self.joint_names)}."
            )
        if model.jnt_bodyid[free] != self.body_ids[0]:
            raise ValueError("The first body of the motion must be the floating base.")

        # Root channels. The root velocity of frame t moves the root from frame t-1 to t, in frame t-1's heading.
        root_pos, root_quat = body_pos[:, 0], body_quat[:, 0]
        self.root_xy = root_pos[:, :2]
        self.root_yaw = np.unwrap(yaw_from_quat(root_quat))
        heading = quat_from_yaw(self.root_yaw)
        self.root_z = root_pos[:, 2]
        self.root_vz = np.gradient(self.root_z, self.dt)
        self.root_tilt = quat_mul(quat_conj(heading), root_quat)
        self.root_tilt /= np.linalg.norm(self.root_tilt, axis=-1, keepdims=True)
        local_vel = rotate2d(-self.root_yaw[:-1], np.diff(self.root_xy, axis=0)) / self.dt
        self.root_vel = np.concatenate([local_vel[:1], local_vel])
        yaw_rate = np.diff(self.root_yaw) / self.dt
        self.root_yaw_rate = np.concatenate([yaw_rate[:1], yaw_rate])
        # Tilt angular velocity in the heading frame.
        tilt_vel = quat_to_rotvec(quat_mul(self.root_tilt[1:], quat_conj(self.root_tilt[:-1]))) / self.dt
        self.root_tilt_vel = np.concatenate([tilt_vel, tilt_vel[-1:]])

        # Features.
        self.trajectory_frames = np.array([int(round(t * self.fps)) for t in cfg.trajectory_times])
        horizon = int(self.trajectory_frames.max())
        if self.num_frames <= horizon + 1:
            raise ValueError(f"{motion_file} is shorter than the trajectory horizon.")
        # Frames with the whole trajectory horizon inside the clip.
        self.num_valid = self.num_frames - horizon
        valid = np.arange(self.num_valid)
        foot_ids = [self.body_names.index(name) for name in foot_body_names]
        inv_heading = quat_conj(heading[valid])[:, None]
        root_offset = np.concatenate([self.root_xy[valid], np.zeros((self.num_valid, 1))], axis=-1)[:, None]
        foot_pos = quat_apply(inv_heading, body_pos[valid][:, foot_ids] - root_offset).reshape(self.num_valid, -1)
        foot_vel = quat_apply(inv_heading, body_lin_vel[valid][:, foot_ids]).reshape(self.num_valid, -1)
        root_vel = np.concatenate([self.root_vel[valid], self.root_yaw_rate[valid, None]], axis=-1)
        traj_pos = np.concatenate(
            [
                rotate2d(-self.root_yaw[valid], self.root_xy[valid + k] - self.root_xy[valid])
                for k in self.trajectory_frames
            ],
            axis=-1,
        )
        traj_dir = np.concatenate(
            [
                np.stack([np.cos(d), np.sin(d)], axis=-1)
                for d in (self.root_yaw[valid + k] - self.root_yaw[valid] for k in self.trajectory_frames)
            ],
            axis=-1,
        )
        groups = [
            (foot_pos, cfg.weight_foot_pos),
            (foot_vel, cfg.weight_foot_vel),
            (root_vel, cfg.weight_root_vel),
            (traj_pos, cfg.weight_trajectory_pos),
            (traj_dir, cfg.weight_trajectory_dir),
        ]
        features = np.concatenate([g for g, _ in groups], axis=-1)
        self.feature_mean = features.mean(axis=0)
        # Each group is scaled by its weight over its average standard deviation, so groups count by weight only.
        self.feature_scale = np.concatenate(
            [np.full(g.shape[1], w / max(g.std(axis=0).mean(), 1e-6)) for g, w in groups]
        )
        self.features = (features - self.feature_mean) * self.feature_scale
        self.num_pose_features = foot_pos.shape[1] + foot_vel.shape[1] + root_vel.shape[1]

        self.search_interval = max(1, int(round(cfg.search_interval * self.fps)))
        self.min_jump = int(round(cfg.min_jump * self.fps))
        self.num_jumps = 0

    # Commands.

    def clip_command(self, forward_speed: float, yaw_rate: float) -> tuple[float, float]:
        c = self.cfg
        return (
            float(np.clip(forward_speed, -c.max_backward_speed, c.max_forward_speed)),
            float(np.clip(yaw_rate, -c.max_yaw_rate, c.max_yaw_rate)),
        )

    def _advance_command_model(
        self, vel: np.ndarray, yaw_rate: float, yaw: float, command: tuple[float, float]
    ) -> tuple[np.ndarray, float, float]:
        """One step of the command model: world-frame velocity and yaw rate converge to the command."""
        alpha = 1.0 - np.exp(-LN2 * self.dt / self.cfg.command_halflife)
        yaw_rate += (command[1] - yaw_rate) * alpha
        yaw += yaw_rate * self.dt
        target = rotate2d(np.array(yaw), np.array([command[0], 0.0]))
        vel = vel + (target - vel) * alpha
        return vel, yaw_rate, yaw

    def desired_trajectory(self, command: tuple[float, float]) -> np.ndarray:
        """Trajectory features predicted by the command model from the current state, in the root's heading frame."""
        vel, yaw_rate, yaw = self.cmd_vel.copy(), self.cmd_yaw_rate, self.yaw
        pos = np.zeros(2)
        positions, directions = [], []
        for k in range(1, int(self.trajectory_frames.max()) + 1):
            vel, yaw_rate, yaw = self._advance_command_model(vel, yaw_rate, yaw, command)
            pos = pos + vel * self.dt
            if k in self.trajectory_frames:
                positions.append(rotate2d(np.array(-self.yaw), pos))
                directions.append([np.cos(yaw - self.yaw), np.sin(yaw - self.yaw)])
        return np.concatenate([np.concatenate(positions), np.concatenate(directions)])

    # Playback.

    def reset(self, frame: int = 0) -> dict[str, np.ndarray]:
        """Start at ``frame`` of the clip with the clip's root pose."""
        if not 0 <= frame < self.num_valid:
            raise ValueError(f"Start frame must be in [0, {self.num_valid}).")
        self.frame = frame
        self.xy = self.root_xy[frame].copy()
        self.yaw = float(self.root_yaw[frame])
        # Offsets live in the rate space (rotation vectors for the tilt).
        self.offsets = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in self._clip_velocities(frame).items()}
        self.channels = self._blended_channels()
        self.cmd_vel = rotate2d(np.array(self.yaw), self.root_vel[frame])
        self.cmd_yaw_rate = float(self.root_yaw_rate[frame])
        self.command = (0.0, 0.0)
        self.frames_since_search = 0
        self.num_jumps = 0
        return self._forward_kinematics()

    def step(self, forward_speed: float, yaw_rate: float) -> dict[str, np.ndarray]:
        """Advance one clip frame (1 / fps s) under the command and return the new reference frame."""
        command = self.clip_command(forward_speed, yaw_rate)
        self.cmd_vel, self.cmd_yaw_rate, _ = self._advance_command_model(
            self.cmd_vel, self.cmd_yaw_rate, self.yaw, command
        )

        # Continue playback, then search: a jump is blended from the pose playback would have shown this frame.
        self.frame += 1
        self.frames_since_search += 1
        for name, (x, v) in self.offsets.items():
            self.offsets[name] = decay_spring(x, v, self.cfg.blend_halflife, self.dt)
        current = self._blended_channels()
        if self.frame >= self.num_valid or self.frames_since_search >= self.search_interval or command != self.command:
            self.frames_since_search = 0
            best = self.search(command)
            if best is not None:
                self._jump(best, current)
                current = self._blended_channels()
        self.command = command

        # Integrate the root in its heading frame, like the clip's own root motion.
        self.xy = self.xy + rotate2d(np.array(self.yaw), current["root_vel"]) * self.dt
        self.yaw += float(current["root_yaw_rate"][0]) * self.dt
        self.channels = current
        return self._forward_kinematics()

    def query(self, command: tuple[float, float]) -> np.ndarray:
        frame = min(self.frame, self.num_valid - 1)
        pose = self.features[frame, : self.num_pose_features]
        trajectory = (
            self.desired_trajectory(command) - self.feature_mean[self.num_pose_features :]
        ) * self.feature_scale[self.num_pose_features :]
        return np.concatenate([pose, trajectory])

    def search(self, command: tuple[float, float]) -> int | None:
        """The frame to jump to, or None to keep playing."""
        costs = np.sum((self.features - self.query(command)) ** 2, axis=-1)
        current_cost = np.inf
        if self.frame < self.num_valid:
            current_cost = costs[self.frame]
            costs[max(0, self.frame - self.min_jump) : self.frame + self.min_jump + 1] = np.inf
        best = int(np.argmin(costs))
        return best if costs[best] < current_cost else None

    def _clip_channels(self, frame: int) -> dict[str, np.ndarray]:
        return {
            "joint_pos": self.joint_pos[frame],
            "root_z": self.root_z[frame : frame + 1],
            "root_tilt": self.root_tilt[frame],
            "root_vel": self.root_vel[frame],
            "root_yaw_rate": self.root_yaw_rate[frame : frame + 1],
        }

    def _clip_velocities(self, frame: int) -> dict[str, np.ndarray]:
        """Rates of the clip channels; the root velocities are blended as values, without rates."""
        return {
            "joint_pos": self.joint_vel[frame],
            "root_z": self.root_vz[frame : frame + 1],
            "root_tilt": self.root_tilt_vel[frame],
            "root_vel": np.zeros(2),
            "root_yaw_rate": np.zeros(1),
        }

    def _blended_channels(self) -> dict[str, np.ndarray]:
        clip = self._clip_channels(self.frame)
        rates = self._clip_velocities(self.frame)
        out = {}
        for name, value in clip.items():
            x, v = self.offsets[name]
            if name == "root_tilt":
                out[name] = quat_mul(rotvec_to_quat(x), value)
            else:
                out[name] = value + x
            out[name + "_rate"] = rates[name] + v
        return out

    def _jump(self, frame: int, current: dict[str, np.ndarray]) -> None:
        """Jump to ``frame``, setting offsets so the output continues from ``current``."""
        clip = self._clip_channels(frame)
        rates = self._clip_velocities(frame)
        for name, value in clip.items():
            if name == "root_tilt":
                x = quat_to_rotvec(quat_mul(current[name], quat_conj(value)))
            else:
                x = current[name] - value
            self.offsets[name] = (x, current[name + "_rate"] - rates[name])
        self.frame = frame
        self.num_jumps += 1

    def _forward_kinematics(self) -> dict[str, np.ndarray]:
        c = self.channels
        m, d = self.model, self.data
        heading = quat_from_yaw(self.yaw)
        root_quat = quat_mul(heading, c["root_tilt"])
        root_quat /= np.linalg.norm(root_quat)
        root_vel = rotate2d(np.array(self.yaw), c["root_vel"])
        ang_vel_w = quat_apply(heading, c["root_tilt_rate"]) + np.array([0.0, 0.0, c["root_yaw_rate"][0]])

        d.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = [*self.xy, c["root_z"][0]]
        d.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = root_quat
        d.qpos[self.qpos_adr] = c["joint_pos"]
        d.qvel[self.root_dof_adr : self.root_dof_adr + 3] = [*root_vel, c["root_z_rate"][0]]
        # Free joint angular velocity is in the body frame.
        d.qvel[self.root_dof_adr + 3 : self.root_dof_adr + 6] = quat_apply(quat_conj(root_quat), ang_vel_w)
        d.qvel[self.dof_adr] = c["joint_pos_rate"]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)

        vel = np.zeros((len(self.body_ids), 6))
        for i, body_id in enumerate(self.body_ids):
            # XBODY: velocity of the body frame origin (mjlab's body_link_*_vel_w), world orientation.
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, body_id, vel[i], 0)
        return {
            "joint_pos": c["joint_pos"].copy(),
            "joint_vel": c["joint_pos_rate"].copy(),
            "body_pos_w": d.xpos[self.body_ids].copy(),
            "body_quat_w": d.xquat[self.body_ids].copy(),
            "body_lin_vel_w": vel[:, 3:],
            "body_ang_vel_w": vel[:, :3],
        }
