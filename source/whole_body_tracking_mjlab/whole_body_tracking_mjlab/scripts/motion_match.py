"""Steer a tracking policy with forward/backward speed and yaw rate commands by motion matching over its training clip.

A ``MotionMatcher`` built from the training npz (e.g. the walking clip) turns the commands into a reference motion,
one frame per policy step. With ``--onnx-file`` the exported tracking policy follows that reference in CPU MuJoCo, built
the same way as ``wbt-sim2sim`` (no randomization). Without it the reference is played back kinematically.

Commands come from ``--commands`` (a schedule) or, in the ``--viewer`` without a schedule, the keyboard:
up/down change the forward speed by 0.1 m/s, left/right the yaw rate by 0.25 rad/s, space stops.

    wbt-motion-match Tracking-Flat-Sprout-Wo-State-Estimation-v0 --motion-file walk1_subject1.npz \
      --onnx-file logs/rsl_rl/sprout_flat/{run}/{run}.onnx --viewer
    wbt-motion-match ... --commands "0,0,0 2,0.6,0 8,0.4,0.8 14,-0.25,0 20,0,0" --duration 24 --video-file out.mp4
"""

from __future__ import annotations

import json
import mujoco
import numpy as np
import re
import time
import tyro
from dataclasses import dataclass, field
from pathlib import Path

import mjlab
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg

import whole_body_tracking_mjlab  # noqa: F401
from whole_body_tracking_mjlab.motion_matching import MotionMatcher, MotionMatcherCfg, yaw_from_quat
from whole_body_tracking_mjlab.scripts.sim2sim import Sim2Sim, Sim2SimConfig, _add_reference_markers

ENTITY = "robot"
FOOT_BODY_PATTERN = r"^(left|right)_(foot|ankle_roll)_link$"
# GLFW key codes.
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_SPACE = 262, 263, 264, 265, 32


@dataclass(frozen=True)
class MotionMatchConfig:
    motion_file: str
    """Training npz used as the motion database (e.g. the walking clip the policy was trained on)."""
    onnx_file: str | None = None
    """Exported tracking policy to drive with the matched reference. Without it the reference is played kinematically."""
    commands: str = ""
    """Command schedule: space-separated ``time,forward_speed,yaw_rate`` entries (s, m/s, rad/s), each held until the
    next, e.g. ``"0,0,0 2,0.6,0 8,0,1"``. Without one,
    ``--viewer`` takes commands from the keyboard."""
    duration: float | None = None
    """Rollout length (s). Defaults to 3 s past the last scheduled command; keyboard control runs until the viewer
    closes."""
    start_frame: int = 0
    """Clip frame to start from."""
    foot_body_names: tuple[str, str] | None = None
    """Feet for the matching features. Defaults to the bodies matching ``(left|right)_(foot|ankle_roll)_link``."""
    matcher: MotionMatcherCfg = field(default_factory=MotionMatcherCfg)
    actuator_delay: int | None = None
    """As in ``wbt-sim2sim``."""
    obs_delay: int | None = None
    """As in ``wbt-sim2sim``."""
    stop_on_termination: bool = True
    """End the rollout when a training termination fires."""
    settle_time: float = 1.0
    """Time (s) after each command change left out of the per-command velocity averages."""
    viewer: bool = False
    """Show the rollout in the MuJoCo viewer in real time."""
    video_file: str | None = None
    """Render the rollout to this mp4 (offscreen; set ``MUJOCO_GL=egl`` on a headless machine)."""
    output_file: str | None = None
    """Write the generated reference to this npz, in the training npz format plus the per-frame ``command``."""
    metrics_file: str | None = None
    """Write the metrics to this JSON file."""


def parse_schedule(entries: str) -> list[tuple[float, float, float]]:
    schedule = []
    for entry in entries.split():
        parts = [float(x) for x in entry.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Command {entry!r} is not time,forward_speed,yaw_rate.")
        schedule.append(tuple(parts))
    schedule.sort()
    return schedule


class Keyboard:
    """Arrow-key command input for the passive viewer."""

    def __init__(self, matcher: MotionMatcher):
        self.matcher = matcher
        self.command = (0.0, 0.0)

    def __call__(self, key: int) -> None:
        vx, wz = self.command
        if key == KEY_UP:
            vx += 0.1
        elif key == KEY_DOWN:
            vx -= 0.1
        elif key == KEY_LEFT:
            wz += 0.25
        elif key == KEY_RIGHT:
            wz -= 0.25
        elif key == KEY_SPACE:
            vx, wz = 0.0, 0.0
        else:
            return
        self.command = self.matcher.clip_command(round(vx, 2), round(wz, 2))
        print(f"[INFO] Command: forward {self.command[0]:+.2f} m/s, yaw rate {self.command[1]:+.2f} rad/s")


def heading_velocities(xy: np.ndarray, yaw: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward speed and yaw rate of a root trajectory, per step (the first step repeats the second)."""
    yaw = np.unwrap(yaw)
    v = np.diff(xy, axis=0) / dt
    forward = np.cos(yaw[:-1]) * v[:, 0] + np.sin(yaw[:-1]) * v[:, 1]
    yaw_rate = np.diff(yaw) / dt
    return np.concatenate([forward[:1], forward]), np.concatenate([yaw_rate[:1], yaw_rate])


def command_segments(
    commands: np.ndarray, tracks: dict[str, tuple[np.ndarray, np.ndarray]], dt: float, settle_time: float
) -> list[dict]:
    """Mean forward speed and yaw rate of each track over each constant-command segment, after ``settle_time``."""
    change = np.flatnonzero(np.any(np.diff(commands, axis=0) != 0, axis=-1)) + 1
    bounds = [0, *change.tolist(), len(commands)]
    settle = int(round(settle_time / dt))
    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end - start - settle < 1:
            continue
        seg = {
            "start_s": start * dt,
            "end_s": end * dt,
            "command_forward": float(commands[start, 0]),
            "command_yaw_rate": float(commands[start, 1]),
        }
        for name, (forward, yaw_rate) in tracks.items():
            seg[f"{name}_forward"] = float(forward[start + settle : end].mean())
            seg[f"{name}_yaw_rate"] = float(yaw_rate[start + settle : end].mean())
        segments.append(seg)
    return segments


def _configure_camera(cam: mujoco.MjvCamera, env_cfg, body_id: int) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body_id
    cam.distance = env_cfg.viewer.distance
    cam.azimuth = env_cfg.viewer.azimuth
    cam.elevation = env_cfg.viewer.elevation


def run_motion_match(task_id: str, cfg: MotionMatchConfig) -> dict:
    schedule = parse_schedule(cfg.commands)
    if not schedule and not cfg.viewer:
        raise ValueError("Give a --commands schedule, or use --viewer for keyboard control.")

    sim = None
    if cfg.onnx_file:
        sim = Sim2Sim(
            task_id, Sim2SimConfig(onnx_file=cfg.onnx_file, actuator_delay=cfg.actuator_delay, obs_delay=cfg.obs_delay)
        )
        env_cfg, model, dt = sim.env_cfg, sim.model, sim.step_dt
    else:
        env_cfg = load_env_cfg(task_id, play=True)
        env_cfg.scene.num_envs = 1
        model = Scene(env_cfg.scene, device="cpu").compile()
        env_cfg.sim.mujoco.apply(model)
        dt = model.opt.timestep * env_cfg.decimation
    data = mujoco.MjData(model)

    foot_body_names = cfg.foot_body_names
    if foot_body_names is None:
        names = [model.body(i).name.removeprefix(f"{ENTITY}/") for i in range(model.nbody)]
        foot_body_names = tuple(sorted(n for n in names if re.match(FOOT_BODY_PATTERN, n)))
        if len(foot_body_names) != 2:
            raise ValueError(f"Found feet {foot_body_names}; set --foot-body-names.")
    matcher = MotionMatcher(cfg.motion_file, model, foot_body_names, cfg.matcher, entity=ENTITY)
    if abs(matcher.dt - dt) > 1e-6:
        raise ValueError(f"{cfg.motion_file} is at {matcher.fps:g} fps; the policy runs at {1 / dt:g} Hz.")
    root_body_id = int(matcher.body_ids[0])

    select = None
    if sim is not None:
        p = sim.policy
        joint_idx = [matcher.joint_names.index(n) for n in p.joint_names]
        body_idx = [matcher.body_names.index(n) for n in p.body_names]

        def select(ref: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            """Reference in the policy's joint and body order."""
            return {k: v[joint_idx] if k.startswith("joint") else v[body_idx] for k, v in ref.items()}

    if schedule:
        duration = cfg.duration if cfg.duration is not None else schedule[-1][0] + 3.0
        num_steps = int(round(duration / dt))
        times = np.array([s[0] for s in schedule])

        def command_at(t: float) -> tuple[float, float]:
            i = np.searchsorted(times, t + 1e-9, side="right") - 1
            return matcher.clip_command(*schedule[max(i, 0)][1:])

    else:
        keyboard = Keyboard(matcher)
        num_steps = int(round(cfg.duration / dt)) if cfg.duration is not None else None

        def command_at(t: float) -> tuple[float, float]:
            return keyboard.command

    print(f"[INFO] Task {task_id}, motion {cfg.motion_file} ({matcher.num_frames} frames, feet {foot_body_names})")
    print(f"[INFO] {'Policy ' + cfg.onnx_file if sim else 'Kinematic playback'}, dt {dt:.3f} s")
    if not schedule:
        print("[INFO] Keyboard: up/down forward speed, left/right yaw rate, space stops.")

    viewer = None
    if cfg.viewer:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(
            model,
            sim.data if sim else data,
            key_callback=None if schedule else keyboard,
            show_left_ui=False,
            show_right_ui=False,
        )
        _configure_camera(viewer.cam, env_cfg, root_body_id)
    renderer, camera, video = None, None, None
    if cfg.video_file:
        import mediapy

        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        _configure_camera(camera, env_cfg, root_body_id)
        Path(cfg.video_file).parent.mkdir(parents=True, exist_ok=True)
        video = mediapy.VideoWriter(cfg.video_file, shape=(480, 640), fps=1.0 / dt)
        video.__enter__()

    ref = matcher.reset(cfg.start_frame)
    if sim is not None:
        sim.reset_to(select(ref))
    log: dict[str, list] = {k: [] for k in ref}
    log.update(command=[], robot_xy=[], robot_yaw=[])
    sums: dict[str, float] = {}
    termination, steps = None, 0
    wall_start = time.perf_counter()

    while num_steps is None or steps < num_steps:
        step_start = time.perf_counter()
        command = command_at(steps * dt)
        for k, v in ref.items():
            log[k].append(v)
        log["command"].append(command)

        fired = []
        if sim is not None:
            # The policy is commanded with this frame; the post-step state is scored against it.
            policy_ref = select(ref)
            sim.step(sim.observation(policy_ref), 0)
            errors, fired = sim.evaluate(policy_ref)
            for k, v in errors.items():
                sums[k] = sums.get(k, 0.0) + v
            log["robot_xy"].append(sim.data.xpos[root_body_id, :2].copy())
            log["robot_yaw"].append(float(yaw_from_quat(sim.data.xquat[root_body_id])))
        steps += 1

        ref = matcher.step(*command)

        if sim is None:
            data.qpos[:] = matcher.data.qpos
            mujoco.mj_forward(model, data)
        markers = sim.relative_reference(select(ref))[0] if sim is not None else None
        if renderer is not None:
            renderer.update_scene(sim.data if sim else data, camera)
            if markers is not None:
                _add_reference_markers(renderer.scene, markers)
            video.add_image(renderer.render())
        if viewer is not None:
            if not viewer.is_running():
                break
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if markers is not None:
                    _add_reference_markers(viewer.user_scn, markers)
                viewer.set_texts(
                    (
                        None,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "forward\nyaw rate\nclip frame",
                        f"{command[0]:+.2f} m/s\n{command[1]:+.2f} rad/s\n{matcher.frame}",
                    )
                )
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - step_start)))

        if fired and termination is None:
            termination = {"time_s": steps * dt, "terms": fired}
            print(f"[INFO] Terminated at {steps * dt:.2f} s: {', '.join(fired)}")
            if cfg.stop_on_termination:
                break

    wall_time = time.perf_counter() - wall_start
    if viewer is not None:
        viewer.close()

    commands = np.array(log["command"]).reshape(-1, 2)
    root_xy = np.array(log["body_pos_w"])[:, 0, :2]
    root_yaw = yaw_from_quat(np.array(log["body_quat_w"])[:, 0])
    tracks = {"reference": heading_velocities(root_xy, root_yaw, dt)}
    if sim is not None:
        tracks["robot"] = heading_velocities(np.array(log["robot_xy"]), np.array(log["robot_yaw"]), dt)
    segments = command_segments(commands, tracks, dt, cfg.settle_time) if steps > 1 else []

    metrics = {
        "task": task_id,
        "motion_file": cfg.motion_file,
        "onnx_file": cfg.onnx_file,
        "success": termination is None and (num_steps is None or steps == num_steps),
        "steps": steps,
        "duration_s": steps * dt,
        "termination": termination,
        "num_jumps": matcher.num_jumps,
        **{k: v / max(steps, 1) for k, v in sums.items()},
        "realtime_factor": steps * dt / wall_time,
    }
    print("\n" + "=" * 50)
    print("Motion Matching Results")
    print("=" * 50)
    for name, value in metrics.items():
        if name in ("task", "motion_file", "onnx_file"):
            continue
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    if segments:
        header = "  time (s)      command (m/s, rad/s)" + "".join(f"   {name:>18}" for name in tracks)
        print(header)
        for seg in segments:
            row = (
                f"  {seg['start_s']:5.1f}-{seg['end_s']:5.1f}   {seg['command_forward']:+.2f}, "
                f"{seg['command_yaw_rate']:+.2f}        "
            )
            row += "".join(f"   {seg[f'{name}_forward']:+.2f}, {seg[f'{name}_yaw_rate']:+.2f}  " for name in tracks)
            print(row)
    print("=" * 50)

    if cfg.metrics_file:
        path = Path(cfg.metrics_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**metrics, "segments": segments}, indent=2))
        print(f"[INFO] Metrics saved to {path}")
    if cfg.output_file:
        path = Path(cfg.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        motion_keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        np.savez(
            path,
            fps=[1.0 / dt],
            command=commands,
            **{k: np.array(log[k], dtype=np.float32) for k in motion_keys},
        )
        print(f"[INFO] Reference saved to {path}")
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
    cfg = tyro.cli(MotionMatchConfig, args=remaining_args, prog=f"wbt-motion-match {task_id}", config=mjlab.TYRO_FLAGS)
    run_motion_match(task_id, cfg)


if __name__ == "__main__":
    main()
