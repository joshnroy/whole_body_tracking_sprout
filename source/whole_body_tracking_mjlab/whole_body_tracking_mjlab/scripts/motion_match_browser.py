"""Browser steering of an ONNX tracking policy with Viser and CPU MuJoCo."""

from __future__ import annotations

import numpy as np
import time
from contextlib import suppress
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue

import mjlab
import tyro
from mjlab.tasks.registry import list_tasks
from whole_body_tracking_mjlab.motion_matching import MotionMatcher, MotionMatcherCfg, yaw_from_quat
from whole_body_tracking_mjlab.scripts.browser_camera import ChaseCamera
from whole_body_tracking_mjlab.scripts.browser_keyboard import KeyboardBridge
from whole_body_tracking_mjlab.scripts.motion_match import FOOT_BODY_PATTERN
from whole_body_tracking_mjlab.scripts.sim2sim import Sim2Sim, Sim2SimConfig


@dataclass(frozen=True)
class BrowserConfig:
    onnx_file: str
    """Exported tracking policy, including reference motion and observation metadata."""
    motion_file: str | None = None
    """Optional training npz. Defaults to the motion embedded in the ONNX at the task's policy rate."""
    host: str = "127.0.0.1"
    """Listen address. Bind a private network address for remote access, or use an SSH tunnel."""
    port: int = 8080
    """HTTP and WebSocket port for the browser viewer."""
    keyboard_port: int | None = None
    """Keyboard WebSocket port. Defaults to the viewer port plus one; expose both for remote access."""
    start_frame: int = 0
    foot_body_names: tuple[str, str] | None = None
    matcher: MotionMatcherCfg = field(default_factory=lambda: MotionMatcherCfg(command_heading=True))
    actuator_delay: int | None = None
    obs_delay: int | None = None


class MotionMatchSession:
    """Simulation state owned exclusively by the browser viewer's main loop."""

    def __init__(self, task_id: str, cfg: BrowserConfig):
        import re

        self.cfg = cfg
        self.sim = Sim2Sim(
            task_id, Sim2SimConfig(onnx_file=cfg.onnx_file, actuator_delay=cfg.actuator_delay, obs_delay=cfg.obs_delay)
        )
        model = self.sim.model
        feet = cfg.foot_body_names
        if feet is None:
            names = [model.body(i).name.removeprefix("robot/") for i in range(model.nbody)]
            feet = tuple(sorted(n for n in names if re.match(FOOT_BODY_PATTERN, n)))
        if len(feet) != 2:
            raise ValueError(f"Found feet {feet}; set --foot-body-names.")
        if cfg.motion_file:
            self.matcher = MotionMatcher(cfg.motion_file, model, feet, cfg.matcher)
        else:
            self.matcher = MotionMatcher.from_onnx(cfg.onnx_file, model, feet, 1 / self.sim.step_dt, cfg.matcher)
        if not np.isclose(self.matcher.dt, self.sim.step_dt, rtol=0, atol=1e-6):
            raise ValueError("Motion frame rate does not match the task's policy rate.")
        self.joint_idx = [self.matcher.joint_names.index(n) for n in self.sim.policy.joint_names]
        self.body_idx = [self.matcher.body_names.index(n) for n in self.sim.policy.body_names]
        self.reset()

    def _select(self, ref: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: v[self.joint_idx] if k.startswith("joint") else v[self.body_idx] for k, v in ref.items()}

    def reset(self) -> None:
        self.ref = self._select(self.matcher.reset(self.cfg.start_frame))
        self.sim.reset_to(self.ref)
        self.command = (0.0, 0.0, 0.0)
        self.paused = True
        self.termination: list[str] = []
        self.steps = 0

    def control(self, action: str, value: float = 0.0) -> None:
        """Apply queued UI input on the simulation thread; never change MuJoCo from callbacks."""
        vx, vy, wz = self.command
        if action == "reset":
            self.reset()
        elif action == "pause":
            self.paused = True
        elif action == "toggle_pause":
            if not self.termination:
                self.paused = not self.paused
        elif action == "stop":
            self.command = (0.0, 0.0, 0.0)
        elif action in ("forward", "turn", "set_forward", "set_lateral", "set_turn") and np.isfinite(value):
            if action == "forward":
                vx += value
            elif action == "turn":
                wz += value
            elif action == "set_forward":
                vx = value
            elif action == "set_lateral":
                vy = value
            else:
                wz = value
            vx, wz = self.matcher.clip_command(round(vx, 2), round(wz, 2))
            vy = float(np.clip(round(vy, 2), -self.cfg.matcher.max_lateral_speed, self.cfg.matcher.max_lateral_speed))
            self.command = (vx, vy, wz)

    def step(self) -> None:
        if self.paused:
            return
        self.sim.step(self.sim.observation(self.ref), 0)
        _, self.termination = self.sim.evaluate(self.ref)
        self.steps += 1
        if self.termination:
            self.paused = True
            self.command = (0.0, 0.0, 0.0)
        else:
            self.ref = self._select(self.matcher.step(self.command[0], self.command[2], lateral_speed=self.command[1]))


class BrowserViewer:
    """A shared browser session, paused initially and when its last client disconnects."""

    def __init__(self, session: MotionMatchSession):
        import viser
        from mjviser import ViserMujocoScene

        self.session = session
        self.events: SimpleQueue = SimpleQueue()
        self.keyboard_owner = None
        self.keyboard_deadline = 0.0
        self.keyboard = None
        self.camera = ChaseCamera()
        self._camera_updated = time.monotonic()
        self._display_command: tuple[float, float, float] | None = None
        cfg = session.cfg
        self.server = viser.ViserServer(host=cfg.host, port=cfg.port, label="Motion control")
        if self.server.get_port() != cfg.port:
            self.server.stop()
            raise RuntimeError(f"Port {cfg.port} is occupied; choose another --port.")
        try:
            self.server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False)
            self.scene = ViserMujocoScene(self.server, session.sim.model, num_envs=1)
            self.server.gui.add_markdown(
                "## Motion control\n"
                "Click the 3D scene, then press **P** to play.\n\n"
                "Hold **W / ↑** to walk forward, **S / ↓** to walk backward.\n\n"
                "Hold **A / ←** to strafe left, **D / →** to strafe right.\n\n"
                "Hold **Q / E** to turn left / right. Release to stop commanding movement.\n\n"
                "The camera follows smoothly behind Sprout. Mouse movement does not affect the robot or camera.\n\n"
                "**Space**: stop commands · **P**: play/pause · **R**: reset\n\n"
                "Keys control the robot only; input boxes keep normal typing behavior.\n\n"
                "Motion comes from the recorded clip. Requested velocities are approximate; "
                "sideways motion may be weak or inconsistent with a walking-only policy."
            )
            self.status = self.server.gui.add_markdown("")
            with self.server.gui.add_folder("Drive"):
                self.forward = self.server.gui.add_slider(
                    "Forward speed (m/s)",
                    min=-cfg.matcher.max_backward_speed,
                    max=cfg.matcher.max_forward_speed,
                    step=0.1,
                    initial_value=0.0,
                )
                self.lateral = self.server.gui.add_slider(
                    "Strafe (m/s, +left)",
                    min=-cfg.matcher.max_lateral_speed,
                    max=cfg.matcher.max_lateral_speed,
                    step=0.05,
                    initial_value=0.0,
                )
                self.turn = self.server.gui.add_slider(
                    "Turn rate (rad/s)",
                    min=-cfg.matcher.max_yaw_rate,
                    max=cfg.matcher.max_yaw_rate,
                    step=0.05,
                    initial_value=0.0,
                )
                self.forward.on_update(self._slider_callback("set_forward"))
                self.lateral.on_update(self._slider_callback("set_lateral"))
                self.turn.on_update(self._slider_callback("set_turn"))
                for label, action, value in (
                    ("Forward ↑", "forward", 0.1),
                    ("Backward ↓", "forward", -0.1),
                    ("Turn left (Q)", "turn", 0.25),
                    ("Turn right (E)", "turn", -0.25),
                    ("Stop commands", "stop", 0.0),
                ):
                    self._button(label, action, value)
            self.play_button = self._button("Play", "toggle_pause", 0.0)
            self._button("Reset", "reset", 0.0)
            # Keep the world fixed: recentering geometry on every torso update
            # would reintroduce gait bob even with a damped camera.
            self.scene.camera_tracking_enabled = False

            self.keyboard = KeyboardBridge(self.server, self.events, cfg.host, cfg.keyboard_port or cfg.port + 1)

            @self.server.on_client_disconnect
            def _(_client):
                if not self.server.get_clients():
                    self.events.put(("pause", 0.0))

            self.render()
        except BaseException:
            self.server.stop()
            raise

    def _slider_callback(self, action):
        async def callback(event):
            # Updating GUI values from the loop must not enqueue stale commands.
            # Capture the value on Viser's event loop, before a later update can
            # change the handle; the simulation itself still runs on one thread.
            if event.client is not None:
                self.events.put((action, float(event.target.value)))

        return callback

    def _button(self, label, action, value):
        button = self.server.gui.add_button(label)

        async def callback(_event):
            self.events.put((action, value))

        button.on_click(callback)
        return button

    def _control(self, action, value):
        if action == "keyboard":
            owner, received, forward, lateral, turn = value
            # Ignore input that waited in the queue longer than its lease.
            if time.monotonic() - received < 0.6:
                self.keyboard_owner = owner
                self.keyboard_deadline = received + 0.6
                scale = 1 / np.sqrt(2) if forward and lateral else 1.0
                self.session.control("set_forward", forward * (0.4 if forward > 0 else 0.2) * scale)
                self.session.control("set_lateral", lateral * 0.2 * scale)
                self.session.control("set_turn", turn * 0.5)
        elif action == "release":
            if value == self.keyboard_owner:
                self.session.control("stop")
                self.keyboard_owner = None
        else:
            if self.keyboard_owner is not None:
                self.session.control("stop")
            self.keyboard_owner = None
            self.session.control(action, value)
            if action == "reset":
                self.camera.reset()

    def render(self) -> None:
        s = self.session
        self.scene.update_from_mjdata(s.sim.data)
        body_id = s.matcher.body_ids[0]
        yaw = float(yaw_from_quat(s.sim.data.xquat[body_id]))
        now = time.monotonic()
        eye, look_at = self.camera.update(s.sim.data.xpos[body_id], yaw, now - self._camera_updated)
        self._camera_updated = now
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.up_direction = (0.0, 0.0, 1.0)
                # Viser translates look_at when position changes; set the
                # target last so both commands describe the same camera pose.
                client.camera.position = eye
                client.camera.look_at = look_at
        # Do not overwrite a slider edit while its callback is waiting for the
        # next simulation tick, or continually reset an in-progress numeric edit.
        if s.command != self._display_command:
            self.forward.value, self.lateral.value, self.turn.value = s.command
            self._display_command = s.command
        self.play_button.label = "Play" if s.paused else "Pause"
        self.play_button.disabled = bool(s.termination)
        state = "Paused" if s.paused else "Running"
        if s.termination:
            state = "Stopped — reset to continue (" + ", ".join(s.termination) + ")"
        self.status.content = (
            f"**{state}** · {s.steps * s.sim.step_dt:.2f} s\n\n"
            f"Forward **{s.command[0]:+.2f} m/s** · Strafe **{s.command[1]:+.2f} m/s**\n\n"
            f"Turn **{s.command[2]:+.2f} rad/s**\n\n"
            f"Clip frame {s.matcher.frame} · {s.matcher.num_jumps} motion transitions"
        )

    def run(self) -> None:
        """Bounded-memory real-time loop. Only this thread touches policy or physics state."""
        s = self.session
        print(f"[INFO] Browser: http://{s.cfg.host}:{self.server.get_port()} (starts paused)", flush=True)
        try:
            while True:
                started = time.monotonic()
                while True:
                    try:
                        self._control(*self.events.get_nowait())
                    except Empty:
                        break
                if self.keyboard_owner is not None and time.monotonic() >= self.keyboard_deadline:
                    self._control("release", self.keyboard_owner)
                if not self.server.get_clients():
                    s.control("stop")
                    s.control("pause")
                s.step()
                self.render()
                time.sleep(max(0.0, s.sim.step_dt - (time.monotonic() - started)))
        finally:
            if self.keyboard is not None:
                self.keyboard.stop()
            self.server.stop()


def main():
    task_id, args = tyro.cli(
        tyro.extras.literal_type_from_choices([t for t in list_tasks() if "Tracking" in t]),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(BrowserConfig, args=args, prog=f"wbt-motion-match-browser {task_id}", config=mjlab.TYRO_FLAGS)
    with suppress(KeyboardInterrupt):
        BrowserViewer(MotionMatchSession(task_id, cfg)).run()


if __name__ == "__main__":
    main()
