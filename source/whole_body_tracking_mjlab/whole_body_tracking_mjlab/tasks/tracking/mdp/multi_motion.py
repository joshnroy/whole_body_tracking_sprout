"""Train one tracking policy on several reference motions.

``MultiMotionCommand`` extends mjlab's ``MotionCommand``. Its ``motion_file`` may be

- one ``.npz`` file (the single-motion behavior, unchanged),
- a directory: every ``.npz`` in it, sorted by name, or
- a ``.txt`` manifest: one ``.npz`` path per line, relative to the manifest; blank lines and ``#`` comments are
  ignored.

The clips are concatenated into one reference, so every per-frame lookup of the parent command (and the ONNX export,
which bundles the reference) works unchanged on a global frame index. Each env also records which clip it is in:

- An env that reaches the end of its clip resamples, instead of running on into the next clip.
- Adaptive sampling keeps mjlab's one-second failure bins, but per clip, so no bin straddles two clips.
- ``sampling_mode="start"`` starts each env at the first frame of a randomly chosen clip.
"""

from __future__ import annotations

import dataclasses
import math
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def resolve_motion_files(motion_file: str) -> list[str]:
    """Expand ``motion_file`` (a ``.npz``, a directory of them, or a ``.txt`` manifest) into a list of ``.npz`` paths."""
    path = Path(motion_file)
    if path.is_dir():
        files = sorted(str(p) for p in path.glob("*.npz"))
        if not files:
            raise ValueError(f"Motion directory {path} contains no .npz files.")
        return files
    if path.suffix == ".txt":
        files = []
        for line in path.read_text().splitlines():
            entry = line.split("#", 1)[0].strip()
            if not entry:
                continue
            clip = Path(entry) if Path(entry).is_absolute() else path.parent / entry
            if not clip.is_file():
                raise FileNotFoundError(f"Motion manifest {path} lists {entry}, which does not exist.")
            files.append(str(clip))
        if not files:
            raise ValueError(f"Motion manifest {path} lists no .npz files.")
        return files
    return [motion_file]


class MultiMotionLoader:
    """Several motion ``.npz`` files concatenated along time, with the attributes of mjlab's ``MotionLoader``."""

    def __init__(self, motion_files: list[str], body_indexes: torch.Tensor, device: str = "cpu") -> None:
        keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
        lengths, fps_values = [], []
        for f in motion_files:
            data = np.load(f)
            if parts["joint_pos"] and data["joint_pos"].shape[1] != parts["joint_pos"][0].shape[1]:
                raise ValueError(
                    f"{f} has {data['joint_pos'].shape[1]} joints; {motion_files[0]} has"
                    f" {parts['joint_pos'][0].shape[1]}. All motions must be retargeted to the same robot."
                )
            if parts["body_pos_w"] and data["body_pos_w"].shape[1] != parts["body_pos_w"][0].shape[1]:
                raise ValueError(f"{f} and {motion_files[0]} have different body counts.")
            fps_values.append(float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else None)
            for k in keys:
                parts[k].append(data[k])
            lengths.append(data["joint_pos"].shape[0])
        if len({v for v in fps_values}) > 1:
            raise ValueError(
                f"Motions have different fps ({sorted(set(fps_values), key=str)}); resample them to one rate."
            )

        def cat(k: str) -> torch.Tensor:
            return torch.tensor(np.concatenate(parts[k]), dtype=torch.float32, device=device)

        self.files = list(motion_files)
        self.clip_names = [Path(f).stem for f in motion_files]
        self.num_clips = len(motion_files)
        self.fps = fps_values[0]
        self.joint_pos = cat("joint_pos")
        self.joint_vel = cat("joint_vel")
        self._body_indexes = body_indexes
        self.body_pos_w = cat("body_pos_w")[:, body_indexes]
        self.body_quat_w = cat("body_quat_w")[:, body_indexes]
        self.body_lin_vel_w = cat("body_lin_vel_w")[:, body_indexes]
        self.body_ang_vel_w = cat("body_ang_vel_w")[:, body_indexes]
        self.time_step_total = self.joint_pos.shape[0]
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.clip_starts = torch.cumsum(self.clip_lengths, 0) - self.clip_lengths

    def clip_of(self, frames: torch.Tensor) -> torch.Tensor:
        """Index of the clip each global frame belongs to."""
        return torch.searchsorted(self.clip_starts, frames.to(self.clip_starts.device), right=True) - 1

    def clip_end_of(self, frames: torch.Tensor) -> torch.Tensor:
        """One past the last global frame of each frame's clip."""
        c = self.clip_of(frames)
        return self.clip_starts[c] + self.clip_lengths[c]


class MultiMotionCommand(MotionCommand):
    cfg: MultiMotionCommandCfg
    motion: MultiMotionLoader

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv):
        files = list(cfg.motion_files) or resolve_motion_files(cfg.motion_file)
        # The parent loads one motion file; give it the first clip, then replace its motion and sampling bins.
        super().__init__(dataclasses.replace(cfg, motion_file=files[0]), env)
        self.cfg = cfg
        self.motion = MultiMotionLoader(files, self.body_indexes, device=self.device)

        # mjlab's bin count for one motion, applied to each clip.
        bins = [int(int(n) // (1 / env.step_dt)) + 1 for n in self.motion.clip_lengths]
        self.bins_per_clip = torch.tensor(bins, dtype=torch.long, device=self.device)
        self.bin_offset = torch.cumsum(self.bins_per_clip, 0) - self.bins_per_clip
        self.bin_clip = torch.repeat_interleave(torch.arange(len(bins), device=self.device), self.bins_per_clip)
        self.bin_count = int(sum(bins))
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)

        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sample_clip_starts = False

    def _frame_to_bin(self, frames: torch.Tensor) -> torch.Tensor:
        """Global sampling bin of each global frame (the parent's frame-to-bin mapping, within its clip)."""
        c = self.motion.clip_of(frames)
        local = frames - self.motion.clip_starts[c]
        n = self.bins_per_clip[c]
        local_bin = torch.minimum((local * n) // self.motion.clip_lengths[c].clamp(min=1), n - 1)
        return self.bin_offset[c] + local_bin

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            fail_bins = self._frame_to_bin(self.time_steps[env_ids][episode_failed])
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample a bin as the parent does, then a frame inside it.
        probs = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        probs = torch.nn.functional.pad(
            probs.unsqueeze(0).unsqueeze(0), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
        )
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
        probs = probs / probs.sum()
        sampled_bins = torch.multinomial(probs, len(env_ids), replacement=True)

        c = self.bin_clip[sampled_bins]
        local_bin = sampled_bins - self.bin_offset[c]
        u = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        local_frame = ((local_bin + u) / self.bins_per_clip[c] * (self.motion.clip_lengths[c] - 1)).long()
        self.time_steps[env_ids] = self.motion.clip_starts[c] + local_frame

        H = -(probs * (probs + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        pmax, imax = probs.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _uniform_sampling(self, env_ids: torch.Tensor):
        if self._sample_clip_starts:
            # "start" mode: the first frame of a clip, each clip equally likely.
            c = torch.randint(0, self.motion.num_clips, (len(env_ids),), device=self.device)
            self.time_steps[env_ids] = self.motion.clip_starts[c]
            return
        super()._uniform_sampling(env_ids)  # uniform over all frames, so longer clips are picked more often

    def _resample_command(self, env_ids: torch.Tensor):
        # The parent sends every env to global frame 0 in "start" mode, which is only the first clip; route it
        # through uniform sampling with clip starts instead.
        start_mode = self.cfg.sampling_mode == "start"
        if start_mode:
            self.cfg.sampling_mode = "uniform"
            self._sample_clip_starts = True
        try:
            super()._resample_command(env_ids)
        finally:
            if start_mode:
                self.cfg.sampling_mode = "start"
                self._sample_clip_starts = False
        self.clip_ids[env_ids] = self.motion.clip_of(self.time_steps[env_ids])

    def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
        super().reset_to_frame(env_ids, frame)
        self.clip_ids[env_ids] = self.motion.clip_of(self.time_steps[env_ids])

    def _update_command(self, env_ids: torch.Tensor | None = None):
        # The parent's update, with the wrap at the end of each env's own clip instead of the end of all frames.
        if env_ids is None:
            self.time_steps += 1
        else:
            self.time_steps[env_ids] += 1
        clip_ends = self.motion.clip_starts[self.clip_ids] + self.motion.clip_lengths[self.clip_ids]
        wrap_ids = torch.where(self.time_steps >= clip_ends)[0]
        if wrap_ids.numel() > 0:
            self._resample_command(wrap_ids)

        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()

        self.update_relative_body_poses()

        if env_ids is None and self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
    motion_files: tuple[str, ...] = ()
    """Explicit list of motion files. When empty, ``motion_file`` is expanded (a file, directory or .txt manifest)."""

    def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)
