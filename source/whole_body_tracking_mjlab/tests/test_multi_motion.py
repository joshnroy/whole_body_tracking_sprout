"""Tests for training one tracking policy on several reference motions.

The loader tests run on the CPU without a simulator. The command tests build a small G1 environment and need CUDA
(MuJoCo Warp); they are skipped without it.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from pathlib import Path

import pytest

from whole_body_tracking_mjlab.tasks.tracking.mdp.multi_motion import MultiMotionLoader, resolve_motion_files

NUM_BODIES = 40  # more than any robot here has, so every body index is valid
NUM_JOINTS = 29  # G1


def write_clip(path: Path, frames: int, *, fps: float = 50.0, joints: int = NUM_JOINTS, marker: float = 0.0) -> Path:
    """A standing-height reference whose joint positions are all ``marker``, so tests can tell clips apart."""
    body_pos = np.zeros((frames, NUM_BODIES, 3), dtype=np.float32)
    body_pos[:, :, 2] = 0.8
    body_quat = np.zeros((frames, NUM_BODIES, 4), dtype=np.float32)
    body_quat[:, :, 0] = 1.0
    np.savez(
        path,
        fps=[fps],
        joint_pos=np.full((frames, joints), marker, dtype=np.float32),
        joint_vel=np.zeros((frames, joints), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, NUM_BODIES, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, NUM_BODIES, 3), dtype=np.float32),
    )
    return path


BODY_INDEXES = torch.tensor([0, 3, 7])


##
# Resolving motion_file
##


def test_resolve_single_file(tmp_path):
    f = write_clip(tmp_path / "walk.npz", 10)
    assert resolve_motion_files(str(f)) == [str(f)]


def test_resolve_directory_sorted(tmp_path):
    write_clip(tmp_path / "b.npz", 10)
    write_clip(tmp_path / "a.npz", 10)
    (tmp_path / "notes.txt").write_text("not a motion")
    assert resolve_motion_files(str(tmp_path)) == [str(tmp_path / "a.npz"), str(tmp_path / "b.npz")]


def test_resolve_manifest_relative_paths_and_comments(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    write_clip(clips / "a.npz", 10)
    write_clip(clips / "b.npz", 10)
    manifest = tmp_path / "set.txt"
    manifest.write_text("# training set\nclips/b.npz\n\n  clips/a.npz  \n")
    assert resolve_motion_files(str(manifest)) == [str(clips / "b.npz"), str(clips / "a.npz")]


def test_resolve_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="no .npz"):
        resolve_motion_files(str(tmp_path))


def test_resolve_missing_manifest_entry_raises(tmp_path):
    manifest = tmp_path / "set.txt"
    manifest.write_text("missing.npz\n")
    with pytest.raises(FileNotFoundError, match="missing.npz"):
        resolve_motion_files(str(manifest))


##
# Loading and concatenating clips
##


def test_loader_concatenates_clips(tmp_path):
    a = write_clip(tmp_path / "a.npz", 30, marker=1.0)
    b = write_clip(tmp_path / "b.npz", 70, marker=2.0)
    m = MultiMotionLoader([str(a), str(b)], BODY_INDEXES)
    assert m.time_step_total == 100
    assert m.num_clips == 2
    assert m.clip_names == ["a", "b"]
    assert m.clip_starts.tolist() == [0, 30]
    assert m.clip_lengths.tolist() == [30, 70]
    assert m.fps == 50.0
    assert m.joint_pos.shape == (100, NUM_JOINTS)
    assert torch.all(m.joint_pos[:30] == 1.0) and torch.all(m.joint_pos[30:] == 2.0)
    # Bodies are selected with body_indexes, like mjlab's MotionLoader.
    assert m.body_pos_w.shape == (100, len(BODY_INDEXES), 3)
    assert m.body_quat_w.shape == (100, len(BODY_INDEXES), 4)


def test_loader_maps_frames_to_clips(tmp_path):
    a = write_clip(tmp_path / "a.npz", 30)
    b = write_clip(tmp_path / "b.npz", 70)
    m = MultiMotionLoader([str(a), str(b)], BODY_INDEXES)
    frames = torch.tensor([0, 29, 30, 99])
    assert m.clip_of(frames).tolist() == [0, 0, 1, 1]
    assert m.clip_end_of(frames).tolist() == [30, 30, 100, 100]


def test_loader_rejects_mismatched_fps(tmp_path):
    a = write_clip(tmp_path / "a.npz", 30, fps=50.0)
    b = write_clip(tmp_path / "b.npz", 30, fps=30.0)
    with pytest.raises(ValueError, match="fps"):
        MultiMotionLoader([str(a), str(b)], BODY_INDEXES)


def test_loader_rejects_mismatched_joints(tmp_path):
    a = write_clip(tmp_path / "a.npz", 30, joints=29)
    b = write_clip(tmp_path / "b.npz", 30, joints=27)
    with pytest.raises(ValueError, match="joints"):
        MultiMotionLoader([str(a), str(b)], BODY_INDEXES)


##
# The motion command inside an environment
##

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for MuJoCo Warp")
CLIP_FRAMES = (60, 130)  # at 50 Hz: 2 and 3 one-second sampling bins


def make_env(motion_file: str, num_envs: int = 64, sampling_mode: str = "uniform"):
    from mjlab.envs import ManagerBasedRlEnv
    from whole_body_tracking_mjlab.tasks.tracking.config.g1.flat_env_cfg import g1_flat_env_cfg

    cfg = g1_flat_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.commands["motion"].motion_file = motion_file
    cfg.commands["motion"].sampling_mode = sampling_mode
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


@pytest.fixture(scope="module")
def clip_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    for i, n in enumerate(CLIP_FRAMES):
        write_clip(d / f"clip{i}.npz", n, marker=float(i + 1))
    return d


@pytest.fixture(scope="module")
def env(clip_dir):
    e = make_env(str(clip_dir))
    e.reset()
    yield e
    e.close()


def motion_term(env):
    return env.command_manager.get_term("motion")


def assert_consistent(cmd):
    """Every env's frame lies inside the clip it is assigned to."""
    starts = cmd.motion.clip_starts[cmd.clip_ids]
    ends = starts + cmd.motion.clip_lengths[cmd.clip_ids]
    assert torch.all(cmd.time_steps >= starts) and torch.all(cmd.time_steps < ends)
    assert torch.all(cmd.motion.clip_of(cmd.time_steps) == cmd.clip_ids)


def test_task_configs_use_multi_motion_command():
    from whole_body_tracking_mjlab.tasks.tracking.config.g1.flat_env_cfg import g1_flat_env_cfg
    from whole_body_tracking_mjlab.tasks.tracking.mdp.multi_motion import MultiMotionCommandCfg

    assert isinstance(g1_flat_env_cfg().commands["motion"], MultiMotionCommandCfg)


@cuda
def test_command_loads_every_clip(env):
    cmd = motion_term(env)
    assert cmd.motion.num_clips == 2
    assert cmd.motion.time_step_total == sum(CLIP_FRAMES)
    assert_consistent(cmd)


@cuda
def test_reference_comes_from_assigned_clip(env):
    cmd = motion_term(env)
    # write_clip marks clip i's joint positions with i + 1.
    assert torch.allclose(cmd.joint_pos[:, 0], (cmd.clip_ids + 1).float())


@cuda
def test_reaching_clip_end_resamples_instead_of_running_into_next_clip(env):
    cmd = motion_term(env)
    end_of_first = int(cmd.motion.clip_lengths[0])
    cmd.time_steps[:] = end_of_first - 1
    cmd.clip_ids[:] = 0
    cmd._update_command()
    assert_consistent(cmd)
    # Without the per-clip wrap every env would now sit on the next clip's first frame, still labelled clip 0.
    moved_on_unlabelled = (cmd.time_steps == end_of_first) & (cmd.clip_ids == 0)
    assert not torch.any(moved_on_unlabelled)


@cuda
def test_adaptive_bins_are_per_clip(clip_dir):
    e = make_env(str(clip_dir), num_envs=512, sampling_mode="adaptive")
    try:
        cmd = motion_term(e)
        steps_per_bin = round(1 / e.step_dt)
        expected = [n // steps_per_bin + 1 for n in CLIP_FRAMES]  # mjlab's per-motion bin count, per clip
        assert cmd.bin_count == sum(expected)
        assert cmd.bin_clip.tolist() == [0] * expected[0] + [1] * expected[1]

        # Make the second clip's last bin the only one that fails: sampling should concentrate there.
        cmd.bin_failed_count.zero_()
        cmd.bin_failed_count[-1] = 1.0
        env_ids = torch.arange(e.num_envs, device=e.device)
        cmd._adaptive_sampling(env_ids)
        frames = cmd.time_steps
        # mjlab stretches a motion's bins to cover it exactly: bin k spans [k, k + 1) / bins * (frames - 1).
        last_bin_start = int(cmd.motion.clip_starts[1]) + math.floor(
            (expected[1] - 1) / expected[1] * (CLIP_FRAMES[1] - 1)
        )
        in_last_bin = (frames >= last_bin_start) & (frames < cmd.motion.time_step_total)
        assert in_last_bin.float().mean() > 0.8
        assert torch.all(frames < cmd.motion.time_step_total)
    finally:
        e.close()


@cuda
def test_start_mode_starts_each_env_at_a_clip_start(clip_dir):
    e = make_env(str(clip_dir), num_envs=128, sampling_mode="start")
    try:
        cmd = motion_term(e)
        # Resample directly: env.reset() also advances the command one step past the sampled frame.
        cmd._resample_command(torch.arange(e.num_envs, device=e.device))
        starts = set(cmd.motion.clip_starts.tolist())
        assert set(cmd.time_steps.tolist()) <= starts
        assert set(cmd.clip_ids.tolist()) == {0, 1}  # 128 envs: both clips get picked
        assert_consistent(cmd)
    finally:
        e.close()


@cuda
def test_single_file_still_works(clip_dir):
    e = make_env(str(clip_dir / "clip1.npz"), num_envs=16)
    try:
        e.reset()
        cmd = motion_term(e)
        assert cmd.motion.num_clips == 1
        assert cmd.motion.time_step_total == CLIP_FRAMES[1]
        assert cmd.bin_count == math.floor(CLIP_FRAMES[1] / round(1 / e.step_dt)) + 1
        assert_consistent(cmd)
    finally:
        e.close()
