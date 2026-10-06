"""Convert a retargeted motion to the npz format used for training, via forward kinematics.

Input is a CSV (``root_pos(3), root_quat_xyzw(4), joint_pos(n)`` per row, Unitree convention) or a GMR pickle
(``root_pos``, ``root_rot`` in xyzw, ``dof_pos``, ``fps``). Unlike ``mjlab.scripts.csv_to_npz``, this supports every
robot in this package, writes the npz locally, and uploads to the WandB registry only when ``--registry-upload`` is set.
"""

import numpy as np
import pickle
import tempfile
import torch
import tyro
from pathlib import Path
from typing import Literal

import mjlab
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg

from whole_body_tracking_mjlab.robots.sprout import SPROUT_JOINT_NAMES
from whole_body_tracking_mjlab.tasks.tracking.config.g1.flat_env_cfg import g1_flat_env_cfg
from whole_body_tracking_mjlab.tasks.tracking.config.sprout.flat_env_cfg import sprout_flat_env_cfg

# Column order of the Unitree-retargeted LAFAN1 CSVs.
G1_CSV_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

ROBOTS = {
    "g1": (g1_flat_env_cfg, G1_CSV_JOINT_NAMES),
    "sprout": (sprout_flat_env_cfg, SPROUT_JOINT_NAMES),
}


def _gmr_pickle_to_csv(pkl_file: str, csv_file: str) -> float:
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    np.savetxt(csv_file, np.hstack([data["root_pos"], data["root_rot"], data["dof_pos"]]), delimiter=",")
    return float(data["fps"])


def main(
    input_file: str,
    output_file: str,
    robot: Literal["g1", "sprout"] = "g1",
    input_fps: float | None = None,
    output_fps: float = 50.0,
    device: str = "cuda:0",
    line_range: tuple[int, int] | None = None,
    registry_upload: str | None = None,
):
    """
    Args:
        input_file: Retargeted motion, ``.csv`` or GMR ``.pkl``.
        output_file: Path of the npz to write.
        robot: Robot the motion was retargeted to.
        input_fps: Frame rate of the input. Required for CSV; read from the file for GMR pickles.
        output_fps: Frame rate of the npz. Must match the policy rate (1 / (decimation * timestep)).
        device: Torch / warp device.
        line_range: Inclusive 1-based range of CSV lines to convert.
        registry_upload: If set, also upload the npz to the WandB "motions" registry under this collection name.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING]: CUDA is not available. Falling back to CPU.")
        device = "cpu"

    env_cfg_fn, joint_names = ROBOTS[robot]

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = input_file
        if input_file.endswith(".pkl"):
            csv_file = str(Path(tmp_dir) / "motion.csv")
            pkl_fps = _gmr_pickle_to_csv(input_file, csv_file)
            input_fps = input_fps or pkl_fps
        if input_fps is None:
            raise ValueError("--input-fps is required for CSV input.")
        motion = MotionLoader(csv_file, input_fps, output_fps, device, line_range=line_range)

    if motion.motion_dof_poss.shape[1] != len(joint_names):
        raise ValueError(f"Motion has {motion.motion_dof_poss.shape[1]} joints; {robot} expects {len(joint_names)}.")

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps
    scene = Scene(env_cfg_fn().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot_entity = scene["robot"]
    joint_ids = robot_entity.find_joints(joint_names, preserve_order=True)[0]
    scene.reset()

    log = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    for _ in range(motion.output_frames):
        (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), _ = motion.get_next_state()

        root_state = robot_entity.data.default_root_state.clone()
        root_state[:, 0:3] = base_pos
        root_state[:, 3:7] = base_rot
        root_state[:, 7:10] = base_lin_vel
        root_state[:, 10:] = base_ang_vel
        robot_entity.write_root_state_to_sim(root_state)

        joint_pos = robot_entity.data.default_joint_pos.clone()
        joint_vel = robot_entity.data.default_joint_vel.clone()
        joint_pos[:, joint_ids] = dof_pos
        joint_vel[:, joint_ids] = dof_vel
        robot_entity.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        data = robot_entity.data
        log["joint_pos"].append(data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(data.body_link_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(data.body_link_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(data.body_link_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(data.body_link_ang_vel_w[0].cpu().numpy().copy())

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, fps=[output_fps], **{k: np.stack(v) for k, v in log.items()})
    print(f"[INFO]: Saved {motion.output_frames} frames at {output_fps} fps to {output_file}")

    if registry_upload is not None:
        import wandb

        run = wandb.init(project="csv_to_npz", name=registry_upload)
        artifact = wandb.Artifact(name=registry_upload, type="motions")
        # mjlab's train/play download "motion.npz" from the artifact.
        artifact.add_file(output_file, name="motion.npz")
        artifact = run.log_artifact(artifact)
        run.link_artifact(artifact=artifact, target_path=f"wandb-registry-motions/{registry_upload}")
        print(f"[INFO]: Motion saved to wandb registry: motions/{registry_upload}")
        wandb.finish()


def cli():
    tyro.cli(main, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
    cli()
