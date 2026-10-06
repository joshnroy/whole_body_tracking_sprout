"""Sprout humanoid for mjlab.

The model files (``sprout.xml`` and ``meshes/``) are not distributed with this repository. They are read from
``SPROUT_MODEL_DIR`` if set, else ``whole_body_tracking_mjlab/assets/sprout`` if it exists, else
``~/sprout-model-v0.1.4``.

The vendor MJCF's position actuators are replaced by the vendor's DC motor models (``sprout_motors.py``); joint damping
still comes from the MJCF. The standing pose is the vendor's default.
"""

import mujoco
import os
from pathlib import Path

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from whole_body_tracking_mjlab.robots.sprout_motors import SPROUT_ACTUATORS

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"  # Gitignored.
# SPROUT_MODEL_DIR if set, else the asset folder if the model was unzipped there, else the vendor zip unpacked in ~.
if "SPROUT_MODEL_DIR" in os.environ:
    SPROUT_MODEL_DIR = Path(os.environ["SPROUT_MODEL_DIR"]).expanduser()
elif (ASSET_DIR / "sprout").exists():
    SPROUT_MODEL_DIR = ASSET_DIR / "sprout"
else:
    SPROUT_MODEL_DIR = Path.home() / "sprout-model-v0.1.4"

# Joint order of the vendor MJCF; also the ``dof_pos`` column order of GMR-retargeted motions.
SPROUT_JOINT_NAMES = (
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_gripper_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_gripper_joint",
)

SPROUT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=SPROUT_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)

# BeyondMimic's per-joint 0.25 * effort_limit / stiffness would give the 3.5 Nm 3536 joints (neck pitch, wrist pitch,
# grippers) only 0.027 rad, so the scale is a fixed constant instead.
SPROUT_ACTION_SCALE = 0.25

# Vendor default standing pose, offset from the zero (T-pose).
STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.7),
    joint_pos={
        ".*_hip_pitch_joint": -0.06981317,
        ".*_knee_joint": 0.122173048,
        ".*_ankle_joint": -0.034906585,
        ".*_shoulder_pitch_joint": 0.073303829,
        "left_shoulder_roll_joint": -1.420698011,
        "right_shoulder_roll_joint": 1.420698011,
        ".*_shoulder_yaw_joint": 0.212930169,
        "left_elbow_joint": -0.174532925,
        "right_elbow_joint": 0.174532925,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)


def get_spec() -> mujoco.MjSpec:
    xml = SPROUT_MODEL_DIR / "sprout.xml"
    if not xml.exists():
        raise FileNotFoundError(f"Sprout model not found at {xml}. Unzip it there or set SPROUT_MODEL_DIR.")
    spec = mujoco.MjSpec.from_file(str(xml))

    # mjlab adds actuators from SPROUT_ARTICULATION.
    for actuator in list(spec.actuators):
        spec.delete(actuator)

    # The vendor frame sensors are in the world frame. Replace them with body-frame IMU readings, named like the
    # G1 sensors so the shared observation config applies.
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu_site")
    spec.add_sensor(
        name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu_site"
    )
    return spec


def get_sprout_robot_cfg() -> EntityCfg:
    return EntityCfg(init_state=STANDING_KEYFRAME, spec_fn=get_spec, articulation=SPROUT_ARTICULATION)
