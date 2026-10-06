from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

import whole_body_tracking_mjlab.tasks.tracking.mdp as mdp
from whole_body_tracking_mjlab.tasks.tracking.config.g1.rl_cfg import LOW_FREQ_SCALE
from whole_body_tracking_mjlab.tasks.tracking.tracking_env_cfg import (
    apply_low_freq,
    apply_play_mode,
    make_tracking_env_cfg,
    remove_domain_randomization,
    remove_state_estimation,
)

END_EFFECTOR_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_g1_robot_cfg()}
    # Contacts on any body other than the end effectors (ground or self) are penalized.
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="undesired_contacts",
            primary=ContactMatch(mode="body", pattern=".*", entity="robot", exclude=END_EFFECTOR_BODY_NAMES),
            fields=("found", "force"),
            reduce="netforce",
            history_length=cfg.decimation,
        ),
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, mdp.MotionCommandCfg)
    motion_cmd.anchor_body_name = "torso_link"
    motion_cmd.body_names = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )

    cfg.events["physics_material"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTOR_BODY_NAMES
    cfg.viewer.body_name = "torso_link"

    if play:
        apply_play_mode(cfg)

    return cfg


def g1_flat_wo_state_estimation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = g1_flat_env_cfg(play=play)
    remove_state_estimation(cfg)
    return cfg


def g1_flat_low_freq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = g1_flat_env_cfg(play=play)
    apply_low_freq(cfg, LOW_FREQ_SCALE)
    return cfg


def g1_flat_wo_state_estimation_no_dr_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = g1_flat_wo_state_estimation_env_cfg(play=play)
    remove_domain_randomization(cfg)
    return cfg
