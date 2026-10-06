from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

import whole_body_tracking_mjlab.tasks.tracking.mdp as mdp
from whole_body_tracking_mjlab.robots.sprout import SPROUT_ACTION_SCALE, get_sprout_robot_cfg
from whole_body_tracking_mjlab.tasks.tracking.config.g1.rl_cfg import LOW_FREQ_SCALE
from whole_body_tracking_mjlab.tasks.tracking.tracking_env_cfg import (
    apply_low_freq,
    apply_play_mode,
    make_tracking_env_cfg,
    remove_domain_randomization,
    remove_state_estimation,
)

END_EFFECTOR_BODY_NAMES = (
    "left_foot_link",
    "right_foot_link",
    "left_gripper_lower_link",
    "right_gripper_lower_link",
)
# Push velocities relative to G1.
PUSH_SCALE = 0.5

# The gripper fingers may touch things too.
CONTACT_ALLOWED_BODY_NAMES = END_EFFECTOR_BODY_NAMES + ("left_gripper_upper_link", "right_gripper_upper_link")
# Actor terms read from the robot's sensors or state estimator. Each is delayed by 0 or 1 policy steps, resampled every
# step; the reference motion and the last action are known exactly on the robot.
DELAYED_OBS_TERMS = ("motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel")


def sprout_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_sprout_robot_cfg()}
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="undesired_contacts",
            primary=ContactMatch(mode="body", pattern=".*", entity="robot", exclude=CONTACT_ALLOWED_BODY_NAMES),
            fields=("found", "force"),
            reduce="netforce",
            history_length=cfg.decimation,
        ),
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = SPROUT_ACTION_SCALE

    for name in DELAYED_OBS_TERMS:
        cfg.observations["actor"].terms[name].delay_max_lag = 1

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, mdp.MotionCommandCfg)
    motion_cmd.anchor_body_name = "torso_link"
    # The first body is written as the floating base on reset, so it must be the free-joint body (torso_link).
    motion_cmd.body_names = (
        "torso_link",
        "pelvis_link",
        "left_femur_upper_link",
        "left_tibia_link",
        "left_foot_link",
        "right_femur_upper_link",
        "right_tibia_link",
        "right_foot_link",
        "head_link",
        "left_humerus_upper_link",
        "left_radius_link",
        "left_gripper_lower_link",
        "right_humerus_upper_link",
        "right_radius_link",
        "right_gripper_lower_link",
    )

    cfg.events["physics_material"].params["asset_cfg"].geom_names = r"^(left|right)_foot_link_collision_[01]$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    push_params = cfg.events["push_robot"].params
    push_params["velocity_range"] = {
        k: (lo * PUSH_SCALE, hi * PUSH_SCALE) for k, (lo, hi) in push_params["velocity_range"].items()
    }
    cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTOR_BODY_NAMES
    cfg.viewer.body_name = "torso_link"
    cfg.viewer.distance = 2.0

    if play:
        apply_play_mode(cfg)

    return cfg


def sprout_flat_wo_state_estimation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = sprout_flat_env_cfg(play=play)
    remove_state_estimation(cfg)
    return cfg


def sprout_flat_low_freq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = sprout_flat_env_cfg(play=play)
    apply_low_freq(cfg, LOW_FREQ_SCALE)
    return cfg


def sprout_flat_wo_state_estimation_no_dr_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = sprout_flat_wo_state_estimation_env_cfg(play=play)
    remove_domain_randomization(cfg)
    return cfg
