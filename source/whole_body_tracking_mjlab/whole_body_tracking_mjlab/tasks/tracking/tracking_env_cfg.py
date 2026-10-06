"""mjlab port of ``whole_body_tracking/tasks/tracking/tracking_env_cfg.py``.

Terms and hyperparameters follow the Isaac Lab config one-to-one. Where the two simulators differ:

- ``base_lin_vel`` / ``base_ang_vel`` read the robot's IMU sensors.
- ``add_joint_default_pos`` (calibration error) becomes mjlab's ``encoder_bias`` plus ``joint_pos_rel(biased=True)``.
- ``physics_material`` randomizes foot friction only, because the other MuJoCo collision geoms use ``condim=1``
  (frictionless) contacts.
- ``undesired_contacts`` reads a contact sensor that robot configs add under the name ``undesired_contacts``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.observation_manager import ObservationGroupCfg as ObsGroup
from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import whole_body_tracking_mjlab.tasks.tracking.mdp as mdp

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
    """Robot-agnostic tracking config. Robot configs fill in the entity, sensors and body names."""

    ##
    # Observations
    ##

    policy_terms = {
        "command": ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}),
        "motion_anchor_pos_b": ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        ),
        "motion_anchor_ori_b": ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        "base_lin_vel": ObsTerm(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}, noise=Unoise(n_min=-0.5, n_max=0.5)
        ),
        "base_ang_vel": ObsTerm(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=Unoise(n_min=-0.2, n_max=0.2)
        ),
        "joint_pos": ObsTerm(func=mdp.joint_pos_rel, params={"biased": True}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
        "actions": ObsTerm(func=mdp.last_action),
    }

    privileged_terms = {
        "command": ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}),
        "motion_anchor_pos_b": ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}),
        "motion_anchor_ori_b": ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}),
        "body_pos": ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"}),
        "body_ori": ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"}),
        "base_lin_vel": ObsTerm(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
        "base_ang_vel": ObsTerm(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
        "joint_pos": ObsTerm(func=mdp.joint_pos_rel),
        "joint_vel": ObsTerm(func=mdp.joint_vel_rel),
        "actions": ObsTerm(func=mdp.last_action),
    }

    observations = {
        "actor": ObsGroup(terms=policy_terms, concatenate_terms=True, enable_corruption=True),
        "critic": ObsGroup(terms=privileged_terms, concatenate_terms=True, enable_corruption=False),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        # Scale is set per robot.
        "joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), use_default_offset=True)
    }

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "motion": mdp.MotionCommandCfg(
            entity_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range=VELOCITY_RANGE,
            joint_position_range=(-0.1, 0.1),
            # Set per robot / on the command line.
            motion_file="",
            anchor_body_name="",
            body_names=(),
        )
    }

    ##
    # Events
    ##

    events: dict[str, EventTerm] = {
        # startup
        "physics_material": EventTerm(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per robot.
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        ),
        "add_joint_default_pos": EventTerm(
            func=dr.encoder_bias,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
        ),
        "base_com": EventTerm(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per robot.
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        ),
        # interval
        "push_robot": EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": VELOCITY_RANGE},
        ),
    }

    ##
    # Rewards
    ##

    rewards: dict[str, RewTerm] = {
        "motion_global_anchor_pos": RewTerm(
            func=mdp.motion_global_anchor_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_global_anchor_ori": RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_pos": RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_body_ori": RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_lin_vel": RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 1.0},
        ),
        "motion_body_ang_vel": RewTerm(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 3.14},
        ),
        "action_rate_l2": RewTerm(func=mdp.action_rate_l2, weight=-1e-1),
        "joint_limit": RewTerm(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "undesired_contacts": RewTerm(
            func=mdp.undesired_contacts,
            weight=-0.1,
            params={"sensor_name": "undesired_contacts", "threshold": 1.0},
        ),
    }

    ##
    # Terminations
    ##

    terminations: dict[str, DoneTerm] = {
        "time_out": DoneTerm(func=mdp.time_out, time_out=True),
        "anchor_pos": DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25},
        ),
        "anchor_ori": DoneTerm(
            func=mdp.bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
        ),
        "ee_body_pos": DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25, "body_names": ()},  # Set per robot.
        ),
    }

    ##
    # Environment configuration
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=4096, env_spacing=2.5),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per robot.
            distance=2.8,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=250,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=10.0,
    )


def apply_play_mode(cfg: ManagerBasedRlEnvCfg) -> None:
    """Play the motion from the start with no perturbations, and never time out."""
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot")
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, mdp.MotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"


def remove_state_estimation(cfg: ManagerBasedRlEnvCfg) -> None:
    """Drop the policy observations that need a base position / linear velocity estimator."""
    terms = cfg.observations["actor"].terms
    terms.pop("motion_anchor_pos_b")
    terms.pop("base_lin_vel")


def remove_domain_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
    """Drop friction / encoder bias / CoM randomization, pushes and observation noise.

    The reset-state randomization around the reference (``pose_range``, ``velocity_range``, ``joint_position_range``)
    is kept: it is initial-state sampling for learning, not domain randomization.
    """
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        cfg.events.pop(name, None)
    cfg.observations["actor"].enable_corruption = False


def apply_low_freq(cfg: ManagerBasedRlEnvCfg, scale: float) -> None:
    """Run the policy at ``scale`` times the control frequency."""
    cfg.decimation = round(cfg.decimation / scale)
    cfg.rewards["action_rate_l2"].weight *= scale
    # Keep the contact history covering one full policy step.
    for sensor in cfg.scene.sensors:
        if sensor.name == "undesired_contacts":
            sensor.history_length = cfg.decimation
