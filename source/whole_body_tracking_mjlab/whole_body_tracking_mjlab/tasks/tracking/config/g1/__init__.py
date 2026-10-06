from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .flat_env_cfg import (
    g1_flat_env_cfg,
    g1_flat_low_freq_env_cfg,
    g1_flat_wo_state_estimation_env_cfg,
    g1_flat_wo_state_estimation_no_dr_env_cfg,
)
from .rl_cfg import g1_flat_low_freq_ppo_runner_cfg, g1_flat_ppo_runner_cfg

##
# Register mjlab tasks. IDs match the Isaac Lab gym IDs of this repository.
##

register_mjlab_task(
    task_id="Tracking-Flat-G1-v0",
    env_cfg=g1_flat_env_cfg(),
    play_env_cfg=g1_flat_env_cfg(play=True),
    rl_cfg=g1_flat_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    env_cfg=g1_flat_wo_state_estimation_env_cfg(),
    play_env_cfg=g1_flat_wo_state_estimation_env_cfg(play=True),
    rl_cfg=g1_flat_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Tracking-Flat-G1-Low-Freq-v0",
    env_cfg=g1_flat_low_freq_env_cfg(),
    play_env_cfg=g1_flat_low_freq_env_cfg(play=True),
    rl_cfg=g1_flat_low_freq_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Tracking-Flat-G1-Wo-State-Estimation-No-DR-v0",
    env_cfg=g1_flat_wo_state_estimation_no_dr_env_cfg(),
    play_env_cfg=g1_flat_wo_state_estimation_no_dr_env_cfg(play=True),
    rl_cfg=g1_flat_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
