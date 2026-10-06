from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .flat_env_cfg import (
    sprout_flat_env_cfg,
    sprout_flat_low_freq_env_cfg,
    sprout_flat_wo_state_estimation_env_cfg,
    sprout_flat_wo_state_estimation_no_dr_env_cfg,
)
from .rl_cfg import sprout_flat_low_freq_ppo_runner_cfg, sprout_flat_ppo_runner_cfg

register_mjlab_task(
    task_id="Tracking-Flat-Sprout-v0",
    env_cfg=sprout_flat_env_cfg(),
    play_env_cfg=sprout_flat_env_cfg(play=True),
    rl_cfg=sprout_flat_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Tracking-Flat-Sprout-Wo-State-Estimation-v0",
    env_cfg=sprout_flat_wo_state_estimation_env_cfg(),
    play_env_cfg=sprout_flat_wo_state_estimation_env_cfg(play=True),
    rl_cfg=sprout_flat_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Tracking-Flat-Sprout-Low-Freq-v0",
    env_cfg=sprout_flat_low_freq_env_cfg(),
    play_env_cfg=sprout_flat_low_freq_env_cfg(play=True),
    rl_cfg=sprout_flat_low_freq_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Tracking-Flat-Sprout-Wo-State-Estimation-No-DR-v0",
    env_cfg=sprout_flat_wo_state_estimation_no_dr_env_cfg(),
    play_env_cfg=sprout_flat_wo_state_estimation_no_dr_env_cfg(play=True),
    rl_cfg=sprout_flat_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
