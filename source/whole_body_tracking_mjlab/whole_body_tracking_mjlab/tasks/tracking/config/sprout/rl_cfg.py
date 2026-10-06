from mjlab.rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking_mjlab.tasks.tracking.config.g1.rl_cfg import (
    g1_flat_low_freq_ppo_runner_cfg,
    g1_flat_ppo_runner_cfg,
)

# Same PPO hyperparameters as G1.


def sprout_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = g1_flat_ppo_runner_cfg()
    cfg.experiment_name = "sprout_flat"
    return cfg


def sprout_flat_low_freq_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = g1_flat_low_freq_ppo_runner_cfg()
    cfg.experiment_name = "sprout_flat"
    return cfg
