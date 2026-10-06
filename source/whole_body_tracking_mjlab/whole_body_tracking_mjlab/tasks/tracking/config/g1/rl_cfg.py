from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

LOW_FREQ_SCALE = 0.5


def g1_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=30000,
        save_interval=500,
        experiment_name="g1_flat",
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


def g1_flat_low_freq_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = g1_flat_ppo_runner_cfg()
    cfg.num_steps_per_env = round(cfg.num_steps_per_env * LOW_FREQ_SCALE)
    cfg.algorithm.gamma = cfg.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
    cfg.algorithm.lam = cfg.algorithm.lam ** (1 / LOW_FREQ_SCALE)
    return cfg
