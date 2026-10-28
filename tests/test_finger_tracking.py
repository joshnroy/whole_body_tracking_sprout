"""CPU-only checks; no Sprout model assets or GPU are needed.

Run with: python -m unittest discover -s tests -v
"""

import torch
import unittest
from dataclasses import asdict
from types import SimpleNamespace

import whole_body_tracking_mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from whole_body_tracking_mjlab.tasks.tracking.mdp.rewards import motion_joint_position_error_exp

BASE_TASK = "Tracking-Flat-Sprout-Wo-State-Estimation-v0"
FINGERS_TASK = "Tracking-Flat-Sprout-Wo-State-Estimation-Fingers-v0"
FINGERS = ("left_gripper_joint", "right_gripper_joint")


class FingerRewardTests(unittest.TestCase):
    def setUp(self):
        # Deliberately interleave other joints and reverse left/right order.
        self.joint_names = ("arm", "right_gripper_joint", "leg", "left_gripper_joint")
        self.actual = torch.tensor([[9.0, -0.3, -7.0, 0.2], [1.0, -0.1, 2.0, 0.4], [0.0, 0.1, 0.0, 0.4]])
        self.reference = torch.tensor([[0.0, -0.3, 0.0, 0.2], [0.0, -0.1, 0.0, 0.2], [0.0, -0.1, 0.0, 0.2]])
        self.command = SimpleNamespace(
            robot=SimpleNamespace(joint_names=self.joint_names, data=SimpleNamespace(joint_pos=self.actual)),
            joint_pos=self.reference,
        )
        self.env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: self.command))

    def reward(self):
        return motion_joint_position_error_exp(self.env, "motion", 0.1, FINGERS)

    def test_reference_angles_and_selected_joint_mean(self):
        # Exact nonzero reference; one finger off by .2 rad; both off by .2 rad.
        torch.testing.assert_close(self.reward(), torch.exp(torch.tensor([0.0, -2.0, -4.0])))

    def test_other_joints_do_not_affect_reward(self):
        before = self.reward()
        self.actual[:, [0, 2]] += 100.0
        torch.testing.assert_close(self.reward(), before)

    def test_reward_follows_updated_reference(self):
        self.command.joint_pos = self.actual.clone()
        torch.testing.assert_close(self.reward(), torch.ones(3))

    def test_empty_joint_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "joint_names"):
            motion_joint_position_error_exp(self.env, "motion", 0.1, ())


class FingerTaskTests(unittest.TestCase):
    def test_only_one_reward_differs_from_base_in_train_and_play(self):
        for play in (False, True):
            with self.subTest(play=play):
                base = asdict(load_env_cfg(BASE_TASK, play=play))
                fingers = asdict(load_env_cfg(FINGERS_TASK, play=play))
                self.assertEqual(set(fingers["rewards"]) - set(base["rewards"]), {"motion_finger_joint_pos"})
                added = fingers["rewards"].pop("motion_finger_joint_pos")
                # EntityCfg creates a fresh stateless terrain factory per config.
                # Check its implementation before normalizing function identity.
                base_factory = base["scene"]["terrain"]["spec_fn"]
                fingers_factory = fingers["scene"]["terrain"]["spec_fn"]
                self.assertEqual(base_factory.__code__, fingers_factory.__code__)
                self.assertIs(base_factory.__closure__, None)
                self.assertIs(fingers_factory.__closure__, None)
                self.assertEqual(base_factory.__defaults__, fingers_factory.__defaults__)
                self.assertEqual(base_factory.__kwdefaults__, fingers_factory.__kwdefaults__)
                fingers["scene"]["terrain"]["spec_fn"] = base_factory
                self.assertEqual(fingers, base)
                self.assertEqual(added["func"], motion_joint_position_error_exp)
                self.assertEqual(added["weight"], 0.5)
                self.assertEqual(added["params"], {"command_name": "motion", "std": 0.1, "joint_names": FINGERS})

    def test_ppo_configuration_is_unchanged(self):
        self.assertEqual(asdict(load_rl_cfg(FINGERS_TASK)), asdict(load_rl_cfg(BASE_TASK)))


if __name__ == "__main__":
    unittest.main()
