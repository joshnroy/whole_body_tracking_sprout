"""Optional physics integration checks using a locally supplied policy and robot model."""

import numpy as np
import os
import unittest

from whole_body_tracking_mjlab.scripts.motion_match_browser import BrowserConfig, MotionMatchSession


@unittest.skipUnless(os.environ.get("WBT_TEST_ONNX"), "Set WBT_TEST_ONNX to a compatible local policy")
class BrowserSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = MotionMatchSession(
            os.environ.get("WBT_TEST_TASK", "Tracking-Flat-Sprout-Wo-State-Estimation-v0"),
            BrowserConfig(onnx_file=os.environ["WBT_TEST_ONNX"]),
        )

    def setUp(self):
        self.session.reset()

    def test_pause_and_reset_restore_physics(self):
        s = self.session
        initial = s.sim.data.qpos.copy()
        s.control("set_forward", 0.4)
        s.control("toggle_pause")
        for _ in range(100):
            s.step()
        self.assertEqual(s.steps, 100)
        self.assertFalse(s.termination)
        self.assertGreater(np.linalg.norm(s.sim.data.qpos - initial), 0.01)
        s.control("pause")
        frozen = s.sim.data.qpos.copy()
        for _ in range(10):
            s.step()
        np.testing.assert_array_equal(s.sim.data.qpos, frozen)
        self.assertEqual(s.steps, 100)
        s.control("reset")
        np.testing.assert_allclose(s.sim.data.qpos, initial)
        self.assertEqual(s.steps, 0)
        self.assertEqual(s.command, (0.0, 0.0, 0.0))
        self.assertTrue(s.paused)
        np.testing.assert_array_equal(s.sim.last_action, 0)

    def test_termination_requires_reset(self):
        s = self.session
        s.control("toggle_pause")
        # Move the robot above its reference to trigger the task's actual termination.
        s.sim.data.qpos[s.sim.root_qpos_adr + 2] += 3.0
        s.step()
        self.assertTrue(s.termination)
        frozen = s.sim.data.qpos.copy()
        s.control("toggle_pause")
        s.step()
        np.testing.assert_array_equal(s.sim.data.qpos, frozen)
        self.assertTrue(s.paused)
        s.control("reset")
        s.control("toggle_pause")
        s.step()
        self.assertFalse(s.termination)
        self.assertFalse(s.paused)

    def test_commands_are_bounded_and_stop_clears_both(self):
        s = self.session
        s.control("set_forward", 100)
        s.control("set_turn", -100)
        s.control("set_lateral", 100)
        self.assertEqual(
            s.command, (s.cfg.matcher.max_forward_speed, s.cfg.matcher.max_lateral_speed, -s.cfg.matcher.max_yaw_rate)
        )
        s.control("set_forward", float("nan"))
        self.assertTrue(np.isfinite(s.command).all())
        s.control("stop")
        self.assertEqual(s.command, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
