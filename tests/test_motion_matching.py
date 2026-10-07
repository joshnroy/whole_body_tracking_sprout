"""Asset-free checks for legacy and named/embedded motion databases."""

import numpy as np
import tempfile
import unittest
from pathlib import Path

import mujoco
import onnx
from onnx import helper, numpy_helper
from whole_body_tracking_mjlab.motion_matching import MotionMatcher


class MotionDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="robot/base" pos="0 0 1"><freejoint name="robot/root"/>'
            '<geom size=".1" mass="1"/><body name="robot/left_foot" pos="0 .2 -.5">'
            '<joint name="robot/left"/><geom size=".05" mass=".1"/></body>'
            '<body name="robot/right_foot" pos="0 -.2 -.5"><joint name="robot/right"/>'
            '<geom size=".05" mass=".1"/></body><body name="robot/extra" pos="0 0 .2">'
            '<geom size=".05" mass=".1"/></body></body></worldbody></mujoco>'
        )
        self.feet = ("left_foot", "right_foot")
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        n = 150
        pos = np.repeat(data.xpos[None, 1:], n, axis=0)
        pos[:, :, 0] += np.arange(n)[:, None] * 0.004
        self.motion = {
            "fps": np.array([50.0]),
            "joint_pos": np.stack([0.1 * np.sin(np.arange(n) / 50), 0.2 * np.cos(np.arange(n) / 50)], axis=1),
            "joint_vel": np.stack([0.1 * np.cos(np.arange(n) / 50), -0.2 * np.sin(np.arange(n) / 50)], axis=1),
            "body_pos_w": pos,
            "body_quat_w": np.repeat(data.xquat[None, 1:], n, axis=0),
            "body_lin_vel_w": np.broadcast_to([0.2, 0, 0], (n, 4, 3)).copy(),
        }

    def named_motion(self):
        # ONNX keeps a subset of bodies, and names need not be in model order.
        order = [2, 0, 1]
        motion = dict(self.motion)
        for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w"):
            motion[key] = motion[key][:, order]
        for key in ("joint_pos", "joint_vel"):
            motion[key] = motion[key][:, ::-1]
        motion["body_names"] = np.array(["right_foot", "base", "left_foot"])
        motion["joint_names"] = np.array(["right", "left"])
        return motion

    def test_named_subset_matches_legacy_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            np.savez(path, **self.motion)
            legacy = MotionMatcher(str(path), self.model, self.feet)
        named = MotionMatcher(self.named_motion(), self.model, self.feet)
        np.testing.assert_allclose(legacy.features, named.features)
        legacy.reset()
        named.reset()
        for _ in range(20):
            expected, actual = legacy.step(0.2, 0.0), named.step(0.2, 0.0)
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key])

    def test_onnx_motion_is_read_without_running_policy(self):
        motion = self.named_motion()
        tensors = [
            numpy_helper.from_array(v, name=k + ".1")
            for k, v in motion.items()
            if k.startswith(("body_", "joint_")) and not k.endswith("names")
        ]
        exported = helper.make_model(helper.make_graph([], "reference", [], [], initializer=tensors))
        helper.set_model_props(exported, {k: ",".join(motion[k]) for k in ("joint_names", "body_names")})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.onnx"
            onnx.save(exported, path)
            matcher = MotionMatcher.from_onnx(str(path), self.model, self.feet, fps=50)
        np.testing.assert_allclose(matcher.features, MotionMatcher(self.motion, self.model, self.feet).features)

    def test_missing_foot_is_rejected(self):
        motion = self.named_motion()
        motion["body_names"][0] = "unknown"
        with self.assertRaisesRegex(ValueError, "missing required bodies"):
            MotionMatcher(motion, self.model, self.feet)

    def test_wrong_joint_names_are_rejected(self):
        motion = self.named_motion()
        motion["joint_names"][0] = "wrong"
        with self.assertRaisesRegex(ValueError, "joint names do not match"):
            MotionMatcher(motion, self.model, self.feet)

    def test_nonfinite_motion_is_rejected(self):
        motion = self.named_motion()
        motion["joint_pos"] = motion["joint_pos"].copy()
        motion["joint_pos"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite values"):
            MotionMatcher(motion, self.model, self.feet)

    def test_onnx_without_motion_explains_npz_fallback(self):
        exported = helper.make_model(helper.make_graph([], "policy", [], []))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            onnx.save(exported, path)
            with self.assertRaisesRegex(ValueError, "provide --motion-file"):
                MotionMatcher.from_onnx(str(path), self.model, self.feet, fps=50)


if __name__ == "__main__":
    unittest.main()
