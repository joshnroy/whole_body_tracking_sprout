"""Camera response checks independent of robot assets and the browser server."""

import numpy as np
import unittest

from whole_body_tracking_mjlab.scripts.browser_camera import ChaseCamera


class ChaseCameraTests(unittest.TestCase):
    def test_follow_converges_without_overshoot_at_different_frame_rates(self):
        results = []
        for fps in (30, 120):
            camera = ChaseCamera()
            camera.update([0, 0, 1], 0, 0)
            previous = 0.0
            for _ in range(fps):
                eye, target = camera.update([1, 0, 1], 0, 1 / fps)
                self.assertGreaterEqual(target[0], previous)
                self.assertLess(target[0], 1)
                previous = target[0]
            self.assertGreater(target[0], 0.99)
            np.testing.assert_allclose(eye - target, [-2.5, 0, 1.05])
            results.append(target)
        np.testing.assert_allclose(*results, atol=1e-12)

    def test_heading_wrap_uses_short_arc(self):
        camera = ChaseCamera()
        camera.update([0, 0, 1], np.deg2rad(179), 0)
        headings = []
        for _ in range(60):
            camera.update([0, 0, 1], np.deg2rad(-179), 1 / 60)
            headings.append(camera.yaw)
        travel = np.ptp(np.unwrap(headings))
        self.assertLess(travel, np.deg2rad(2))
        self.assertAlmostEqual(camera.yaw, np.deg2rad(-179), places=4)

    def test_gait_bob_and_heading_wobble_are_attenuated(self):
        camera = ChaseCamera()
        raw, filtered = [], []
        for frame in range(400):
            t = frame / 100
            z = 1 + 0.05 * np.sin(2 * np.pi * 3 * t)
            yaw = 0.1 * np.sin(2 * np.pi * 3 * t)
            _, target = camera.update([0, 0, z], yaw, 0.01)
            if frame >= 200:
                raw.append([z, yaw])
                filtered.append([target[2], camera.yaw])
        self.assertTrue(np.all(np.std(filtered, axis=0) < 0.4 * np.std(raw, axis=0)))

    def test_reset_snaps_to_new_robot_pose(self):
        camera = ChaseCamera()
        camera.update([100, 100, 2], 2, 0)
        camera.reset()
        eye, target = camera.update([0, 0, 1], 0, 0.02)
        np.testing.assert_allclose(target, [0, 0, 1.15])
        np.testing.assert_allclose(eye, [-2.5, 0, 2.2])


if __name__ == "__main__":
    unittest.main()
