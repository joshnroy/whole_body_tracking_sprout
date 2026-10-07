"""Damped chase camera, independent of the physics rate and robot gait bob."""

import numpy as np

from whole_body_tracking_mjlab.motion_matching import decay_spring


class ChaseCamera:
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        self.yaw_velocity = np.array(0.0)

    def update(self, position, yaw, dt):
        position = np.asarray(position, dtype=float)
        if self.target is None:
            self.target = position.copy()
            self.yaw = yaw
        else:
            # Follow translation promptly, but reject more of the walking bob.
            for axes, halflife in ((slice(0, 2), 0.10), (slice(2, 3), 0.22)):
                error, self.velocity[axes] = decay_spring(
                    self.target[axes] - position[axes], self.velocity[axes], halflife, dt
                )
                self.target[axes] = position[axes] + error
            # Use the shortest arc even when the measured heading crosses +/-pi.
            error = np.arctan2(np.sin(self.yaw - yaw), np.cos(self.yaw - yaw))
            error, self.yaw_velocity = decay_spring(np.array(error), self.yaw_velocity, 0.12, dt)
            self.yaw = yaw + float(error)
        look_at = self.target + np.array([0.0, 0.0, 0.15])
        eye = self.target + np.array([-2.5 * np.cos(self.yaw), -2.5 * np.sin(self.yaw), 1.2])
        return eye, look_at
