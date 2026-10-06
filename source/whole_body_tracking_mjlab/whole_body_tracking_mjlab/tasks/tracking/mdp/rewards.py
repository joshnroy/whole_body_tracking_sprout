from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def undesired_contacts(env: ManagerBasedRlEnv, sensor_name: str, threshold: float) -> torch.Tensor:
    """Number of sensed bodies whose net contact force exceeded ``threshold`` during the last policy step.

    Mirrors Isaac Lab's ``undesired_contacts``: the force history spans the physics substeps of one policy step, so
    brief contacts that resolve mid-step still count.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1).amax(dim=-1)
    else:
        assert data.force is not None
        force_mag = torch.norm(data.force, dim=-1)
    return torch.sum(force_mag > threshold, dim=-1).float()
