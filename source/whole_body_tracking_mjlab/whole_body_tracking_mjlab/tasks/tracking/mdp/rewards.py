from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def motion_joint_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, joint_names: tuple[str, ...]
) -> torch.Tensor:
    """Match selected joints to their current reference angles, independently of tracked bodies.

    This supervises joints such as Sprout's moving fingers, whose motion does not
    change any of the existing tracked body poses. Average squared error only
    over the selected joints before applying the Gaussian kernel.
    """
    if not joint_names:
        raise ValueError("joint_names must contain at least one joint")
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    joint_ids = [command.robot.joint_names.index(name) for name in joint_names]
    error = torch.square(command.robot.data.joint_pos[:, joint_ids] - command.joint_pos[:, joint_ids]).mean(-1)
    return torch.exp(-error / std**2)


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
