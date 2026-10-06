"""MDP terms for BeyondMimic tracking on mjlab.

The motion command, observations, rewards and terminations come from mjlab's port of this repository
(``mjlab.tasks.tracking.mdp``). Terms that mjlab does not provide are defined here.
"""

from mjlab.tasks.tracking.mdp import *  # noqa: F401, F403

from .rewards import *  # noqa: F401, F403
