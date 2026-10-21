"""MDP terms for BeyondMimic tracking on mjlab.

The motion command, observations, rewards and terminations come from mjlab's port of this repository
(``mjlab.tasks.tracking.mdp``). Terms that mjlab does not provide are defined here, along with
``MultiMotionCommand``, which extends mjlab's motion command to several reference motions.
"""

from mjlab.tasks.tracking.mdp import *  # noqa: F401, F403

from .multi_motion import MultiMotionCommand, MultiMotionCommandCfg, MultiMotionLoader  # noqa: F401
from .rewards import *  # noqa: F401, F403
