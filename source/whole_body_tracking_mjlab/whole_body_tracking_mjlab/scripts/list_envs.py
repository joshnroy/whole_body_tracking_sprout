"""List registered tasks, including this package's tasks."""

from mjlab.scripts.list_envs import main

import whole_body_tracking_mjlab  # noqa: F401

if __name__ == "__main__":
    main()
