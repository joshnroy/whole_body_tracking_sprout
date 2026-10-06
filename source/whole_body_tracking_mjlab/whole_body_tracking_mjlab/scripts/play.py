"""Play a tracking policy: ``mjlab.scripts.play`` with this package's tasks registered."""

from mjlab.scripts.play import main

import whole_body_tracking_mjlab  # noqa: F401

if __name__ == "__main__":
    main()
