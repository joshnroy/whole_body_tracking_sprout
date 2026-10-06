"""Train a tracking policy: ``mjlab.scripts.train`` with this package's tasks registered.

Video recording is on by default: a clip is rendered every ``--video-interval`` env steps into the run's
``videos/train`` folder and uploaded to W&B by the rsl_rl logger. Pass ``--video False`` to turn it off.
"""

import dataclasses

import mjlab.scripts.train as _mjlab_train

import whole_body_tracking_mjlab  # noqa: F401

_from_task = _mjlab_train.TrainConfig.from_task


def from_task(task_id: str) -> _mjlab_train.TrainConfig:
    return dataclasses.replace(_from_task(task_id), video=True)


_mjlab_train.TrainConfig.from_task = staticmethod(from_task)

_run_train = _mjlab_train.run_train


def run_train(*args, **kwargs):
    # Multi-GPU workers unpickle this function by module path, which imports this module and registers the tasks
    # before mjlab looks them up. mjlab's own ``run_train`` would not.
    return _run_train(*args, **kwargs)


_mjlab_train.run_train = run_train


def main():
    _mjlab_train.main()


if __name__ == "__main__":
    main()
