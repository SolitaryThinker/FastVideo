# SPDX-License-Identifier: Apache-2.0
"""Request-owned scheduling shared by Wan's separate sampling algorithms."""

from contextlib import contextmanager
from copy import deepcopy

from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage


class WanTimestepPreparationStage(TimestepPreparationStage):
    """Keep the reusable scheduler template free of request timesteps/history."""

    def forward(self, batch, fastvideo_args):
        batch.scheduler = deepcopy(self.scheduler)
        return super().forward(batch, fastvideo_args)


@contextmanager
def request_scheduler(batch, template, *, training_noise=False):
    """Own a prepared schedule (or a full DMD table) for one denoising call.

    Direct component callers may prepare the template themselves; copy that
    prepared schedule without recomputing timesteps. Dense DMD deliberately
    ignores the inference schedule and uses its separate training-noise table.
    """
    if training_noise or batch.scheduler is None:
        batch.scheduler = deepcopy(template)
    try:
        yield batch.scheduler
    finally:
        # Do not retain UniPC's GPU history or serialize it back to the caller,
        # including when a transformer/scheduler step fails.
        batch.scheduler = None
