# SPDX-License-Identifier: Apache-2.0
"""Inference-specific composed pipeline implementation."""

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger

from .composed_pipeline_base import ComposedPipelineBase

logger = init_logger(__name__)


class InferencePipeline(ComposedPipelineBase):
    """Base class for inference pipelines built on top of composed pipelines."""

    def setup_pipeline(self) -> None:
        assert isinstance(self.fastvideo_args, FastVideoArgs)
        if getattr(self.fastvideo_args, "training_mode", False):
            raise RuntimeError(
                "InferencePipeline requires training_mode to be False."
            )

        self.initialize_pipeline(self.fastvideo_args)

        if self.fastvideo_args.enable_torch_compile:
            self.modules["transformer"] = torch.compile(
                self.modules["transformer"])
            logger.info("Torch Compile enabled for DiT")

        logger.info("Creating pipeline stages...")
        self.create_pipeline_stages(self.fastvideo_args)


__all__ = ["InferencePipeline"]
