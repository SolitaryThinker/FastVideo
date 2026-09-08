# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for causal Wan scheduling and cache lifetime."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import pytest

from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from fastvideo.tests.stages._denoising_fixtures import NullProgressBar, _patch_denoising_module
from fastvideo.tests.stages._denoising_fixtures import _args, _batch


class TinyCausalDenoiser(torch.nn.Module):
    hidden_size = 2
    num_attention_heads = 1
    attention_head_dim = 2
    local_attn_size = 3
    sink_size = 1

    def __init__(self):
        super().__init__()
        self.blocks = [None, None]
        self.config = SimpleNamespace(arch_config=SimpleNamespace(
            num_frames_per_block=3, sliding_window_num_frames=6, patch_size=(1, 1, 1)))
        self.calls = []

    def forward(self, latents, prompts, timestep, *, kv_cache, crossattn_cache, current_start, start_frame):
        self.calls.append((timestep.clone(), start_frame, kv_cache, crossattn_cache))
        return latents.float() * 0.125


class RecordingUniPC(FlowUniPCMultistepScheduler):

    def __init__(self):
        super().__init__(shift=3.0)
        self.resets = 0

    def set_timesteps(self, *args, **kwargs):
        self.resets += 1
        return super().set_timesteps(*args, **kwargs)


def test_causal_standard_resets_scheduler_per_block_and_caches_per_request(monkeypatch):
    _patch_denoising_module(monkeypatch, "1.0")
    from fastvideo.pipelines.basic.wan.stages import causal_denoising
    monkeypatch.setattr(causal_denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(causal_denoising, "set_forward_context", lambda **kwargs: nullcontext())
    model, scheduler = TinyCausalDenoiser(), RecordingUniPC()
    reset_schedulers = []
    set_timesteps = RecordingUniPC.set_timesteps

    def record_reset(scheduler, *args, **kwargs):
        reset_schedulers.append(scheduler)
        return set_timesteps(scheduler, *args, **kwargs)

    monkeypatch.setattr(RecordingUniPC, "set_timesteps", record_reset)
    stage = causal_denoising.CausalDenoisingStage(model, scheduler)
    stage.progress_bar = lambda **kwargs: NullProgressBar()
    args = _args()
    args.pipeline_config.text_encoder_configs = [SimpleNamespace(arch_config=SimpleNamespace(text_len=8))]
    args.pipeline_config.context_noise = 0

    outputs = []
    for _ in range(2):
        batch = _batch(steps=50, cfg=False)
        batch.latents = batch.latents.repeat(1, 1, 2, 1, 1)
        outputs.append(stage.forward(batch, args).latents.clone())

    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert scheduler.resets == 0  # Only request-local copies acquire history.
    assert len(reset_schedulers) == 4  # Each request resets both blocks.
    assert reset_schedulers[0] is reset_schedulers[1]
    assert reset_schedulers[2] is reset_schedulers[3]
    assert reset_schedulers[0] is not reset_schedulers[2]
    assert len(model.calls) == 2 * 2 * (50 + 1)  # Each block also writes its clean context to the cache.
    first_cache, first_cross = model.calls[0][2:]
    assert all(call[2] is first_cache and call[3] is first_cross for call in model.calls[:102])
    assert model.calls[102][2] is not first_cache
    assert model.calls[102][3] is not first_cross
    assert first_cache[0]["k"].shape == (1, 3 * 8, 1, 2)
    assert first_cross[0]["k"].shape == (1, 8, 1, 2)
    assert [model.calls[i][1] for i in (0, 50, 51, 101)] == [0, 0, 3, 3]
    assert model.calls[50][0].eq(0).all() and model.calls[101][0].eq(0).all()


@pytest.mark.parametrize("sampler", ["CausalDenoisingStage", "CausalDMDDenosingStage"])
def test_causal_cache_geometry_and_failure_are_request_local(monkeypatch, sampler):
    _patch_denoising_module(monkeypatch, "1.0")
    from fastvideo.pipelines.basic.wan.stages import causal_denoising
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    monkeypatch.setattr(causal_denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(causal_denoising, "set_forward_context", lambda **kwargs: nullcontext())
    model = TinyCausalDenoiser()
    scheduler = RecordingUniPC() if sampler == "CausalDenoisingStage" else FlowMatchEulerDiscreteScheduler(shift=5.0)
    stage = getattr(causal_denoising, sampler)(model, scheduler)
    stage.progress_bar = lambda **kwargs: NullProgressBar()
    args = _args()
    args.pipeline_config.text_encoder_configs = [SimpleNamespace(arch_config=SimpleNamespace(text_len=8))]
    args.pipeline_config.context_noise = 0
    args.pipeline_config.dmd_denoising_steps = [1000, 750, 500]
    args.pipeline_config.warp_denoising_step = False
    batches, caches, outputs = [], [], []
    for index, width in enumerate((4, 6, 4)):
        batch = _batch(steps=3, cfg=False)
        batch.latents = torch.full((1, 2, 3, 2, width), 0.25)
        batch.generator = [torch.Generator().manual_seed(123)]
        batches.append(batch)
        start = len(model.calls)
        original_forward = model.forward

        def forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            if index == 1 and len(model.calls) - start == 2:
                raise RuntimeError("interrupted causal block")
            return result

        model.forward = forward
        if index == 1:
            with pytest.raises(RuntimeError, match="interrupted causal block"):
                stage.forward(batch, args)
        else:
            outputs.append(stage.forward(batch, args).latents.clone())
        model.forward = original_forward
        caches.append(model.calls[start][2])
        assert batch.scheduler is None
        assert caches[-1][0]["k"].shape[1] == 3 * 2 * width
    assert len({id(cache) for cache in caches}) == 3
    assert not hasattr(stage, "frame_seq_length")
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
