# SPDX-License-Identifier: Apache-2.0
"""Wan's public request boundary, without weights or a legacy sampling hop."""

from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from fastvideo.api.compat import normalize_generation_request, request_to_batch, request_to_sampling_param
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.models.wan.pipeline_config import WanT2V480PConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.tests.api.test_wan_definitions import LEGACY_VARIANTS
from fastvideo.utils import shallow_asdict

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@pytest.mark.parametrize("model_path", [case[0] for case in LEGACY_VARIANTS])
@pytest.mark.parametrize("overrides", [{}, {"sampling": {"num_frames": 1, "guidance_scale": 1.0},
                                         "output": {"save_video": False, "return_frames": False}}])
def test_direct_batch_matches_legacy_defaults_and_explicit_overrides(model_path, overrides):
    request = normalize_generation_request({"prompt": "test", **overrides})
    legacy = ForwardBatch(**shallow_asdict(request_to_sampling_param(request, model_path=model_path)))
    legacy.prompt = request.prompt
    direct = request_to_batch(request, model_path=model_path)
    for field in fields(ForwardBatch):
        if field.name != "logging_info":
            assert getattr(direct, field.name) == getattr(legacy, field.name), field.name


def _generator(tmp_path):
    generator = VideoGenerator.__new__(VideoGenerator)
    generator.fastvideo_args = SimpleNamespace(
        model_path=MODEL, prompt_txt=None, workload_type=WorkloadType.T2V,
        pipeline_config=WanT2V480PConfig(), VSA_sparsity=0.125,
        output_type="latent", pin_cpu_memory=False,
    )
    calls = []

    def execute(batch, args):
        calls.append(deepcopy(batch))
        # Simulate a pipeline that mutates its transport batch before returning.
        batch.height = 160
        batch.output = torch.zeros(1, 2, 1, 2, 2)
        batch.extra["worker_only"] = True
        if batch.prompt == "fail":
            raise RuntimeError("injected request failure")
        return batch

    generator.executor = SimpleNamespace(execute_forward=execute)
    request = {
        "prompt": "  a bird  ", "negative_prompt": "  blur  ",
        "sampling": {"num_frames": 9, "height": 32, "width": 48, "num_inference_steps": 3, "seed": 17},
        "output": {"output_path": str(tmp_path), "save_video": False, "return_frames": False},
        "extensions": {"vsa_mode": "dense"},
    }
    return generator, calls, request


@pytest.mark.parametrize("api", ["mapping", "typed", "legacy_kwargs", "legacy_sampling"])
def test_public_wan_apis_bypass_sampling_round_trip(monkeypatch, tmp_path, api):
    generator, calls, request = _generator(tmp_path)

    def fail(*args, **kwargs):
        pytest.fail("Wan generation must not construct or update legacy sampling parameters")

    monkeypatch.setattr(SamplingParam, "from_pretrained", fail)
    monkeypatch.setattr(SamplingParam, "update", fail)
    if api in {"mapping", "typed"}:
        result = generator.generate(normalize_generation_request(request) if api == "typed" else request)
        assert result.samples is None
    else:
        kwargs = {**request["sampling"], **request["output"], "negative_prompt": request["negative_prompt"]}
        with pytest.warns(DeprecationWarning):
            if api == "legacy_sampling":
                result = generator.generate_video(request["prompt"], SamplingParam(**kwargs), vsa_mode="dense")
            else:
                result = generator.generate_video(request["prompt"], **kwargs, vsa_mode="dense")
        assert result["samples"] is None
    assert len(calls) == 1
    batch = calls[0]
    assert (batch.prompt, batch.negative_prompt) == ("a bird", "blur")
    assert (batch.height, batch.width, batch.num_frames, batch.n_tokens) == (32, 48, 9, 72)
    assert (batch.seed, batch.num_inference_steps, batch.VSA_sparsity) == (17, 3, 0.125)
    assert batch.extra == {"vsa_mode": "dense"}
    assert request["sampling"]["height"] == 32


def test_reused_request_and_prompt_fanout_do_not_share_worker_state(tmp_path):
    generator, calls, raw = _generator(tmp_path)
    request = normalize_generation_request(raw)
    generator.generate(request)
    request.prompt = "fail"
    request.sampling.height = 48
    with pytest.raises(RuntimeError, match="injected request failure"):
        generator.generate(request)
    request.prompt = ["first", "second"]
    request.inputs.image_path = ["first.png", "second.png"]
    results = generator.generate(request)
    assert [result.prompt_index for result in results] == [0, 1]
    assert [batch.image_path for batch in calls[-2:]] == ["first.png", "second.png"]
    assert [batch.height for batch in calls] == [32, 48, 48, 48]
    assert all("worker_only" not in batch.extra for batch in calls)
    assert request.inputs.image_path == ["first.png", "second.png"]


def test_prompt_file_continues_after_failure_with_fresh_batches(tmp_path):
    generator, calls, request = _generator(tmp_path)
    prompt_path = tmp_path / "prompts.txt"
    prompt_path.write_text("first\n\nfail\nlast\n")
    request["inputs"] = {"prompt_path": str(prompt_path)}
    results = generator.generate(request)
    assert [result.prompt_index for result in results] == [0, 2]
    assert [batch.prompt for batch in calls] == ["first", "fail", "last"]
    assert all(batch.height == 32 for batch in calls)
    assert all("worker_only" not in batch.extra for batch in calls)


@pytest.mark.parametrize("update, message", [
    ({"sampling": {"height": 0}}, "must be positive"),
    ({"extensions": {"unknown_option": True}}, "not supported"),
    ({"inputs": {"prompt_path": "prompts.csv"}}, "must be a txt file"),
])
def test_invalid_request_never_reaches_executor(tmp_path, update, message):
    generator, calls, request = _generator(tmp_path)
    for section, values in update.items():
        request.setdefault(section, {}).update(values)
    with pytest.raises(ValueError, match=message):
        generator.generate(request)
    assert not calls
