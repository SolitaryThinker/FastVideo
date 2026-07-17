# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
    run_image_to_video_similarity_test,
)
logger = init_logger(__name__)

REQUIRED_GPUS = 4

device_reference_folder = resolve_inference_device_reference_folder(logger)

_MODEL_REPO = "robbyant/lingbot-world-v2-14b-causal-fast"
_MODEL_REVISION = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASET_DIR = _REPO_ROOT / "examples" / "dataset" / "lingbotworld2"
_CONVERTER = _REPO_ROOT / "scripts" / "checkpoint_conversion" / "convert_lingbotworld2_causal_fast.py"

LINGBOTWORLD2_MODEL_TO_PARAMS = {
    "robbyant__lingbot-world-v2-14b-causal-fast": {
        "model_path": _MODEL_REPO,
        "num_gpus": 4,
        "sp_size": 4,
        "tp_size": 1,
        "height": 480,
        "width": 832,
        "num_frames": 33,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": 42,
        "fps": 16,
    },
}

FULL_QUALITY_LINGBOTWORLD2_MODEL_TO_PARAMS = {
    "robbyant__lingbot-world-v2-14b-causal-fast": {
        **LINGBOTWORLD2_MODEL_TO_PARAMS["robbyant__lingbot-world-v2-14b-causal-fast"],
        "num_frames": 65,
    },
}

_PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains "
    "under a bright blue sky with drifting white clouds; gentle ripples reflect the tree and sky, creating a "
    "tranquil, meditative atmosphere."
)


def _prepare_model_bundle(root: Path) -> Path:
    source = Path(snapshot_download(repo_id=_MODEL_REPO, revision=_MODEL_REVISION))
    bundle = root / "model"
    subprocess.run(
        [sys.executable, str(_CONVERTER), "--source-dir", str(source), "--output-dir", str(bundle)],
        check=True,
    )
    return bundle


@pytest.mark.parametrize("prompt", [_PROMPT])
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(LINGBOTWORLD2_MODEL_TO_PARAMS))
def test_lingbotworld2_causal_fast_similarity(
    prompt: str,
    attention_backend_name: str,
    model_id: str,
    tmp_path: Path,
) -> None:
    model_path = _prepare_model_bundle(tmp_path)
    default_params = {
        model_id: {
            **LINGBOTWORLD2_MODEL_TO_PARAMS[model_id],
            "model_path": str(model_path),
        },
    }
    full_quality_params = {
        model_id: {
            **FULL_QUALITY_LINGBOTWORLD2_MODEL_TO_PARAMS[model_id],
            "model_path": str(model_path),
        },
    }

    run_image_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        image_path=str(_DATASET_DIR / "image.jpg"),
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=default_params,
        full_quality_params_map=full_quality_params,
        min_acceptable_ssim=0.97,
        init_kwargs_override={
            "hsdp_shard_dim": 4,
            "override_pipeline_cls_name": "LingBotWorld2CausalFastPipeline",
        },
        generation_kwargs_override={
            "action_path": str(_DATASET_DIR),
            "negative_prompt": "",
        },
    )
