# SPDX-License-Identifier: Apache-2.0
"""Real-import compatibility for the family-local Wan package.

Use an installed FastVideo environment with its runtime dependencies; these
checks do not download checkpoints or instantiate the full transformer.
"""

import importlib
import pickle
import subprocess
import sys

import pytest

from fastvideo.models.wan import config, transformer

TRANSFORMER_EXPORTS = (
    "EntryClass",
    "LayerNormScaleShift",
    "PatchEmbed",
    "WanI2VCrossAttention",
    "WanImageEmbedding",
    "WanSelfAttention",
    "WanT2VCrossAttention",
    "WanTimeTextImageEmbedding",
    "WanTransformer3DModel",
    "WanTransformerBlock",
    "WanTransformerBlock_VSA",
)
CONFIG_EXPORTS = ("WanVideoArchConfig", "WanVideoConfig", "is_blocks")


def test_legacy_transformer_exports_are_the_canonical_objects():
    legacy = importlib.import_module("fastvideo.models.dits.wanvideo")
    assert set(legacy.__all__) == set(TRANSFORMER_EXPORTS)
    for name in TRANSFORMER_EXPORTS:
        assert getattr(legacy, name) is getattr(transformer, name)
    assert legacy.EntryClass is transformer.WanTransformer3DModel


def test_legacy_and_aggregate_config_exports_are_the_canonical_objects():
    legacy = importlib.import_module("fastvideo.configs.models.dits.wanvideo")
    aggregate = importlib.import_module("fastvideo.configs.models.dits")
    assert set(legacy.__all__) == set(CONFIG_EXPORTS)
    for name in CONFIG_EXPORTS:
        assert getattr(legacy, name) is getattr(config, name)
    assert aggregate.WanVideoConfig is config.WanVideoConfig


@pytest.mark.parametrize("module_name", [
    "fastvideo.models.dits.causal_wanvideo",
    "fastvideo.models.dits.dreamx_world",
    "fastvideo.models.dits.matrixgame2.model",
    "fastvideo.models.dits.matrixgame2.causal_model",
    "fastvideo.models.dits.matrixgame3.model",
    "fastvideo.models.dits.lingbotworld.model",
    "fastvideo.configs.models.dits.dreamx_world",
    "fastvideo.configs.models.dits.matrixgame2",
    "fastvideo.configs.models.dits.matrixgame3",
    "fastvideo.configs.pipelines.wan",
])
def test_downstream_wan_consumers_import(module_name):
    importlib.import_module(module_name)


def test_registry_discovers_and_loads_the_canonical_wan_class(caplog):
    from fastvideo.models import registry

    expected = ("wan", "transformer", "WanTransformer3DModel")
    discovered = registry._discover_and_register_models()
    assert discovered["WanTransformer3DModel"] == expected
    assert registry._LEGACY_FAST_VIDEO_MODELS["WanTransformer3DModel"] == expected
    assert registry._FAST_VIDEO_MODELS["WanTransformer3DModel"] == expected
    assert "Duplicate architecture found: WanTransformer3DModel." not in caplog.text
    model_cls, architecture = registry.ModelRegistry.resolve_model_cls("WanTransformer3DModel")
    assert model_cls is transformer.WanTransformer3DModel
    assert architecture == "WanTransformer3DModel"


@pytest.mark.parametrize("first_module", [
    "fastvideo.models.wan.config",
    "fastvideo.models.wan.transformer",
    "fastvideo.configs.models.dits.wanvideo",
    "fastvideo.models.dits.wanvideo",
])
def test_import_order_and_config_transport_in_fresh_process(first_module):
    original = config.WanVideoConfig()
    # Protocol 0 GLOBAL references reproduce the old qualified names without
    # changing __module__ or keeping a second config implementation alive.
    serialized = pickle.dumps(original, protocol=0)
    legacy_serialized = serialized.replace(
        b"fastvideo.models.wan.config\n",
        b"fastvideo.configs.models.dits.wanvideo\n",
    )
    assert legacy_serialized != serialized
    script = """
import importlib
import pickle
import sys

importlib.import_module(sys.argv[1])
from fastvideo.models.wan.config import WanVideoArchConfig, WanVideoConfig, is_blocks
from fastvideo.configs.models.dits import WanVideoConfig as AggregateConfig
from fastvideo.models.wan.transformer import WanTransformer3DModel
from fastvideo.models.dits.wanvideo import WanTransformer3DModel as LegacyTransformer

assert AggregateConfig is WanVideoConfig
assert LegacyTransformer is WanTransformer3DModel
assert is_blocks('blocks.0', None)
assert not is_blocks('blocks.0.attn1', None)
for payload in pickle.load(sys.stdin.buffer):
    value = pickle.loads(payload)
    assert type(value) is WanVideoConfig
    assert type(value.arch_config) is WanVideoArchConfig
    assert value == WanVideoConfig()
    assert value.arch_config.hidden_size == 40 * 128
    assert value.arch_config._fsdp_shard_conditions == [is_blocks]
    assert pickle.loads(pickle.dumps(value)) == value
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_module],
        input=pickle.dumps((serialized, legacy_serialized)),
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
