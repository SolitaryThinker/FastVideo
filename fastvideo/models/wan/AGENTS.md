# Wan model family

This package owns the dense Wan transformer (`transformer.py`), its architecture,
FSDP predicates, and checkpoint/LoRA mappings (`config.py`), and the Wan VAE
(`vae.py`) with its config (`vae_config.py`). The VAE includes both the video
encoder and decoder. Shared text/image encoders, VAE utilities, pipelines,
pipeline configs, and training adapters remain in their existing directories.
Causal Wan and other families reuse these classes through compatibility imports.

## Invariants

- Keep `__init__.py` lightweight: no eager model or pipeline imports.
- Import configs directly from `fastvideo.models.wan.config` or
  `fastvideo.models.wan.vae_config` in the matching component.
- Preserve the old `models.dits.wanvideo` and `configs.models.dits.wanvideo`
  modules as explicit aliases, not subclasses or duplicate implementations.
- Keep `models.vaes.wanvae` and `configs.models.vaes.wanvae` as explicit aliases
  too, including the VAE's cache context variables and compile predicate.
- Keep `EntryClass = WanTransformer3DModel` in `transformer.py` only. Registry
  architecture names, state-dict keys, layer names, and mappings are compatibility
  contracts.
- Keep `EntryClass = AutoencoderKLWan` in `vae.py` only. Preserve latent
  normalization, first-frame handling, cache reset, streaming, tiling, and
  encoder/decoder compile conditions. Do not merge the separate Cosmos25,
  Gen3C, or LingBotWorld2 VAE adapters into this implementation.
- `config.py`, `vae_config.py`, and `__init__.py` are pre-commit checked;
  `transformer.py` and `vae.py` retain the existing model-code exclusion.
  Avoid unrelated reformatting.

## Focused checks

Follow the [testing guide](../../../docs/contributing/testing.md): cheap
compatibility checks first, then the smallest golden covering the changed
component, before default or full-quality renders. Imports may require GPU
dependencies even when a check needs no weights.

```bash
pytest fastvideo/tests/loader/test_wan_family_imports.py -q
pytest fastvideo/tests/contract/test_merge_ci_plan.py -q
pytest fastvideo/tests/vaes/test_wan_vae_compile.py -q
```

`fastvideo/tests/golden_gate/test_wan_t2v.py` checks dense transformer block 0,
not the VAE. Until a VAE golden exists, use `fastvideo/tests/vaes/test_wan_vae.py`
for numerical VAE changes; it requires one CUDA GPU and checkpoint access and
compares encode/decode against Diffusers plus streaming against full decode.
Use focused T2V/I2V SSIM for pipeline behavior not covered by component checks.
For a pure relocation, compare against the unchanged parent with identical
weights, settings, and runtime; aliases alone are not numerical parity evidence.
