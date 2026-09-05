# Wan model family

This package owns the dense Wan transformer (`transformer.py`) and its
architecture, FSDP predicates, and checkpoint/LoRA mappings (`config.py`).
Pipelines, pipeline configs, VAEs, encoders, and training adapters remain in
their existing directories. Causal Wan and other families reuse these classes
through compatibility imports.

## Invariants

- Keep `__init__.py` lightweight: no eager transformer or pipeline imports.
- Import the architecture config directly from `fastvideo.models.wan.config`.
- Preserve the old `models.dits.wanvideo` and `configs.models.dits.wanvideo`
  modules as explicit aliases, not subclasses or duplicate implementations.
- Keep `EntryClass = WanTransformer3DModel` in `transformer.py` only. Registry
  architecture names, state-dict keys, layer names, and mappings are compatibility
  contracts.
- `config.py` and `__init__.py` are pre-commit checked; `transformer.py` retains
  the existing model-code exclusion. Avoid unrelated reformatting.

## Focused checks

Run compatibility checks in an installed FastVideo environment (imports may
require GPU dependencies):

```bash
pytest fastvideo/tests/loader/test_wan_family_imports.py -q
pytest fastvideo/tests/contract/test_merge_ci_plan.py -q
FASTVIDEO_SSIM_MODEL_ID=Wan2.1-T2V-1.3B-Diffusers pytest fastvideo/tests/ssim/test_wan_t2v_similarity.py -vs
```

The SSIM test requires two GPUs and checkpoint/reference access. For a pure
relocation, compare output against the parent revision with identical weights,
settings, and runtime; comparing two aliases of the same class is not numerical
parity evidence.
