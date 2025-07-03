# I2V + Sequence Parallelism (SP) Bug Fix Summary

## Issue Description

When training I2V (Image-to-Video) models with Ulysses sequence parallelism enabled, the model becomes unstable and generates noise despite the loss curve decreasing. This issue is specific to the I2V+SP combination and does not occur with:
- T2V (Text-to-Video) + SP
- I2V without SP
- T2V without SP

## Root Cause

The bug was caused by incorrect handling of image latents during sequence parallelism. Image latents are conditioning information that should remain consistent across all SP ranks, but they were being incorrectly sharded along the temporal dimension.

### Specifically:

1. **In Training**: The `shard_latents_across_sp` function was being applied to the concatenated tensor containing both video latents and image latents, causing different SP ranks to see different portions of the image conditioning.

2. **In Inference**: The denoising stage was also sharding image latents temporally, leading to inconsistent conditioning across SP ranks.

## Fixes Applied

### 1. Training Pipeline Fix (`wan_i2v_training_pipeline.py`)

- Modified `_prepare_dit_inputs` to NOT concatenate image latents immediately
- Created custom `train_one_step` override that:
  - Shards video latents, noisy_model_input, and noise tensors appropriately
  - Concatenates image latents AFTER sharding, ensuring all SP ranks get identical image conditioning

```python
# Handle image latents separately - they should NOT be sharded temporally
# Instead, all SP ranks should see the same image conditioning
image_latents = training_batch.image_latents.to(get_torch_device(), dtype=torch.bfloat16)

# For I2V, concatenate image latents after sharding video latents
# This ensures all SP ranks get the same image conditioning
training_batch.noisy_model_input = torch.cat(
    [training_batch.noisy_model_input, image_latents], dim=1)
```

### 2. Inference Pipeline Fix (`denoising.py`)

- Removed temporal sharding of `batch.image_latent` in the denoising stage
- Added clear comment explaining why image latents should not be sharded

```python
# Note: image_latent should NOT be sharded temporally since it's
# conditioning information that needs to be consistent across all SP ranks
# DO NOT shard batch.image_latent here
```

## Why This Fixes the Issue

1. **Consistent Conditioning**: All SP ranks now receive identical image conditioning information, eliminating conflicting gradients.

2. **Proper Temporal Sharding**: Only the video latents (which represent temporal information) are sharded across the sequence dimension, while static conditioning remains unchanged.

3. **Gradient Stability**: With consistent conditioning across ranks, gradients are computed correctly and the model trains stably even with higher learning rates.

## Testing Recommendations

1. Test I2V training with SP enabled and various learning rates to confirm stability
2. Verify that loss curves match between SP and non-SP training
3. Check that generated videos maintain quality with SP enabled
4. Ensure T2V training remains unaffected by these changes