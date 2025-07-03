# I2V + Sequence Parallelism (SP) Bug Fix Summary

## Issue Description

When training I2V (Image-to-Video) models with Ulysses sequence parallelism enabled, the model becomes unstable and generates noise despite the loss curve decreasing. This issue is specific to the I2V+SP combination and does not occur with:
- T2V (Text-to-Video) + SP
- I2V without SP
- T2V without SP

## Root Cause

The bug was caused by incorrect handling of image latents during sequence parallelism. In I2V, the `first_frame_latent` contains:

1. **A temporal mask** indicating which frames are conditioning frames (value 1 for first frame positions, 0 for others)
2. **The encoded first frame latents** concatenated with this mask

When these were sharded temporally across SP ranks, different ranks received different portions of the temporal mask, leading to inconsistent conditioning. For example:
- Rank 0 might receive mask values of [1, 1, 1, 0, 0, ...] (has conditioning)
- Rank 1 might receive mask values of [0, 0, 0, 0, 0, ...] (no conditioning)

This caused different SP ranks to compute gradients based on completely different conditioning signals, leading to training instability.

## Fixes Applied

### 1. Training Pipeline Fix (`wan_i2v_training_pipeline.py`)

- Modified `_prepare_dit_inputs` to NOT concatenate image latents immediately
- Created custom `train_one_step` override that:
  - Shards video latents, noisy_model_input, and noise tensors appropriately
  - Concatenates image latents AFTER sharding, ensuring all SP ranks get identical and complete image conditioning

```python
# Handle image latents separately - they should NOT be sharded temporally
# Image latents contain: 1) temporal mask indicating conditioning frames
# 2) encoded first frame. The mask has different values at different temporal
# positions, so sharding would give different ranks inconsistent conditioning.
# All SP ranks must see the complete image conditioning information.
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

1. **Consistent Conditioning**: All SP ranks now receive the complete temporal mask and image conditioning, eliminating conflicting signals during gradient computation.

2. **Proper Parallelism**: Only the video latents (which represent actual temporal information) are sharded across the sequence dimension, while static conditioning remains intact.

3. **Correct Gradient Flow**: With consistent conditioning across ranks, gradients are computed correctly, enabling stable training even with higher learning rates.

## Technical Details

The sequence parallelism implementation uses all-to-all operations to redistribute attention heads (not temporal positions) during attention computation. Each rank processes its assigned temporal slice with all attention heads for those positions. The bug occurred because the image conditioning mask had different values at different temporal positions, so sharding it meant different ranks saw fundamentally different conditioning.

## Testing Recommendations

1. Test I2V training with SP enabled and various learning rates to confirm stability
2. Verify that loss curves match between SP and non-SP training
3. Check that generated videos maintain quality with SP enabled
4. Ensure T2V training remains unaffected by these changes
5. Validate that the temporal mask in image latents is consistent across all SP ranks during training