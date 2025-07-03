# I2V + Sequence Parallelism (SP) Training Issue Analysis

## Issue Summary

When training I2V (Image-to-Video) models with Ulysses Sequence Parallelism (SP) enabled, the model quickly breaks and generates noise despite having lower loss curves that match smaller learning rate experiments. The issue only occurs with **I2V + SP combination** - T2V works fine with SP, and I2V works fine without SP.

## Root Cause Analysis

### The Problem: Inconsistent Image Conditioning Across SP Ranks

The core issue lies in the **order of operations** between image latent concatenation and SP sharding:

1. **I2V concatenation** (`wan_i2v_training_pipeline.py:117`):
   ```python
   training_batch.noisy_model_input = torch.cat(
       [training_batch.noisy_model_input, image_latents], dim=1)
   ```
   - Concatenates video latents with image latents along the **channel dimension (dim=1)**
   - This doubles the channel count: `[video_channels + image_channels, time, height, width]`

2. **SP sharding** (`training_pipeline.py:378`):
   ```python
   training_batch.noisy_model_input = shard_latents_across_sp(
       training_batch.noisy_model_input,
       num_latent_t=self.training_args.num_latent_t)
   ```
   - Shards the **already concatenated tensor** along the **temporal dimension (dim=2)**
   - Each SP rank gets a different temporal slice

3. **The `shard_latents_across_sp` function** (`training_utils.py:263-275`):
   ```python
   def shard_latents_across_sp(latents: torch.Tensor, num_latent_t: int) -> torch.Tensor:
       # ... 
       if sp_world_size > 1:
           latents = rearrange(latents, "b c (n s) h w -> b c n s h w", n=sp_world_size)
           latents = latents[:, :, rank_in_sp_group, :, :, :]  # Select rank's shard
       return latents
   ```

### Why This Breaks I2V Training

**For T2V**: Only video latents are present, so sharding works correctly - each SP rank processes a different temporal segment.

**For I2V**: After concatenation, the tensor contains `[video_channels, image_channels, time, height, width]`. When sharded:
- **Each SP rank gets different image latent channels** along with its video segment
- **Image conditioning becomes inconsistent** across SP ranks
- **The model receives conflicting conditioning signals** from different ranks

### Impact on Training Stability

- **Lower learning rates**: The inconsistency is small enough that the model can partially compensate
- **Higher learning rates**: The inconsistent conditioning gets amplified, causing:
  - Gradient conflicts between SP ranks
  - Model parameters diverging rapidly  
  - Generation of noise instead of coherent video

## Proposed Solutions

### Solution 1: Modify I2V Pipeline to Handle SP-Aware Concatenation

Modify `WanI2VTrainingPipeline._prepare_dit_inputs()` to:
1. Shard video latents first
2. Replicate image latents across all SP ranks (don't shard them)
3. Concatenate after sharding

```python
def _prepare_dit_inputs(self, training_batch: TrainingBatch) -> TrainingBatch:
    # Call parent to prepare video latents and noise
    training_batch = super()._prepare_dit_inputs(training_batch)
    
    # Shard video latents BEFORE concatenation
    training_batch.latents = shard_latents_across_sp(
        training_batch.latents, num_latent_t=self.training_args.num_latent_t)
    training_batch.noisy_model_input = shard_latents_across_sp(
        training_batch.noisy_model_input, num_latent_t=self.training_args.num_latent_t)
    training_batch.noise = shard_latents_across_sp(
        training_batch.noise, num_latent_t=self.training_args.num_latent_t)
    
    # Image latents should be same across all SP ranks (don't shard)
    image_latents = training_batch.image_latents.to(get_torch_device(), dtype=torch.bfloat16)
    
    # Concatenate AFTER sharding
    training_batch.noisy_model_input = torch.cat(
        [training_batch.noisy_model_input, image_latents], dim=1)
    
    return training_batch
```

### Solution 2: Create SP-Aware Sharding Function

Create a specialized sharding function for I2V that handles concatenated tensors:

```python
def shard_i2v_latents_across_sp(concatenated_latents: torch.Tensor, 
                                video_channels: int, 
                                num_latent_t: int) -> torch.Tensor:
    """Shard I2V latents by separating video and image components."""
    sp_world_size = get_sp_world_size()
    if sp_world_size <= 1:
        return concatenated_latents
    
    # Split concatenated tensor
    video_latents = concatenated_latents[:, :video_channels]
    image_latents = concatenated_latents[:, video_channels:]
    
    # Shard only video latents
    video_latents = shard_latents_across_sp(video_latents, num_latent_t)
    
    # Keep image latents unchanged (replicated across ranks)
    # Concatenate back
    return torch.cat([video_latents, image_latents], dim=1)
```

### Solution 3: Override Main Training Loop Sharding

Override the main training loop's sharding logic in I2V pipeline:

```python
def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
    # ... existing code ...
    
    # Custom sharding for I2V - handle video and image latents separately  
    training_batch.latents = shard_latents_across_sp(
        training_batch.latents, num_latent_t=self.training_args.num_latent_t)
    training_batch.noise = shard_latents_across_sp(
        training_batch.noise, num_latent_t=self.training_args.num_latent_t)
    
    # For noisy_model_input, we need special handling since it's concatenated
    # Skip the default sharding and let _prepare_dit_inputs handle it
    
    # ... rest of training logic ...
```

## Recommended Implementation

**Solution 1** is the cleanest approach as it:
- Localizes the fix to the I2V pipeline 
- Maintains compatibility with existing T2V code
- Ensures image conditioning consistency across SP ranks
- Requires minimal changes to the codebase

## Additional Considerations

1. **Validation Pipeline**: Ensure the validation pipeline handles SP consistently with training
2. **Memory Usage**: Replicating image latents across SP ranks increases memory usage slightly
3. **VSA Compatibility**: The dit_seq_shape calculation in attention backends may need updates
4. **Testing**: Test with various SP sizes (2, 4, 8) and learning rates to ensure stability

## Files to Modify

1. `fastvideo/v1/training/wan_i2v_training_pipeline.py` - Main fix
2. `fastvideo/v1/training/training_utils.py` - Optional: Add I2V-specific sharding function
3. `fastvideo/v1/training/training_pipeline.py` - Optional: Skip default sharding for I2V

This fix should resolve the I2V + SP training instability while maintaining performance and compatibility.