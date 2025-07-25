(vsa-troubleshooting)=

# 🔧 Troubleshooting

## Common Installation Issues

### C++ Compiler Issues

**Problem**: Build fails with C++20 compiler errors
```bash
error: 'concept' does not name a type
```

**Solution**: Ensure you have gcc-11 or later installed and set as default:
```bash
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11
gcc --version  # Should show 11.x or later
```

### CUDA Path Issues

**Problem**: CUDA not found during compilation
```bash
nvcc: command not found
```

**Solution**: Set proper CUDA environment variables:
```bash
export CUDA_HOME=/usr/local/cuda-12.4  # Adjust version as needed
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

### ThunderKittens Submodule Issues

**Problem**: Missing ThunderKittens dependencies
```bash
fatal: No such file or directory: 'tk/include'
```

**Solution**: Initialize submodules properly:
```bash
git submodule update --init --recursive
ls tk/  # Should show ThunderKittens files
```

## Runtime Issues

### Import Errors

**Problem**: VSA module not found
```python
ImportError: No module named 'vsa'
```

**Solution**: Verify installation and Python path:
```bash
python -c "import vsa; print(vsa.__file__)"  # Should show module location
pip list | grep vsa  # Check if installed
```

### Hardware Compatibility

**Problem**: CUDA kernel launch failures
```bash
RuntimeError: CUDA error: invalid device function
```

**Solution**: Check GPU compatibility and fallback options:
```python
import torch
print(f"GPU: {torch.cuda.get_device_name()}")
major, minor = torch.cuda.get_device_capability()
print(f"Compute capability: {major}.{minor}")

# For non-H100 GPUs, ensure Triton fallback works
from vsa.block_sparse_attn_triton import attention_sparse
```

### Memory Issues

**Problem**: Out of memory errors
```bash
RuntimeError: CUDA out of memory
```

**Solution**: Adjust block size and sparsity parameters:
```python
# Reduce block size for lower memory usage
video_sparse_attn(q, k, v, topk=8, block_size=64)  # Instead of larger blocks

# Monitor memory usage
torch.cuda.empty_cache()
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Performance Issues

### Slow Performance

**Problem**: VSA slower than expected

**Solution**: Check configuration and hardware utilization:
```python
# Ensure optimal block sizes (multiples of 64)
assert block_elements % 64 == 0 and block_elements >= 64

# Profile to identify bottlenecks
import torch.profiler
with torch.profiler.profile() as prof:
    output = video_sparse_attn(q, k, v, topk=16, block_size=64)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Precision Issues

**Problem**: Poor output quality compared to dense attention

**Solution**: Adjust sparsity and validate precision:
```python
# Increase top-k for better approximation quality
video_sparse_attn(q, k, v, topk=32, block_size=64)  # Higher top-k

# Run precision validation
from vsa.tests.test_block_sparse import precision_metric
sim, l1, rmse = precision_metric(vsa_output, dense_output)
print(f"Cosine similarity: {sim:.4f}, L1 error: {l1:.4f}, RMSE: {rmse:.4f}")
```

## Getting Help

If you continue to experience issues:

1. **Check logs**: Enable verbose CUDA logging:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   export TORCH_LOGS="+dynamo,+distributed"
   ```

2. **Create minimal reproduction**: Test with simple inputs:
   ```python
   import torch
   from vsa import video_sparse_attn
   
   # Minimal test case
   q = torch.randn(1, 8, 1024, 64, dtype=torch.bfloat16, device="cuda")
   k = torch.randn(1, 8, 1024, 64, dtype=torch.bfloat16, device="cuda")
   v = torch.randn(1, 8, 1024, 64, dtype=torch.bfloat16, device="cuda")
   
   out = video_sparse_attn(q, k, v, topk=16, block_size=64)
   print(f"Output shape: {out.shape}")
   ```

3. **Report issues**: Include system information:
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU: {torch.cuda.get_device_name()}")
   ```