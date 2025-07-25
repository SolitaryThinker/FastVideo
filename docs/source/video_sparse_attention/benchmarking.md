(vsa-benchmarking)=

# 📊 Benchmarking and Performance

## Running Benchmarks

VSA includes comprehensive benchmarking tools to measure performance across different configurations:

```bash
# Run VSA benchmarks
python csrc/attn/tests/test_block_sparse.py
```

## Performance Characteristics

### Block Size Impact

VSA performance is heavily influenced by block size configuration:

- **Minimum block size**: 64 elements (hardware requirement)
- **Optimal block sizes**: Multiples of 64 for best GPU utilization
- **Memory vs. computation trade-off**: Larger blocks reduce overhead but may decrease sparsity benefits

### Hardware-Specific Optimizations

#### H100 (ThunderKittens)
- Utilizes custom CUDA kernels optimized for Hopper architecture
- Maximum performance for large-scale video generation
- Requires CUDA 12.4+ and C++20 compiler support

#### RTX 4090 (Triton)
- Fallback implementation using Triton for broader hardware support
- Good performance for development and smaller workloads
- More accessible for researchers with consumer hardware

## Precision Metrics

The benchmark suite includes precision validation against Flash Attention 2:

```python
# Precision metrics reported:
# - Cosine similarity
# - L1 relative error  
# - Root Mean Square Error (RMSE)
```

## Memory Usage

VSA significantly reduces memory usage compared to dense attention:

- **Memory scaling**: O(sparse_blocks) vs O(seq_len²) for dense attention
- **Block compression**: Average pooling reduces intermediate memory requirements
- **Gradient computation**: Efficient backward pass with sparse block indexing

## Typical Performance Gains

Expected speedups depend on:
- **Sequence length**: Longer sequences see greater benefits
- **Sparsity ratio**: Higher sparsity (lower top-k) improves performance
- **Block size**: Optimal block sizes balance overhead and sparsity
- **Hardware**: H100 typically shows 2-4x speedup over dense attention

## Configuration Guidelines

### For Memory-Limited Scenarios
```python
# Use smaller block sizes and higher sparsity
video_sparse_attn(q, k, v, topk=8, block_size=64)
```

### For Maximum Performance
```python
# Use larger block sizes with moderate sparsity  
video_sparse_attn(q, k, v, topk=16, block_size=(4, 8, 8))
```

### For Quality-Critical Applications
```python
# Use lower sparsity for better approximation quality
video_sparse_attn(q, k, v, topk=32, block_size=(2, 8, 8))
```

## Debugging and Profiling

Enable detailed profiling:

```bash
# Set environment variables for detailed timing
export CUDA_LAUNCH_BLOCKING=1
python tests/test_block_sparse.py --profile
```

Monitor memory usage:
```python
torch.cuda.memory_summary()  # Check GPU memory before/after VSA calls
```