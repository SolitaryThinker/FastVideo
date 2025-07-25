(vsa-overview)=

# 🎯 Video Sparse Attention (VSA) Overview

Video Sparse Attention (VSA) is an advanced attention mechanism designed specifically for accelerating video generation tasks. Unlike traditional dense attention that computes attention weights between all token pairs, VSA strategically selects only the most relevant blocks to attend to, dramatically reducing computational complexity while maintaining output quality.

## Key Features

- **Block-based Sparsity**: VSA operates on blocks of tokens rather than individual tokens, enabling efficient GPU utilization
- **Dual Hardware Support**: 
  - H100 GPUs via optimized ThunderKittens CUDA kernels
  - RTX 4090 GPUs via Triton implementation
- **Configurable Block Sizes**: Supports flexible block sizes (must be multiples of 64)
- **Video-Optimized**: Designed specifically for the temporal and spatial patterns in video data

## How VSA Works

VSA implements a two-stage attention mechanism:

1. **Compression Stage**: Video tokens are grouped into blocks and compressed using average pooling
2. **Sparse Selection**: Only the top-k most relevant blocks are selected for full attention computation
3. **Block Sparse Attention**: Full attention is computed only between the selected sparse blocks

This approach significantly reduces the quadratic complexity of traditional attention while preserving the most important spatial and temporal relationships in video data.

## Block Structure

VSA organizes video tokens into 3D blocks with dimensions `(T, H, W)` where:
- `T`: Temporal dimension (frames)
- `H`: Height dimension  
- `W`: Width dimension

Each block must contain at least 64 elements and the total elements must be a multiple of 64 for optimal GPU performance.

## Benefits

- **Memory Efficiency**: Reduces memory usage by focusing computation on relevant blocks
- **Speed**: Significantly faster than dense attention for long video sequences
- **Quality**: Maintains high output quality by preserving important spatial-temporal relationships
- **Scalability**: Enables processing of longer video sequences that would be infeasible with dense attention

## Use Cases

VSA is particularly effective for:
- Video generation models (T2V, I2V)
- Long video sequence processing
- Memory-constrained environments
- Applications requiring real-time video processing

The sparse attention pattern makes VSA especially suitable for video diffusion models where spatial locality and temporal coherence are important but global dense attention is computationally prohibitive.