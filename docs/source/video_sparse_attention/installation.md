(vsa-installation)=

# 🔧 Installation

# Building from Source
We test our code on Pytorch 2.7.1 and CUDA>=12.8. We support H100 (via ThunderKittens) and RTX 4090 (via Triton) for VSA.

First, install C++20 for ThunderKittens:

```bash
cd csrc/attn/
sudo apt update
sudo apt install gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

sudo apt update
sudo apt install clang-11
```

Install VSA:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH} 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
git submodule update --init --recursive
python setup_vsa.py install
```

# 🧪 Test

```bash
python tests/test_block_sparse.py # test VSA
```

# 📋 Usage

## Basic Video Sparse Attention

```python
from vsa import video_sparse_attn

# Basic usage for video sparse attention
# q, k, v: [batch_size, num_heads, seq_len, head_dim]
# topk: number of top blocks to attend to
# block_size: size of each block (int or tuple of 3 ints for T, H, W)
out = video_sparse_attn(q, k, v, topk=16, block_size=64)
```

## Block Sparse Attention

```python
from vsa import block_sparse_attn

# For custom sparse patterns
# q, k, v: [batch_size, num_heads, seq_len, head_dim]  
# q2k_block_sparse_index: indices for query-to-key sparse blocks
# q2k_block_sparse_num: number of sparse blocks for each query block
# k2q_block_sparse_index: indices for key-to-query sparse blocks (for backward pass)
# k2q_block_sparse_num: number of sparse blocks for each key block (for backward pass)
out = block_sparse_attn(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, 
                       k2q_block_sparse_index, k2q_block_sparse_num)
```

## Hardware Support

- **H100**: Full support via ThunderKittens CUDA kernels
- **RTX 4090**: Support via Triton implementation
- **Block size**: Must be multiple of 64 and >= 64 elements