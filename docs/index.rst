FastVideo Documentation
=======================

FastVideo is a high-performance framework for training and inference of video diffusion models. This documentation covers the complete API reference, training guides, and best practices.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   training_api
   data_preprocess

Getting Started
===============

FastVideo provides a comprehensive framework for video generation using diffusion models. The framework supports:

- **Multiple Model Architectures**: Mochi, Hunyuan Video, and custom models
- **Efficient Training**: LoRA fine-tuning, knowledge distillation, and adversarial training
- **Distributed Training**: Multi-GPU support with FSDP and sequence parallelism
- **High Performance**: Optimized inference engine with mixed precision support

Quick Start
-----------

Training a Model
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic training
   python fastvideo/train.py \
     --model_type mochi \
     --data_json_path /path/to/data.json \
     --pretrained_model_name_or_path /path/to/model \
     --output_dir ./outputs \
     --train_batch_size 4 \
     --max_train_steps 1000

   # LoRA fine-tuning
   python fastvideo/train.py \
     --model_type mochi \
     --data_json_path /path/to/data.json \
     --pretrained_model_name_or_path /path/to/model \
     --use_lora \
     --lora_rank 64 \
     --learning_rate 1e-5

Running Inference
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastvideo.v1.inference_engine import InferenceEngine
   from fastvideo.v1.inference_args import InferenceArgs
   
   # Create inference configuration
   args = InferenceArgs(
       model_type="mochi",
       model_path="/path/to/model",
       height=480,
       width=848,
       num_frames=25
   )
   
   # Generate video
   engine = InferenceEngine.create_engine(args)
   result = engine.run("A cat playing in the garden", args)

Installation
============

Requirements
------------

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- Additional dependencies in requirements.txt

Setup
-----

.. code-block:: bash

   # Clone repository
   git clone https://github.com/your-org/fastvideo.git
   cd fastvideo
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install in development mode
   pip install -e .

Key Components
==============

Training Framework
------------------

The training framework provides:

- **Standard Training**: Flow matching based training for video diffusion models
- **Knowledge Distillation**: Distill knowledge from larger teacher models
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- **Distributed Training**: Scale training across multiple GPUs

Model Components
----------------

- **DiT Models**: Diffusion Transformer architectures for video generation
- **VAE Models**: Video Variational Autoencoders for latent space encoding
- **Text Encoders**: Various text encoders (CLIP, T5, Llama) for conditioning
- **Schedulers**: Noise schedulers for the diffusion process

Pipeline System
---------------

Modular pipeline system with stages:

- Input validation and preprocessing
- Text encoding and conditioning  
- Latent preparation and denoising
- VAE decoding to pixel space

Distributed Computing
---------------------

Advanced distributed training features:

- **FSDP**: Fully Sharded Data Parallel for large models
- **Sequence Parallelism**: Parallel processing of long video sequences
- **Tensor Parallelism**: Distribute model parameters across devices
- **Mixed Precision**: Efficient training with bf16/fp16

Performance Features
====================

Memory Optimization
-------------------

- Gradient checkpointing for reduced memory usage
- CPU offloading for large models
- Efficient attention mechanisms
- Optimized data loading pipelines

Training Acceleration
---------------------

- TF32 support for Ampere GPUs
- Optimized CUDA kernels
- Efficient dataloader implementations
- Smart batch scheduling

Examples and Tutorials
======================

Training Examples
-----------------

Basic Fine-tuning
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python fastvideo/train.py \
     --model_type hunyuan_hf \
     --data_json_path ./data/training_data.json \
     --pretrained_model_name_or_path hunyuan-video \
     --output_dir ./checkpoints \
     --train_batch_size 2 \
     --max_train_steps 5000 \
     --learning_rate 1e-5 \
     --gradient_checkpointing \
     --mixed_precision bf16

Knowledge Distillation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python fastvideo/distill.py \
     --model_type mochi \
     --data_json_path ./data/training_data.json \
     --pretrained_model_name_or_path mochi-teacher \
     --output_dir ./distilled_model \
     --num_euler_timesteps 4 \
     --multi_phased_distill_schedule "1000-2,2000-4,3000-8" \
     --use_ema \
     --ema_decay 0.999

Distributed Training
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   torchrun --nproc_per_node=8 fastvideo/train.py \
     --model_type mochi \
     --data_json_path ./data/training_data.json \
     --pretrained_model_name_or_path mochi-1b \
     --output_dir ./distributed_training \
     --train_batch_size 1 \
     --gradient_accumulation_steps 8 \
     --sp_size 2 \
     --use_cpu_offload \
     --fsdp_sharding_startegy full

API Reference
=============

For detailed API documentation, see:

- :doc:`training_api` - Complete training APIs and scripts reference
- :doc:`data_preprocess` - Data preprocessing documentation

Community
=========

- **GitHub**: https://github.com/your-org/fastvideo
- **Issues**: Report bugs and feature requests
- **Discussions**: Community support and discussions

Contributing
============

We welcome contributions! Please see our contributing guidelines for:

- Code style and formatting
- Testing requirements  
- Documentation standards
- Pull request process

License
=======

FastVideo is released under the Apache 2.0 License. See LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`