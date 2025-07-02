======================
Training APIs Reference
======================

FastVideo Training Framework provides comprehensive APIs for training diffusion models for video generation. This documentation covers all training-related components in the ``fastvideo/v1/`` module.

Overview
========

The FastVideo training framework supports:

- **Standard Training**: Fine-tuning diffusion models with flow matching
- **Distillation**: Knowledge distillation from teacher models
- **Adversarial Training**: GAN-based adversarial training
- **LoRA Fine-tuning**: Parameter-efficient training with Low-Rank Adaptation
- **Distributed Training**: Multi-GPU training with FSDP and sequence parallelism

Core Training Scripts
=====================

Main Training Script
--------------------

.. py:module:: fastvideo.train

The main training script provides end-to-end training functionality for video diffusion models.

.. py:function:: main(args)

   Main training function that orchestrates the entire training process.
   
   :param argparse.Namespace args: Training configuration arguments
   
   **Key Features:**
   
   - Supports multiple model types: ``mochi``, ``hunyuan_hf``, ``hunyuan``
   - Distributed training with FSDP (Fully Sharded Data Parallel)
   - LoRA fine-tuning support
   - Gradient checkpointing for memory efficiency
   - Sequence parallelism for long video sequences
   - Mixed precision training (bf16/fp16)

.. py:function:: train_one_step(transformer, model_type, optimizer, lr_scheduler, loader, noise_scheduler, noise_random_generator, gradient_accumulation_steps, sp_size, precondition_outputs, max_grad_norm, weighting_scheme, logit_mean, logit_std, mode_scale)

   Executes a single training step with flow matching loss.
   
   :param transformer: The transformer model being trained
   :param str model_type: Type of model (mochi, hunyuan_hf, hunyuan)
   :param optimizer: PyTorch optimizer
   :param lr_scheduler: Learning rate scheduler
   :param loader: Data loader iterator
   :param noise_scheduler: Noise scheduler for diffusion process
   :param noise_random_generator: Random generator for noise
   :param int gradient_accumulation_steps: Number of gradient accumulation steps
   :param int sp_size: Sequence parallel size
   :param bool precondition_outputs: Whether to precondition model outputs
   :param float max_grad_norm: Maximum gradient norm for clipping
   :param str weighting_scheme: Timestep weighting scheme
   :param float logit_mean: Mean for logit normal weighting
   :param float logit_std: Standard deviation for logit normal weighting
   :param float mode_scale: Scale for mode weighting
   :returns: Tuple of (total_loss, grad_norm)
   :rtype: tuple[float, float]

.. py:function:: compute_density_for_timestep_sampling(weighting_scheme, batch_size, generator, logit_mean=None, logit_std=None, mode_scale=None)

   Computes density for timestep sampling during training.
   
   :param str weighting_scheme: Weighting scheme ("uniform", "logit_normal", "mode")
   :param int batch_size: Batch size
   :param generator: Random number generator
   :param float logit_mean: Mean for logit normal distribution
   :param float logit_std: Standard deviation for logit normal distribution  
   :param float mode_scale: Scale parameter for mode weighting
   :returns: Sampled timestep ratios
   :rtype: torch.Tensor

**Training Arguments:**

.. list-table::
   :header-rows: 1
   
   * - Argument
     - Type
     - Default
     - Description
   * - ``model_type``
     - str
     - "mochi"
     - Model type: mochi, hunyuan_hf, hunyuan
   * - ``data_json_path``
     - str
     - Required
     - Path to training data JSON file
   * - ``train_batch_size``
     - int
     - 16
     - Batch size per device
   * - ``max_train_steps``
     - int
     - None
     - Total training steps
   * - ``learning_rate``
     - float
     - 1e-4
     - Initial learning rate
   * - ``use_lora``
     - bool
     - False
     - Enable LoRA fine-tuning
   * - ``lora_rank``
     - int
     - 128
     - LoRA rank parameter
   * - ``sp_size``
     - int
     - 1
     - Sequence parallel size

**Usage Example:**

.. code-block:: bash

   python fastvideo/train.py \
     --model_type mochi \
     --data_json_path /path/to/data.json \
     --pretrained_model_name_or_path /path/to/model \
     --output_dir ./outputs \
     --train_batch_size 4 \
     --max_train_steps 1000 \
     --learning_rate 1e-5 \
     --use_lora \
     --lora_rank 64

Distillation Training
---------------------

.. py:module:: fastvideo.distill

Knowledge distillation framework for creating faster inference models.

.. py:function:: main(args)

   Main distillation training function.
   
   :param argparse.Namespace args: Distillation configuration arguments

.. py:function:: distill_one_step(transformer, model_type, teacher_transformer, ema_transformer, optimizer, lr_scheduler, loader, noise_scheduler, solver, noise_random_generator, gradient_accumulation_steps, sp_size, max_grad_norm, uncond_prompt_embed, uncond_prompt_mask, num_euler_timesteps, multiphase, not_apply_cfg_solver, distill_cfg, ema_decay, pred_decay_weight, pred_decay_type, hunyuan_teacher_disable_cfg)

   Executes one distillation training step.
   
   :param transformer: Student model being trained
   :param str model_type: Model type
   :param teacher_transformer: Teacher model (frozen)
   :param ema_transformer: EMA model (optional)
   :param optimizer: Optimizer for student model
   :param lr_scheduler: Learning rate scheduler
   :param loader: Data loader
   :param noise_scheduler: Noise scheduler
   :param solver: Euler solver for multi-step diffusion
   :param noise_random_generator: Random generator
   :param int gradient_accumulation_steps: Gradient accumulation steps
   :param int sp_size: Sequence parallel size
   :param float max_grad_norm: Maximum gradient norm
   :param torch.Tensor uncond_prompt_embed: Unconditional prompt embeddings
   :param torch.Tensor uncond_prompt_mask: Unconditional prompt mask
   :param int num_euler_timesteps: Number of Euler timesteps
   :param int multiphase: Number of phases for distillation
   :param bool not_apply_cfg_solver: Whether to skip CFG in solver
   :param float distill_cfg: CFG scale for distillation
   :param float ema_decay: EMA decay rate
   :param float pred_decay_weight: Prediction decay weight
   :param str pred_decay_type: Type of prediction decay (l1, l2)
   :param bool hunyuan_teacher_disable_cfg: Disable CFG for Hunyuan teacher
   :returns: Tuple of (loss, grad_norm, pred_norm)
   :rtype: tuple[float, float, dict]

**Key Features:**

- Multi-phase distillation for progressive training
- EMA (Exponential Moving Average) support
- Classifier-free guidance distillation
- Huber loss for robust training
- Teacher-student knowledge transfer

Adversarial Training
--------------------

.. py:module:: fastvideo.distill_adv

GAN-based adversarial training for enhanced video quality.

.. py:function:: gan_g_loss(fake_output, discriminator_weight, fake_output_32)

   Computes generator loss for adversarial training.
   
   :param torch.Tensor fake_output: Fake samples from generator
   :param torch.Tensor discriminator_weight: Discriminator weights
   :param torch.Tensor fake_output_32: Fake samples at 32x32 resolution
   :returns: Generator adversarial loss
   :rtype: torch.Tensor

.. py:function:: gan_d_loss(real_output, fake_output, discriminator_weight, real_output_32, fake_output_32, discriminator_weight_32)

   Computes discriminator loss for adversarial training.
   
   :param torch.Tensor real_output: Real samples
   :param torch.Tensor fake_output: Fake samples  
   :param torch.Tensor discriminator_weight: Discriminator weights
   :param torch.Tensor real_output_32: Real samples at 32x32
   :param torch.Tensor fake_output_32: Fake samples at 32x32
   :param torch.Tensor discriminator_weight_32: Discriminator weights for 32x32
   :returns: Discriminator adversarial loss
   :rtype: torch.Tensor

Core V1 API Components
======================

Inference Engine
----------------

.. py:module:: fastvideo.v1.inference_engine

.. py:class:: InferenceEngine

   Main inference engine for running trained diffusion models.

   .. py:method:: __init__(pipeline, inference_args)
   
      Initialize the inference engine.
      
      :param ComposedPipelineBase pipeline: Pipeline for inference
      :param InferenceArgs inference_args: Inference configuration

   .. py:classmethod:: create_engine(inference_args)
   
      Create an inference engine from arguments.
      
      :param InferenceArgs inference_args: Inference configuration
      :returns: Configured inference engine
      :rtype: InferenceEngine

   .. py:method:: run(prompt, inference_args)
   
      Run inference with the given prompt.
      
      :param str prompt: Text prompt for generation
      :param InferenceArgs inference_args: Inference parameters
      :returns: Dictionary containing generated videos and metadata
      :rtype: dict

**Usage Example:**

.. code-block:: python

   from fastvideo.v1.inference_engine import InferenceEngine
   from fastvideo.v1.inference_args import InferenceArgs
   
   # Create inference arguments
   args = InferenceArgs(
       model_type="mochi",
       model_path="/path/to/model",
       height=480,
       width=848,
       num_frames=25,
       num_inference_steps=50
   )
   
   # Create and run inference
   engine = InferenceEngine.create_engine(args)
   result = engine.run("A cat playing in the garden", args)

Model Components
----------------

DiT Models
~~~~~~~~~~

.. py:module:: fastvideo.v1.models.dits

.. py:class:: HunyuanVideoTransformer3DModel

   Hunyuan Video DiT (Diffusion Transformer) model implementation.
   
   **Key Features:**
   
   - 3D transformer architecture for video generation
   - Temporal and spatial attention mechanisms
   - Classifier-free guidance support
   - Rope positional embeddings
   - Multi-resolution training support

.. py:class:: WanVideoTransformer3DModel

   Wan Video DiT model for video generation tasks.

Text Encoders
~~~~~~~~~~~~~

.. py:module:: fastvideo.v1.models.text_encoder

.. py:class:: HunyuanTextEncoder

   Text encoder for Hunyuan video models.
   
   .. py:method:: encode(prompt, negative_prompt=None, max_sequence_length=256)
   
      Encode text prompts into embeddings.
      
      :param str prompt: Input text prompt
      :param str negative_prompt: Negative prompt (optional)
      :param int max_sequence_length: Maximum sequence length
      :returns: Text embeddings and attention masks
      :rtype: tuple[torch.Tensor, torch.Tensor]

VAE Models
~~~~~~~~~~

.. py:module:: fastvideo.v1.models.vaes

.. py:class:: WanVAE

   Video Variational Autoencoder for encoding/decoding video data.
   
   .. py:method:: encode(x)
   
      Encode video to latent space.
      
      :param torch.Tensor x: Input video tensor
      :returns: Latent representation
      :rtype: torch.Tensor
   
   .. py:method:: decode(z)
   
      Decode latent representation to video.
      
      :param torch.Tensor z: Latent tensor
      :returns: Reconstructed video
      :rtype: torch.Tensor

Pipeline System
---------------

.. py:module:: fastvideo.v1.pipelines

The pipeline system provides modular components for video generation.

Base Pipeline
~~~~~~~~~~~~~

.. py:class:: ComposedPipelineBase

   Base class for composed pipelines.
   
   .. py:method:: forward(batch, inference_args)
   
      Execute the pipeline forward pass.
      
      :param ForwardBatch batch: Input batch data
      :param InferenceArgs inference_args: Inference configuration
      :returns: Pipeline output
      :rtype: PipelineOutput

Pipeline Stages
~~~~~~~~~~~~~~~

.. py:module:: fastvideo.v1.pipelines.stages

The pipeline is composed of modular stages:

.. py:class:: InputValidationStage

   Validates and preprocesses input data.

.. py:class:: ClipTextEncodingStage

   Encodes text using CLIP text encoder.

.. py:class:: LlamaEncodingStage

   Encodes text using Llama-based encoder.

.. py:class:: LatentPreparationStage

   Prepares latent tensors for denoising.

.. py:class:: TimestepPreparationStage

   Prepares timestep schedules.

.. py:class:: DenoisingStage

   Main denoising stage using DiT models.
   
   .. py:method:: forward(batch, timesteps, latents, encoder_hidden_states, encoder_attention_mask)
   
      Perform denoising step.
      
      :param ForwardBatch batch: Input batch
      :param torch.Tensor timesteps: Timestep values
      :param torch.Tensor latents: Latent tensors
      :param torch.Tensor encoder_hidden_states: Text encoder outputs
      :param torch.Tensor encoder_attention_mask: Text attention masks
      :returns: Denoised latents
      :rtype: torch.Tensor

.. py:class:: DecodingStage

   Decodes latents back to pixel space using VAE.

.. py:class:: ConditioningStage

   Handles conditioning information for generation.

Distributed Training
--------------------

.. py:module:: fastvideo.v1.distributed

Parallel State Management
~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

   Initialize model parallelism.
   
   :param int tensor_model_parallel_size: Tensor parallel size
   :param int pipeline_model_parallel_size: Pipeline parallel size

.. py:function:: get_tensor_model_parallel_rank()

   Get current tensor model parallel rank.
   
   :returns: Current rank
   :rtype: int

.. py:function:: get_tensor_model_parallel_world_size()

   Get tensor model parallel world size.
   
   :returns: World size
   :rtype: int

Communication Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: fastvideo.v1.distributed.communication_op

.. py:function:: all_reduce(tensor, op=ReduceOp.SUM, group=None)

   All-reduce operation across distributed processes.
   
   :param torch.Tensor tensor: Input tensor
   :param ReduceOp op: Reduction operation
   :param ProcessGroup group: Process group
   :returns: Reduced tensor
   :rtype: torch.Tensor

Parameter Management
--------------------

.. py:module:: fastvideo.v1.models.parameter

.. py:class:: BasevLLMParameter(Parameter)

   Base parameter class for vLLM-style parameter loading.
   
   .. py:method:: __init__(data, weight_loader)
   
      Initialize parameter with weight loader.
      
      :param torch.Tensor data: Parameter data
      :param callable weight_loader: Weight loading function

.. py:class:: ModelWeightParameter(BasevLLMParameter)

   Parameter class for model weights supporting both column and row parallelism.

.. py:class:: PackedvLLMParameter(ModelWeightParameter)

   Parameter for packed model weights (e.g., quantized weights).
   
   .. py:method:: adjust_shard_indexes_for_packing(shard_size, shard_offset)
   
      Adjust shard indices for packed parameters.
      
      :param int shard_size: Size of shard
      :param int shard_offset: Offset of shard
      :returns: Adjusted shard size and offset
      :rtype: tuple[int, int]

CLI Interface
-------------

.. py:module:: fastvideo.v1.entrypoints.cli

Command Line Tools
~~~~~~~~~~~~~~~~~~

.. py:module:: fastvideo.v1.entrypoints.cli.generate

.. py:function:: main()

   Main CLI entry point for video generation.

.. py:module:: fastvideo.v1.entrypoints.cli.main

Main CLI dispatcher for various FastVideo operations.

**Usage Examples:**

.. code-block:: bash

   # Generate video using CLI
   python -m fastvideo.v1.entrypoints.cli.generate \
     --model_type mochi \
     --model_path /path/to/model \
     --prompt "A beautiful sunset over the ocean" \
     --output_path ./output.mp4

Utilities
=========

Logging
-------

.. py:module:: fastvideo.v1.logger

.. py:function:: init_logger(name, level=logging.INFO)

   Initialize logger for FastVideo components.
   
   :param str name: Logger name
   :param int level: Logging level
   :returns: Configured logger
   :rtype: logging.Logger

Forward Context
---------------

.. py:module:: fastvideo.v1.forward_context

.. py:class:: ForwardContext

   Context manager for forward pass configuration.
   
   .. py:method:: __enter__()
   
      Enter forward context.
   
   .. py:method:: __exit__(exc_type, exc_val, exc_tb)
   
      Exit forward context.

Environment Configuration
-------------------------

.. py:module:: fastvideo.v1.envs

Environment variable management for FastVideo configuration.

.. py:function:: get_env_var(name, default=None, type_func=str)

   Get environment variable with type conversion.
   
   :param str name: Environment variable name
   :param default: Default value if not found
   :param callable type_func: Type conversion function
   :returns: Environment variable value
   :rtype: Any

Best Practices
==============

Training Tips
-------------

1. **Memory Optimization:**
   
   - Use gradient checkpointing for large models
   - Enable CPU offloading for FSDP
   - Use mixed precision training (bf16)
   - Adjust batch size based on GPU memory

2. **Distributed Training:**
   
   - Use sequence parallelism for long videos
   - Balance tensor and data parallelism
   - Monitor memory usage across GPUs

3. **LoRA Fine-tuning:**
   
   - Start with rank 64-128 for most tasks
   - Adjust learning rate (typically 1e-5 to 1e-4)
   - Use target modules: ["to_k", "to_q", "to_v", "to_out.0"]

4. **Data Preprocessing:**
   
   - Precompute text embeddings and VAE latents
   - Use appropriate video resolutions (divisible by 8)
   - Ensure frame counts follow model requirements

Performance Optimization
------------------------

1. **Training Speed:**
   
   - Use multiple GPUs with FSDP
   - Enable TF32 for Ampere GPUs
   - Optimize dataloader workers
   - Use gradient accumulation for large effective batch sizes

2. **Memory Efficiency:**
   
   - Use CPU offloading for large models
   - Enable selective checkpointing
   - Monitor peak memory usage
   - Use efficient data loading

Troubleshooting
===============

Common Issues
-------------

1. **Out of Memory (OOM):**
   
   - Reduce batch size
   - Enable gradient checkpointing
   - Use CPU offloading
   - Check sequence parallel configuration

2. **Slow Training:**
   
   - Increase dataloader workers
   - Check data preprocessing overhead
   - Optimize sequence parallelism settings
   - Use mixed precision training

3. **Convergence Issues:**
   
   - Adjust learning rate
   - Check gradient clipping settings
   - Verify data quality
   - Monitor gradient norms

Configuration Examples
======================

Basic Training Configuration
----------------------------

.. code-block:: python

   # Basic training arguments
   training_config = {
       "model_type": "mochi",
       "data_json_path": "/path/to/training_data.json",
       "pretrained_model_name_or_path": "/path/to/pretrained_model",
       "output_dir": "./training_outputs",
       "train_batch_size": 4,
       "max_train_steps": 10000,
       "learning_rate": 1e-5,
       "gradient_accumulation_steps": 4,
       "gradient_checkpointing": True,
       "mixed_precision": "bf16",
       "use_cpu_offload": True,
       "checkpointing_steps": 500,
       "validation_steps": 100,
       "log_validation": True
   }

LoRA Fine-tuning Configuration
------------------------------

.. code-block:: python

   # LoRA fine-tuning setup
   lora_config = {
       "use_lora": True,
       "lora_rank": 128,
       "lora_alpha": 256,
       "lora_target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
       "learning_rate": 1e-4,
       "train_batch_size": 8,
       "gradient_accumulation_steps": 2
   }

Distillation Configuration
--------------------------

.. code-block:: python

   # Knowledge distillation setup
   distill_config = {
       "model_type": "mochi",
       "teacher_model_path": "/path/to/teacher_model",
       "num_euler_timesteps": 4,
       "multi_phased_distill_schedule": "2000-2,4000-4,6000-8",
       "distill_cfg": 7.5,
       "use_ema": True,
       "ema_decay": 0.999,
       "pred_decay_weight": 0.01,
       "pred_decay_type": "l2"
   }

.. note::
   
   This documentation covers the comprehensive training API for FastVideo v1. 
   For the latest updates and additional examples, refer to the project repository 
   and example scripts in the ``scripts/`` directory.