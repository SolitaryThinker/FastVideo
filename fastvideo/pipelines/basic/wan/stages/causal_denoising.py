# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video, pred_noise_to_x_bound
from fastvideo.pipelines.basic.wan.stages.sampling import request_scheduler
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

try:
    from fastvideo.attention.backends.video_sparse_attn import (VideoSparseAttentionBackend)
    vsa_available = True
except ImportError:
    vsa_available = False
    VideoSparseAttentionBackend = None  # type: ignore

logger = init_logger(__name__)


def _get_transformer_attr(transformer, name: str, default):
    # Currently used only for causal KV-cache runtime settings:
    # `local_attn_size` and `sink_size`.
    #
    # Causal Wan/MatrixGame expose these directly on the transformer. Some
    # wrapper-style DiTs keep the executable module under transformer.model. Do
    # not fall back to transformer.config for these two values: config is only
    # the construction source and does not control them at runtime.
    value = getattr(transformer, name, None)
    if value is not None:
        return value

    inner_model = getattr(transformer, "model", None)
    if inner_model is not None:
        value = getattr(inner_model, name, None)
        if value is not None:
            return value

    return default


@dataclass
class WanCausalState:
    """One request's spatial cache layout and attention history."""
    frame_seq_length: int
    kv_cache: list[dict]
    crossattn_cache: list[dict]
    low_noise_kv_cache: list[dict] | None = None


class WanCausalDenoisingBase(DenoisingStage):
    """Shared Wan cache layout; standard and DMD sampling remain separate algorithms."""

    def __init__(self, transformer, scheduler, transformer_2=None, vae=None) -> None:
        super().__init__(transformer, scheduler, transformer_2=transformer_2)
        # Model-dependent constants (aligned with causal_inference.py assumptions)
        self.num_transformer_blocks = len(self.transformer.blocks)
        self.num_frames_per_block = self.transformer.config.arch_config.num_frames_per_block
        self.sliding_window_num_frames = self.transformer.config.arch_config.sliding_window_num_frames

        self.local_attn_size = _get_transformer_attr(self.transformer, "local_attn_size", -1)
        self.sink_size = _get_transformer_attr(self.transformer, "sink_size", 0)

    def forward(self, batch, fastvideo_args):
        with request_scheduler(batch, self.scheduler) as scheduler:
            return self.denoise(batch, fastvideo_args, scheduler)

    def prepare_caches(self, batch, fastvideo_args, dtype, *, two_experts=False) -> WanCausalState:
        latents = batch.latents
        assert latents is not None, "latents must be provided"
        patch_size = self.transformer.config.arch_config.patch_size
        frame_seq_length = latents.shape[-1] * latents.shape[-2] // (patch_size[-1] * patch_size[-2])
        cache_kwargs = dict(batch_size=latents.shape[0], dtype=dtype, device=latents.device)
        return WanCausalState(
            frame_seq_length=frame_seq_length,
            kv_cache=self._initialize_kv_cache(frame_seq_length, **cache_kwargs),
            low_noise_kv_cache=self._initialize_kv_cache(frame_seq_length, **cache_kwargs) if two_experts else None,
            crossattn_cache=self._initialize_crossattn_cache(
                max_text_len=fastvideo_args.pipeline_config.text_encoder_configs[0].arch_config.text_len,
                **cache_kwargs),
        )

    def _initialize_kv_cache(self, frame_seq_length, batch_size, dtype, device) -> list[dict]:
        """
        Initialize a Per-GPU KV cache aligned with the Wan model assumptions.
        """
        kv_cache1 = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = self.transformer.attention_head_dim
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * frame_seq_length
        else:
            kv_cache_size = frame_seq_length * self.sliding_window_num_frames

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k":
                torch.zeros([batch_size, kv_cache_size, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([batch_size, kv_cache_size, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            })

        return kv_cache1

    def _initialize_crossattn_cache(self, batch_size, max_text_len, dtype, device) -> list[dict]:
        """
        Initialize a Per-GPU cross-attention cache aligned with the Wan model assumptions.
        """
        crossattn_cache = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = self.transformer.attention_head_dim
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k":
                torch.zeros([batch_size, max_text_len, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([batch_size, max_text_len, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "is_init":
                False,
            })
        return crossattn_cache

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify denoising stage inputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check("image_latent", batch.image_latent, V.none_or_tensor_with_dims(5))
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        result.add_check("guidance_scale", batch.guidance_scale, V.positive_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("do_classifier_free_guidance", batch.do_classifier_free_guidance, V.bool_value)
        result.add_check("negative_prompt_embeds", batch.negative_prompt_embeds,
                         lambda x: not batch.do_classifier_free_guidance or V.list_not_empty(x))
        return result


class CausalDMDDenosingStage(WanCausalDenoisingBase):
    """
    Denoising stage for causal diffusion.
    """

    def denoise(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        scheduler,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # TODO(will): make this a parameter once we add i2v support
        independent_first_frame = self.transformer.independent_first_frame if hasattr(
            self.transformer, 'independent_first_frame') else False
        # Timesteps for DMD
        timesteps = torch.tensor(fastvideo_args.pipeline_config.dmd_denoising_steps, dtype=torch.long).cpu()
        if fastvideo_args.pipeline_config.warp_denoising_step:
            scheduler_timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(get_local_torch_device())

        if fastvideo_args.pipeline_config.dit_config.boundary_ratio is not None:
            boundary_timestep = fastvideo_args.pipeline_config.dit_config.boundary_ratio * scheduler.num_train_timesteps
            high_noise_timesteps = timesteps[timesteps >= boundary_timestep]
        else:
            boundary_timestep = None
            high_noise_timesteps = None

        # Image kwargs (kept empty unless caller provides compatible args)
        image_kwargs: dict = {}

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                # "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        # Latents and prompts
        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents  # [B, C, T, H, W]
        b, c, t, h, w = latents.shape
        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0

        # Initialize or reset caches
        state = self.prepare_caches(batch, fastvideo_args, target_dtype, two_experts=boundary_timestep is not None)
        kv_cache1, kv_cache2 = state.kv_cache, state.low_noise_kv_cache

        def _get_kv_cache(timestep: float) -> list[dict]:
            if boundary_timestep is not None:
                if timestep >= boundary_timestep:
                    return kv_cache1
                else:
                    assert kv_cache2 is not None, "kv_cache2 is not initialized"
                    return kv_cache2
            return kv_cache1

        crossattn_cache = state.crossattn_cache

        pos_start_base = 0

        # Determine block sizes
        if t % self.num_frames_per_block != 0:
            raise ValueError("num_frames must be divisible by num_frames_per_block for causal DMD denoising")
        num_blocks = t // self.num_frames_per_block
        block_sizes = [self.num_frames_per_block] * num_blocks
        start_index = 0

        # For now hardcode the first block to be 1 frame assuming the model is Wan2.2-MoE
        if boundary_timestep is not None:
            block_sizes[0] = 1

        first_frame_latent = batch.first_frame_latent
        if batch.pil_image is not None:
            assert first_frame_latent is not None, "Run WanFirstFrameEncodingStage before causal DMD denoising"

            # Fill the low noise and high noise kv cache with first_frame_latent and timestep 0
            t_zero = torch.zeros([latents.shape[0], 1], device=latents.device, dtype=torch.long)
            with torch.autocast(device_type="cuda",
                                dtype=target_dtype,
                                enabled=autocast_enabled), \
                set_forward_context(current_timestep=0,
                                    attn_metadata=None,
                                    forward_batch=batch):
                self.transformer(
                    first_frame_latent.to(target_dtype),
                    prompt_embeds,
                    t_zero,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=(pos_start_base + start_index) * state.frame_seq_length,
                    start_frame=start_index,
                    **image_kwargs,
                    **pos_cond_kwargs,
                )
                if boundary_timestep is not None:
                    self.transformer_2(
                        first_frame_latent.to(target_dtype),
                        prompt_embeds,
                        t_zero,
                        kv_cache=kv_cache2,
                        crossattn_cache=crossattn_cache,
                        current_start=(pos_start_base + start_index) * state.frame_seq_length,
                        start_frame=start_index,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )

            start_index += 1
            block_sizes.pop(0)
            latents[:, :, :1, :, :] = first_frame_latent

        # DMD loop in causal blocks
        with self.progress_bar(total=len(block_sizes) * len(timesteps)) as progress_bar:
            for current_num_frames in block_sizes:
                current_latents = latents[:, :, start_index:start_index + current_num_frames, :, :]
                # use BTCHW for DMD conversion routines
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                video_raw_latent_shape = noise_latents_btchw.shape

                for i, t_cur in enumerate(timesteps):
                    if boundary_timestep is not None and t_cur < boundary_timestep:
                        current_model = self.transformer_2
                    else:
                        current_model = self.transformer
                    # Copy for pred conversion
                    noise_latents = noise_latents_btchw.clone()
                    latent_model_input = current_latents.to(target_dtype)

                    if batch.image_latent is not None and independent_first_frame and start_index == 0:
                        latent_model_input = torch.cat([latent_model_input, batch.image_latent.to(target_dtype)], dim=2)

                    # Prepare inputs
                    t_expand = t_cur.repeat(latent_model_input.shape[0])

                    # Attention metadata if needed
                    if (vsa_available and self.attn_backend == VideoSparseAttentionBackend):
                        attn_metadata_builder_cls = self.attn_backend.get_builder_cls()
                        if attn_metadata_builder_cls is not None:
                            attn_metadata_builder = attn_metadata_builder_cls()
                            attn_metadata = attn_metadata_builder.build(  # type: ignore
                                current_timestep=i,  # type: ignore
                                raw_latent_shape=(current_num_frames, h, w),  # type: ignore
                                patch_size=fastvideo_args.pipeline_config.dit_config.patch_size,  # type: ignore
                                VSA_sparsity=fastvideo_args.VSA_sparsity,  # type: ignore
                                device=get_local_torch_device(),  # type: ignore
                            )  # type: ignore
                            assert attn_metadata is not None, "attn_metadata cannot be None"
                        else:
                            attn_metadata = None
                    else:
                        attn_metadata = None

                    with torch.autocast(device_type="cuda",
                                        dtype=target_dtype,
                                        enabled=autocast_enabled), \
                        set_forward_context(current_timestep=i,
                                            attn_metadata=attn_metadata,
                                            forward_batch=batch):
                        # Run transformer; follow DMD stage pattern
                        t_expanded_noise = t_cur * torch.ones(
                            (latent_model_input.shape[0], 1), device=latent_model_input.device, dtype=torch.long)
                        pred_noise_btchw = current_model(
                            latent_model_input,
                            prompt_embeds,
                            t_expanded_noise,
                            kv_cache=_get_kv_cache(t_cur),
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) * state.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        ).permute(0, 2, 1, 3, 4)

                    # Convert pred noise to pred video with FM Euler scheduler utilities
                    if boundary_timestep is not None and t_cur >= boundary_timestep:
                        pred_video_btchw = pred_noise_to_x_bound(
                            pred_noise=pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=noise_latents.flatten(0, 1),
                            timestep=t_expand,
                            boundary_timestep=torch.ones_like(t_expand) * boundary_timestep,
                            scheduler=scheduler).unflatten(0, pred_noise_btchw.shape[:2])
                    else:
                        pred_video_btchw = pred_noise_to_pred_video(pred_noise=pred_noise_btchw.flatten(0, 1),
                                                                    noise_input_latent=noise_latents.flatten(0, 1),
                                                                    timestep=t_expand,
                                                                    scheduler=scheduler).unflatten(
                                                                        0, pred_noise_btchw.shape[:2])

                    if i < len(timesteps) - 1:
                        next_timestep = timesteps[i + 1] * torch.ones(
                            [1], dtype=torch.long, device=pred_video_btchw.device)
                        noise = torch.randn(video_raw_latent_shape,
                                            dtype=pred_video_btchw.dtype,
                                            generator=(batch.generator[0] if isinstance(batch.generator, list) else
                                                       batch.generator)).to(self.device)
                        noise_btchw = noise
                        if boundary_timestep is not None and i < len(high_noise_timesteps) - 1:
                            noise_latents_btchw = scheduler.add_noise_high(
                                pred_video_btchw.flatten(0, 1), noise_btchw.flatten(0, 1), next_timestep,
                                torch.ones_like(next_timestep) * boundary_timestep).unflatten(
                                    0, pred_video_btchw.shape[:2])
                        elif boundary_timestep is not None and i == len(high_noise_timesteps) - 1:
                            noise_latents_btchw = pred_video_btchw
                        else:
                            noise_latents_btchw = scheduler.add_noise(pred_video_btchw.flatten(0, 1),
                                                                      noise_btchw.flatten(0, 1),
                                                                      next_timestep).unflatten(
                                                                          0, pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
                    else:
                        current_latents = pred_video_btchw.permute(0, 2, 1, 3, 4)

                    if progress_bar is not None:
                        progress_bar.update()

                # Write back and advance
                latents[:, :, start_index:start_index + current_num_frames, :, :] = current_latents

                # Re-run with context timestep to update KV cache using clean context
                context_noise = getattr(fastvideo_args.pipeline_config, "context_noise", 0)
                t_context = torch.ones([latents.shape[0]], device=latents.device, dtype=torch.long) * int(context_noise)
                context_bcthw = current_latents.to(target_dtype)
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled), \
                    set_forward_context(current_timestep=0,
                                        attn_metadata=attn_metadata,
                                        forward_batch=batch):
                    t_expanded_context = t_context.unsqueeze(1)

                    if boundary_timestep is not None:
                        self.transformer_2(
                            context_bcthw,
                            prompt_embeds,
                            t_expanded_context,
                            kv_cache=kv_cache2,
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) * state.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )

                    self.transformer(
                        context_bcthw,
                        prompt_embeds,
                        t_expanded_context,
                        kv_cache=kv_cache1,
                        crossattn_cache=crossattn_cache,
                        current_start=(pos_start_base + start_index) * state.frame_seq_length,
                        start_frame=start_index,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )

                start_index += current_num_frames

        if boundary_timestep is not None:
            num_frames_to_remove = self.num_frames_per_block - 1
            latents = latents[:, :, :-num_frames_to_remove, :, :]

        batch.latents = latents
        return batch


class CausalDenoisingStage(WanCausalDenoisingBase):
    """Causal block-by-block denoising with standard multi-step
    flow matching (scheduler.step), not DMD few-step.

    Each block is fully denoised through all scheduler timesteps
    before moving to the next block.  After each block is denoised,
    the KV cache is updated with clean context so subsequent blocks
    can attend to prior clean frames.
    """

    def denoise(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        scheduler,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32 and not fastvideo_args.disable_autocast)

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        assert batch.latents is not None, ("latents must be provided")
        latents = batch.latents  # [B, C, T, H, W]
        b, c, t, h, w = latents.shape
        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0

        num_inference_steps = batch.num_inference_steps

        # Initialize caches
        state = self.prepare_caches(batch, fastvideo_args, target_dtype)
        kv_cache, crossattn_cache = state.kv_cache, state.crossattn_cache

        # Determine block sizes
        if t % self.num_frames_per_block != 0:
            raise ValueError("num_frames must be divisible by "
                             "num_frames_per_block")
        num_blocks = t // self.num_frames_per_block
        block_sizes = [self.num_frames_per_block] * num_blocks
        start_index = 0
        pos_start_base = 0

        context_noise = getattr(
            fastvideo_args.pipeline_config,
            "context_noise",
            0,
        )

        with self.progress_bar(total=len(block_sizes) * num_inference_steps) as progress_bar:
            for current_num_frames in block_sizes:
                current_latents = latents[
                    :,
                    :,
                    start_index:start_index + current_num_frames,
                    :,
                    :,
                ]

                # Reset scheduler per block so its multi-step
                # history starts fresh.
                scheduler.set_timesteps(
                    num_inference_steps,
                    device=latents.device,
                )
                timesteps = scheduler.timesteps

                for i, t_cur in enumerate(timesteps):
                    latent_model_input = current_latents.to(target_dtype)
                    t_expanded = t_cur * torch.ones(
                        (b, 1),
                        device=latents.device,
                        dtype=torch.long,
                    )

                    with (
                            torch.autocast(
                                device_type="cuda",
                                dtype=target_dtype,
                                enabled=autocast_enabled,
                            ),
                            set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                            ),
                    ):
                        # Transformer returns [B, C, T, H, W],
                        # permute to [B, T, C, H, W] then flatten
                        # B*T for the scheduler.
                        noise_pred = self.transformer(
                            latent_model_input,
                            prompt_embeds,
                            t_expanded,
                            kv_cache=kv_cache,
                            crossattn_cache=crossattn_cache,
                            current_start=((pos_start_base + start_index) * state.frame_seq_length),
                            start_frame=start_index,
                            **pos_cond_kwargs,
                        )

                    # Flatten [B,C,T,H,W] -> [B*T,C,H,W] for
                    # scheduler, then unflatten back.
                    nf = current_num_frames
                    noise_pred_flat = (noise_pred.permute(0, 2, 1, 3, 4).flatten(0, 1))
                    latents_flat = (current_latents.permute(0, 2, 1, 3, 4).flatten(0, 1))
                    updated_flat = scheduler.step(
                        noise_pred_flat,
                        t_cur,
                        latents_flat,
                        return_dict=False,
                    )[0]
                    current_latents = (updated_flat.unflatten(0, (b, nf)).permute(0, 2, 1, 3, 4))

                    if progress_bar is not None:
                        progress_bar.update()

                # Write denoised block back
                latents[
                    :,
                    :,
                    start_index:start_index + current_num_frames,
                    :,
                    :,
                ] = current_latents

                # Update KV cache with clean context
                t_context = torch.ones(
                    [b],
                    device=latents.device,
                    dtype=torch.long,
                ) * int(context_noise)
                context_bcthw = current_latents.to(target_dtype)
                with (
                        torch.autocast(
                            device_type="cuda",
                            dtype=target_dtype,
                            enabled=autocast_enabled,
                        ),
                        set_forward_context(
                            current_timestep=0,
                            attn_metadata=None,
                            forward_batch=batch,
                        ),
                ):
                    self.transformer(
                        context_bcthw,
                        prompt_embeds,
                        t_context.unsqueeze(1),
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=((pos_start_base + start_index) * state.frame_seq_length),
                        start_frame=start_index,
                        **pos_cond_kwargs,
                    )

                start_index += current_num_frames

        batch.latents = latents
        return batch
