from dasa.config import DASAConfig
from dasa.token_selection import local_nxn_match, select_dynamic_tokens, pool_token_similarities, dilate_token_mask
# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import os

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from utils.debug_option import DEBUG
import torch.distributed as dist
import torch.nn.functional as F

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        if DEBUG:
            print(f"args.model_kwargs: {args.model_kwargs}")
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.dasa_config = DASAConfig.from_config(args, framework="longlive")
        self.dasa_config.apply_to_generator(self.generator)
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # hard code for Wan2.1-T2V-1.3B
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def local_nxn_match(self, reference, target, n=3, padding_mode="replicate", eps=1e-6):
        return local_nxn_match(reference, target, n=n, padding_mode=padding_mode, eps=eps)

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """Decode noise [B, T, C, H, W] into video in [0, 1]; optionally return latents."""
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        for attn_block in self.generator.model.blocks:
            attn_block.self_attn.current_frame_num = 0
            attn_block.self_attn.last_reference_frame = None
            attn_block.self_attn.last_update_step = None
            attn_block.self_attn.last_chunk_clean_k = None
            attn_block.self_attn.last_chunk_clean_v = None

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        all_num_frames = [self.num_frame_per_block] * num_blocks
        sparsity_params = {
            "current_denoising_step": None,
            "selected_indices": None,
            "search_range": self.dasa_config.search_size
        }
        for chunk_index, current_num_frames in enumerate(all_num_frames):
            selected_indices = None
            selected_token_indices = None
            unipatch_selected_indices = None

            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]

            for index, current_timestep in enumerate(self.denoising_step_list):
                if profile:
                    print(f"current_timestep: {current_timestep}")

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                sparsity_params["current_denoising_step"] = index
                sparsity_params["selected_indices"] = selected_token_indices
                sparsity_params["kvcache_reduction"] = False
                if index < len(self.denoising_step_list) - 1:
                    if selected_token_indices is not None and selected_token_indices.numel() == 0:
                        denoised_pred = self.last_chunk_latents
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            sparsity_params=sparsity_params,
                        )

                    cur_chunk = denoised_pred # (b, t, c, h, w)
                    # Reuse latents after the first chunk.
                    if chunk_index > 0:
                        last_chunk = self.last_chunk_latents

                        # Reuse only the intermediate denoising steps.
                        if selected_indices is not None:
                            cached_indices = ~unipatch_selected_indices.unsqueeze(dim=2)

                            cur_chunk = torch.where(
                                cached_indices,
                                aligned_prev,
                                cur_chunk
                            )
                            denoised_pred = cur_chunk

                        # Select dynamic tokens on the first step.
                        if index == 0:
                            if sparsity_params["search_range"] > 1:
                                aligned_prev, cosine_sims = self.local_nxn_match(last_chunk, denoised_pred, sparsity_params["search_range"])
                            else:
                                aligned_prev = last_chunk
                                cosine_sims = F.cosine_similarity(denoised_pred, last_chunk, dim=2)
                            selected_indices = select_dynamic_tokens(
                                cosine_sims, self.dasa_config.similarity_threshold, self.dasa_config.dilation_size
                            )

                            selected_indices = selected_indices.flatten()
                            # Reuse token positions across layers; retain the mask for blending.
                            selected_token_indices = selected_indices.nonzero(as_tuple=True)[0]
                            if profile:
                                print(f"Keep ratio: {selected_indices.float().mean()}")
                            unipatch_selected_indices = selected_indices.view(
                                cosine_sims.shape[0], cosine_sims.shape[1], cosine_sims.shape[2] // 2, cosine_sims.shape[3] // 2
                                ).repeat_interleave(2, -1).repeat_interleave(2, -2)

                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    sparsity_params["current_denoising_step"] = index
                    sparsity_params["selected_indices"] = None
                    sparsity_params["kvcache_reduction"] = False
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        sparsity_params=sparsity_params,
                    )

            self.last_chunk_latents = denoised_pred[:, -1:].repeat(1, 3, 1, 1, 1)
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # Refresh KV with clean context at timestep zero.
            sparsity_params["current_denoising_step"] = index + 1
            sparsity_params["selected_indices"] = None
            sparsity_params["kvcache_reduction"] = True
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                sparsity_params=sparsity_params,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        if getattr(self.args.model_kwargs, "use_infinite_attention", False):
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
        else:
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """Initialize a Per-GPU KV cache for the Wan model."""
        kv_cache1 = []
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                kv_cache_size = 171360

        # Keep counters on CPU to avoid CUDA synchronization.
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device="cpu"),
                "local_end_index": torch.tensor([0], dtype=torch.long, device="cpu")
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize a Per-GPU cross-attention cache for the Wan model."""
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """Set per-module attention limits; -1 uses the global token limit."""
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass
