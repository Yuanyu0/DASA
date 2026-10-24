from dasa.config import DASAConfig
from dasa.token_selection import local_nxn_match, select_dynamic_tokens, pool_token_similarities, dilate_token_mask
# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.memory import log_gpu_memory


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.dasa_config = getattr(generator, "dasa_config", DASAConfig.from_config(framework="longlive", training=True))
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        # Accept a fixed attention window or a per-timestep schedule.
        self.local_attn_size = kwargs.get("local_attn_size", -1)
        if not isinstance(self.local_attn_size, int) and hasattr(self.local_attn_size, "__iter__"):
            self.local_attn_size = list(self.local_attn_size)
        if isinstance(self.local_attn_size, (list, tuple)):
            assert len(self.local_attn_size) == len(self.denoising_step_list), (
                f"local_attn_size length ({len(self.local_attn_size)}) must match denoising_step_list length ({len(self.denoising_step_list)})."
            )
            if DEBUG:
                print(f"local_attn_size schedule length: {len(self.local_attn_size)}, denoising steps: {len(self.denoising_step_list)}")
        else:
            if DEBUG:
                print(f"Using static local_attn_size: {self.local_attn_size}")

        num_training_frames: Optional[int] = kwargs.get("num_training_frames", 21)
        slice_last_frames: int = int(kwargs.get("slice_last_frames", 21))

        # Compute KV cache supporting list/int and global attention (-1)
        def _resolve_kv_frames(local_cfg):
            if isinstance(local_cfg, (list, tuple)):
                base = int(max(local_cfg)) if len(local_cfg) > 0 else -1
                return min(base + slice_last_frames, num_training_frames)
            else:
                base = int(local_cfg)
                return min(base + slice_last_frames, num_training_frames)

        kv_frames = _resolve_kv_frames(self.local_attn_size)
        if DEBUG:
            print(f"[KV policy] local_attn_size={self.local_attn_size} slice_last_frames={slice_last_frames} num_training_frames={num_training_frames} -> kv_frames={kv_frames}")
        self.kv_cache_size = int(kv_frames) * self.frame_seq_length

    def local_nxn_match(self, reference, target, n=3, padding_mode="replicate", eps=1e-6):
        return local_nxn_match(reference, target, n=n, padding_mode=padding_mode, eps=eps)

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """Generate a cached chunk [B, T, C, H, W], optionally returning denoising timestep bounds."""
        batch_size, chunk_frames, num_channels, height, width = noise.shape

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] generate_chunk_with_cache: batch_size={batch_size}, chunk_frames={chunk_frames}")
            print(f"[SeqTrain-Pipeline] current_start_frame={current_start_frame}, requires_grad={requires_grad}")

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: Before chunk generation", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        if not self.independent_first_frame or chunk_frames % self.num_frame_per_block == 0:
            assert chunk_frames % self.num_frame_per_block == 0
            num_blocks = chunk_frames // self.num_frame_per_block
            all_num_frames = [self.num_frame_per_block] * num_blocks
        else:
            assert (chunk_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (chunk_frames - 1) // self.num_frame_per_block
            all_num_frames = [1] + [self.num_frame_per_block] * num_blocks

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] Block config: num_blocks={num_blocks}, all_num_frames={all_num_frames}")
            print(f"[SeqTrain-Pipeline] independent_first_frame={self.independent_first_frame}")

        output = torch.zeros_like(noise)

        # Randomly select denoising steps (synced across ranks)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] Denoising steps: {num_denoising_steps}, exit_flags: {exit_flags}")

        # Enable gradients only in the requested frame range.
        if not requires_grad:
            start_gradient_frame_index = chunk_frames
        else:
            start_gradient_frame_index = 0

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] start_gradient_frame_index={start_gradient_frame_index}")

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: Before block generation loop", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        local_start_frame = 0
        # If static local_attn_size, set it on the model before the step loop
        if not (isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes)))):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))

        for attn_block in self.generator.model.blocks:
            attn_block.self_attn.current_frame_num = 0
            attn_block.self_attn.last_reference_frame = None
            attn_block.self_attn.last_update_step = None
            attn_block.self_attn.last_chunk_clean_k = None
            attn_block.self_attn.last_chunk_clean_v = None

        sparsity_params = {
            "current_denoising_step": None,
            "selected_indices": None,
            "search_range": self.dasa_config.search_size
        }
        last_chunk_latents = None
        for block_index, current_num_frames in enumerate(all_num_frames):
            selected_indices = None
            unipatch_selected_indices = None

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Pipeline] Processing block {block_index}: frames {local_start_frame}-{local_start_frame + current_num_frames}")

            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and block_index == 0:
                log_gpu_memory(f"SeqTrain-Pipeline: Before first block generation", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)

            noisy_input = noise[:, local_start_frame:local_start_frame + current_num_frames]

            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                # If scheduled, set local_attn_size dynamically per timestep
                if isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes))):
                    self.generator.model.local_attn_size = int(self.local_attn_size[step_idx])
                    if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                        print(f"[denoise step {step_idx}] timestep={float(current_timestep)} local_attn_size={self.generator.model.local_attn_size}")
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[step_idx]))
                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep

                sparsity_params["current_denoising_step"] = step_idx
                sparsity_params["kvcache_reduction"] = False
                if step_idx == 0 or step_idx == 3:
                    sparsity_params["selected_indices"] = None
                else:
                    sparsity_params["selected_indices"] = selected_indices

                if not exit_flag:
                    # Intermediate steps: no gradients
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} intermediate steps (no grad)")

                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                            sparsity_params=sparsity_params,
                        )

                        cur_chunk = denoised_pred # (b, t, c, h, w)
                        # Reuse latents after the first chunk.
                        if block_index > 0:
                            last_chunk = last_chunk_latents

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
                            if step_idx == 0:
                                if sparsity_params["search_range"] > 1:
                                    aligned_prev, cosine_sims = self.local_nxn_match(last_chunk, denoised_pred, sparsity_params["search_range"])
                                else:
                                    aligned_prev = last_chunk
                                    cosine_sims = F.cosine_similarity(denoised_pred, last_chunk, dim=2)
                                selected_indices = select_dynamic_tokens(
                                    cosine_sims, self.dasa_config.similarity_threshold, self.dasa_config.dilation_size
                                )

                                selected_indices = selected_indices.flatten()

                                if selected_indices.sum() == 0:
                                    rand_idx = torch.randperm(selected_indices.numel(), device=selected_indices.device)[:5]
                                    selected_indices[rand_idx] = True

                                unipatch_selected_indices = selected_indices.view(
                                    cosine_sims.shape[0], cosine_sims.shape[1], cosine_sims.shape[2] // 2, cosine_sims.shape[3] // 2
                                    ).repeat_interleave(2, -1).repeat_interleave(2, -2)

                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Final step may require gradients
                    enable_grad = local_start_frame >= start_gradient_frame_index

                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} final step: enable_grad={enable_grad}")

                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                            sparsity_params=sparsity_params,
                        )

                    cur_chunk = denoised_pred # (b, t, c, h, w)
                    # Reuse latents after the first chunk.
                    if block_index > 0 and step_idx < len(self.denoising_step_list) - 1: # Reuse static latents only at intermediate steps.
                        last_chunk = last_chunk_latents
                        # Reuse only the intermediate denoising steps.
                        if selected_indices is not None:
                            cached_indices = ~unipatch_selected_indices.unsqueeze(dim=2)
                            cur_chunk = torch.where(
                                cached_indices,
                                aligned_prev,
                                cur_chunk
                            )
                            denoised_pred = cur_chunk
                    break

            last_chunk_latents = denoised_pred[:, -1:].repeat(1, 3, 1, 1, 1).detach()
            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred

            # Update cache with context noise
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])

            if DEBUG and block_index == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Pipeline] Updating cache with context_noise={self.context_noise}")

            sparsity_params["current_denoising_step"] = len(self.denoising_step_list)
            sparsity_params["selected_indices"] = None
            sparsity_params["kvcache_reduction"] = True
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                    sparsity_params=sparsity_params,
                )

            local_start_frame += current_num_frames

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: After all blocks generated", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            slice_last_frames: int = 21,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # Noise frames must form full chunks when the first frame is supplied.
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Generate an independent first frame before full chunks.
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - slice_last_frames

        grad_enable_mask = torch.zeros((batch_size, sum(all_num_frames)), dtype=torch.bool)
        if not isinstance(self.local_attn_size, (list, tuple)):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            for index, current_timestep in enumerate(self.denoising_step_list):
                # If scheduled, set local_attn_size dynamically per timestep
                if isinstance(self.local_attn_size, (list, tuple)):
                    self.generator.model.local_attn_size = int(self.local_attn_size[index])
                    if not dist.is_initialized() or dist.get_rank() == 0 and DEBUG:
                        print(f"[denoise step {index}] timestep={float(current_timestep)} local_attn_size={self.generator.model.local_attn_size}")
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[index]))
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Backpropagate only at the timestep shared by all ranks.
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                if DEBUG and dist.get_rank() == 0:
                    print(f"rank {dist.get_rank()}, current_start_frame: {current_start_frame}, current_num_frames: {current_num_frames}, current_timestep: {current_timestep}")
                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    if current_start_frame < start_gradient_frame_index:
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = False
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = True
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            if DEBUG and dist.get_rank() == 0:
                print(f"rank {dist.get_rank()}, current_start_frame: {current_start_frame}, current_num_frames: {current_num_frames}, current_timestep: {current_timestep}")
                print(f"rank {dist.get_rank()}, rerun_for_cache")
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            current_start_frame += current_num_frames

        if dist.get_rank() == 0 and DEBUG:
            print(f"grad_enable_mask: {grad_enable_mask[0, :]}")

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize a Per-GPU KV cache for the Wan model."""
        kv_cache1 = []
        if DEBUG:
            print(f"rank {dist.get_rank()} initialize kv cache with batch_size: {batch_size}, kv_cache_size: {self.kv_cache_size}")
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
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

    def clear_kv_cache(self):
        """Zero KV and cross-attention caches while preserving their allocated storage."""

        if getattr(self, "kv_cache1", None) is not None:
            for blk in self.kv_cache1:
                blk["k"].zero_()
                blk["v"].zero_()
                if "global_end_index" in blk:
                    blk["global_end_index"].zero_()
                if "local_end_index" in blk:
                    blk["local_end_index"].zero_()

        if getattr(self, "crossattn_cache", None) is not None:
            for blk in self.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

        for attn_block in self.generator.model.blocks:
            attn_block.self_attn.current_frame_num = 0
            attn_block.self_attn.last_reference_frame = None
            attn_block.self_attn.last_update_step = None
            attn_block.self_attn.last_chunk_clean_k = None
            attn_block.self_attn.last_chunk_clean_v = None

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """Set per-module attention limits; -1 uses the global token limit."""
        if isinstance(local_attn_size_value, (list, tuple)):
            raise ValueError("_set_all_modules_max_attention_size expects an int, got list/tuple.")

        if int(local_attn_size_value) == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        if hasattr(self.generator.model, "max_attention_size"):
            try:
                _ = getattr(self.generator.model, "max_attention_size")
            except Exception:
                pass
            setattr(self.generator.model, "max_attention_size", target_size)

        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    setattr(module, "max_attention_size", target_size)
                except Exception:
                    pass
