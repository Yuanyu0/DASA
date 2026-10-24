# Isolate backbone imports and resolve repository paths.
import sys as _dasa_sys
from pathlib import Path as _DASAPath
_dasa_root = next(p for p in _DASAPath(__file__).resolve().parents if (p / 'dasa' / 'runtime.py').is_file())
if str(_dasa_root) not in _dasa_sys.path:
    _dasa_sys.path.insert(0, str(_dasa_root))
from dasa.runtime import prepare_runtime, load_config, config_path
prepare_runtime('self_forcing')

import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=None,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=None, help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action=argparse.BooleanOptionalAction, default=None,
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()
if args.config_path is None:
    args.config_path = str(config_path("self_forcing", "inference.yaml"))
config = load_config(args.config_path, "self_forcing")
# Explicit CLI values override YAML; defaults live in the canonical inference config.
for _field, _fallback in {
    "data_path": None, "extended_prompt_path": None,
    "output_folder": "outputs/self_forcing/inference", "num_output_frames": 21,
    "seed": 0, "num_samples": 1, "use_ema": False, "save_with_index": False,
}.items():
    if getattr(args, _field) is None:
        setattr(args, _field, config.get(_field, _fallback))
if args.checkpoint_path is None:
    args.checkpoint_path = config.get("generator_ckpt", None)
if not args.data_path:
    parser.error("Provide --data_path or data_path in the inference YAML")

if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")

    state_dict = state_dict['generator' if not args.use_ema else 'generator_ema']
    from collections import OrderedDict
    def clean_key(k):
        k = k.replace("model._fsdp_wrapped_module.", "model.")
        k = k.replace("_fsdp_wrapped_module.", "")
        return k
    new_state_dict = OrderedDict((clean_key(k), v) for k, v in state_dict.items())

    pipeline.generator.load_state_dict(new_state_dict)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output

import time
duration = 0
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    for seed_idx in range(1):
        idx = batch_data['idx'].item()

        # Unpack the single-item batch.
        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]

        all_video = []
        num_generated_frames = 0

        if args.i2v:
            # For image-to-video, batch contains image and caption
            prompt = batch['prompts'][0]  
            prompts = [prompt] * args.num_samples

            image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
        else:
            prompt = batch['prompts'][0]
            extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
            if extended_prompt is not None:
                prompts = [extended_prompt] * args.num_samples
            else:
                prompts = [prompt] * args.num_samples
            initial_latent = None

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
            )

        torch.cuda.synchronize()
        start_time = time.time()

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            low_memory=low_memory,
        )

        torch.cuda.synchronize()
        end_time = time.time()
        duration += end_time - start_time

        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        num_generated_frames += latents.shape[1]

        video = 255.0 * torch.cat(all_video, dim=1)

        pipeline.vae.model.clear_cache()

        # Save the video if the current prompt is not a dummy prompt
        if idx < num_prompts:
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:200]}-{seed_idx}.mp4')
            write_video(output_path, video[0], fps=16)

print(f"执行时长: {duration:.4f} 秒")
