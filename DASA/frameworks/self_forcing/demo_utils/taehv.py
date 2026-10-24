#!/usr/bin/env python3
"""Tiny autoencoder for Hunyuan Video latents."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True),
                                  conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """Process [N, T, C, H, W] video using O(T) parallel or O(1) sequential memory."""
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N * T, C, H, W)
        # parallel over input timesteps, iterate over blocks
        for b in tqdm(model, disable=not show_progress_bar):
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        out = []
        # Traverse temporal pooling and growth with a work queue.
        work_queue = [TWorkItem(xt, 0) for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))]
        # Advance progress once per consumed source node.
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        # we'll also need a separate addressable memory per node as well
        mem = [None] * len(model)
        while work_queue:
            xt, i = work_queue.pop(0)
            if i == 0:
                progress_bar.update(1)
            if i == len(model):
                out.append(xt)
            else:
                b = model[i]
                if isinstance(b, MemBlock):
                    # mem blocks are simple since we're visiting the graph in causal order
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i].copy_(xt)
                    work_queue.insert(0, TWorkItem(xt_new, i + 1))
                elif isinstance(b, TPool):
                    if mem[i] is None:
                        mem[i] = []  # pool memory is itself a queue of inputs to pool
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        raise ValueError("???")
                    elif len(mem[i]) < b.stride:
                        pass
                    else:
                        N, C, H, W = xt.shape
                        xt = b(torch.cat(mem[i], 1).view(N * b.stride, C, H, W))
                        mem[i] = []
                        work_queue.insert(0, TWorkItem(xt, i + 1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C, H, W = xt.shape
                    # each tgrow has multiple successor nodes
                    for xt_next in reversed(xt.view(N, b.stride * C, H, W).chunk(b.stride, 1)):
                        work_queue.insert(0, TWorkItem(xt_next, i + 1))
                else:
                    xt = b(xt)
                    work_queue.insert(0, TWorkItem(xt, i + 1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x


class TAEHV(nn.Module):
    latent_channels = 16
    image_channels = 3

    def __init__(self, checkpoint_path="taehv.pth", decoder_time_upscale=(True, True), decoder_space_upscale=(True, True, True)):
        """Load TAEHV or Wan weights with optional reduced-resolution decoding."""
        super().__init__()
        self.encoder = nn.Sequential(
            conv(TAEHV.image_channels, 64), nn.ReLU(inplace=True),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, TAEHV.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2**sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(), conv(TAEHV.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(
                scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(
                scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(
                scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], TAEHV.image_channels),
        )
        if checkpoint_path is not None:
            self.load_state_dict(self.patch_tgrow_layers(torch.load(
                checkpoint_path, map_location="cpu", weights_only=True)))

    def patch_tgrow_layers(self, sd):
        """Adjust TGrow kernels to match the loaded state dictionary."""
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        """Encode RGB [N, T, 3, H, W] in [0, 1]; sequential mode uses less memory."""
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=True, show_progress_bar=False):
        """Decode latents to RGB in [0, 1]; sequential mode uses less memory."""
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        return x

    def forward(self, x):
        return self.c(x)


@torch.no_grad()
def main():
    """Run TAEHV roundtrip reconstruction on the given video paths."""
    import os
    import sys
    import cv2

    class VideoTensorReader:
        def __init__(self, video_file_path):
            self.cap = cv2.VideoCapture(video_file_path)
            assert self.cap.isOpened(), f"Could not load {video_file_path}"
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        def __iter__(self):
            return self

        def __next__(self):
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration
            return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)  # BGR HWC -> RGB CHW

    class VideoTensorWriter:
        def __init__(self, video_file_path, width_height, fps=30):
            self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width_height)
            assert self.writer.isOpened(), f"Could not create writer for {video_file_path}"

        def write(self, frame_tensor):
            assert frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3, f"{frame_tensor.shape}??"
            self.writer.write(cv2.cvtColor(frame_tensor.permute(1, 2, 0).numpy(),
                              cv2.COLOR_RGB2BGR))  # RGB CHW -> BGR HWC

        def __del__(self):
            if hasattr(self, 'writer'):
                self.writer.release()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    checkpoint_path = os.getenv("TAEHV_CHECKPOINT_PATH", "taehv.pth")
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    print(
        f"Using device \033[31m{dev}\033[0m, dtype \033[32m{dtype}\033[0m, checkpoint \033[34m{checkpoint_name}\033[0m ({checkpoint_path})")
    taehv = TAEHV(checkpoint_path=checkpoint_path).to(dev, dtype)
    for video_path in sys.argv[1:]:
        print(f"Processing {video_path}...")
        video_in = VideoTensorReader(video_path)
        video = torch.stack(list(video_in), 0)[None]
        vid_dev = video.to(dev, dtype).div_(255.0)
        if video.numel() < 100_000_000:
            print(f"  {video_path} seems small enough, will process all frames in parallel")
            vid_enc = taehv.encode_video(vid_dev)
            print(f"  Encoded {video_path} -> {vid_enc.shape}. Decoding...")
            vid_dec = taehv.decode_video(vid_enc)
            print(f"  Decoded {video_path} -> {vid_dec.shape}")
        else:
            print(f"  {video_path} seems large, will process each frame sequentially")
            vid_enc = taehv.encode_video(vid_dev, parallel=False)
            print(f"  Encoded {video_path} -> {vid_enc.shape}. Decoding...")
            vid_dec = taehv.decode_video(vid_enc, parallel=False)
            print(f"  Decoded {video_path} -> {vid_dec.shape}")
        video_out_path = video_path + f".reconstructed_by_{checkpoint_name}.mp4"
        video_out = VideoTensorWriter(
            video_out_path, (vid_dec.shape[-1], vid_dec.shape[-2]), fps=int(round(video_in.fps)))
        for frame in vid_dec.clamp_(0, 1).mul_(255).round_().byte().cpu()[0]:
            video_out.write(frame)
        print(f"  Saved to {video_out_path}")


if __name__ == "__main__":
    main()
