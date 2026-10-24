# Isolate backbone imports and resolve repository paths.
import sys as _dasa_sys
from pathlib import Path as _DASAPath
_dasa_root = next(p for p in _DASAPath(__file__).resolve().parents if (p / 'dasa' / 'runtime.py').is_file())
if str(_dasa_root) not in _dasa_sys.path:
    _dasa_sys.path.insert(0, str(_dasa_root))
from dasa.runtime import prepare_runtime, load_config, config_path
prepare_runtime('self_forcing')

# ---- INT8 (optional) ----
from demo_utils.vae import (
    VAEDecoderWrapperSingle,
    ZERO_VAE_CACHE
)
import pycuda.driver as cuda
import pycuda.autoinit  # noqa

import sys
from pathlib import Path

import torch
import tensorrt as trt

from utils.dataset import ShardingLMDBDataset

data_path = "/mnt/localssd/wanx_14B_shift-3.0_cfg-5.0_lmdb_oneshard"
dataset = ShardingLMDBDataset(data_path, max_pair=int(1e8))
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=0
)

# Prepare the VAE export inputs.

# Example inputs at 480 x 832 resolution.
dummy_input = torch.randn(1, 1, 16, 60, 104).half().cuda()
is_first_frame = torch.tensor([1.0], device="cuda", dtype=torch.float16)
dummy_cache_input = [
    torch.randn(*s.shape).half().cuda() if isinstance(s, torch.Tensor) else s
    for s in ZERO_VAE_CACHE               # keep exactly the same ordering
]
inputs = [dummy_input, is_first_frame, *dummy_cache_input]

# Export to ONNX.
model = VAEDecoderWrapperSingle().half().cuda().eval()

vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
decoder_state_dict = {}
for key, value in vae_state_dict.items():
    if 'decoder.' in key or 'conv2' in key:
        decoder_state_dict[key] = value
model.load_state_dict(decoder_state_dict)
model = model.half().cuda().eval()                          # only batch dim dynamic

onnx_path = Path("vae_decoder.onnx")
feat_names = [f"vae_cache_{i}" for i in range(len(dummy_cache_input))]
all_inputs_names = ["z", "use_cache"] + feat_names

with torch.inference_mode():
    torch.onnx.export(
        model,
        tuple(inputs),
        onnx_path.as_posix(),
        input_names=all_inputs_names,
        output_names=["rgb_out", "cache_out"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=True
    )
print(f"✅  ONNX graph saved to {onnx_path.resolve()}")

try:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path.as_posix(),
                                providers=["CUDAExecutionProvider"])
    ort_inputs = {n: t.cpu().numpy() for n, t in zip(all_inputs_names, inputs)}
    _ = sess.run(None, ort_inputs)
    print("✅  ONNX graph is executable")
except Exception as e:
    print("⚠️  ONNX check failed:", e)

# Build the TensorRT engine.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit("❌  ONNX → TRT parsing failed")

config = builder.create_builder_config()


def set_workspace(config, bytes_):
    """Version-agnostic workspace limit."""
    if hasattr(config, "max_workspace_size"):                # TRT 8 / 9
        config.max_workspace_size = bytes_
    else:                                                    # TRT 10+
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_)


config = builder.create_builder_config()
set_workspace(config, 4 << 30)          # 4 GB
# 4 GB

if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)


def set_workspace(config: trt.IBuilderConfig, bytes_: int = 4 << 30):
    """Set the workspace limit using the API supported by this TensorRT version."""
    if hasattr(config, "max_workspace_size"):                     # TRT 8 / 9
        config.max_workspace_size = bytes_
    else:                                                         # TRT 10+
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     bytes_)

# Optional INT8 calibration requires PyCUDA.


class VAECalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, loader, cache="calibration.cache", max_batches=10):
        super().__init__()
        self.loader = iter(loader)
        self.batch_size = loader.batch_size or 1
        self.max_batches = max_batches
        self.count = 0
        self.cache_file = cache
        self.stream = cuda.Stream()
        self.dev_ptrs = {}

    # Support both TensorRT calibration APIs.
    def get_batch_size(self):
        return self.batch_size

    def getBatchSize(self):
        return self.batch_size

    def get_batch(self, names):
        if self.count >= self.max_batches:
            return None

        import random
        vae_idx = random.randint(0, 10)
        data = next(self.loader)

        latent = data['ode_latent'][0][:, :1]
        is_first_frame = torch.tensor([1.0], device="cuda", dtype=torch.float16)
        feat_cache = ZERO_VAE_CACHE
        for i in range(vae_idx):
            inputs = [latent, is_first_frame, *feat_cache]
            with torch.inference_mode():
                outputs = model(*inputs)
            latent = data['ode_latent'][0][:, i + 1:i + 2]
            is_first_frame = torch.tensor([0.0], device="cuda", dtype=torch.float16)
            feat_cache = outputs[1:]

        # -------- ensure context is current --------
        z_np = latent.cpu().numpy().astype('float32')

        ptrs = []                # list[int] – one entry per name
        for name in names:         # <-- match TRT's binding order
            if name == "z":
                arr = z_np
            elif name == "use_cache":
                arr = is_first_frame.cpu().numpy().astype('float32')
            else:
                idx = int(name.split('_')[-1])   # "vae_cache_17" -> 17
                arr = feat_cache[idx].cpu().numpy().astype('float32')

            if name not in self.dev_ptrs:
                self.dev_ptrs[name] = cuda.mem_alloc(arr.nbytes)

            cuda.memcpy_htod_async(self.dev_ptrs[name], arr, self.stream)
            ptrs.append(int(self.dev_ptrs[name]))   # TensorRT expects integer device pointers.

        self.stream.synchronize()
        self.count += 1
        print(f"Calibration batch {self.count}/{self.max_batches}")
        return ptrs

    # --- calibration-cache helpers (both spellings) ---
    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def readCalibrationCache(self):
        return self.read_calibration_cache()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def writeCalibrationCache(self, cache):
        self.write_calibration_cache(cache)


config = builder.create_builder_config()
set_workspace(config, 4 << 30)                    # 4 GB

# ► enable FP16 if possible
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# ► enable INT-8  (delete this block if you don’t need it)
if cuda is not None:
    config.set_flag(trt.BuilderFlag.INT8)
    # supply any representative batch you like – here we reuse the latent z
    calib = VAECalibrator(dataloader)
    # TRT-10 renamed the setter:
    if hasattr(config, "set_int8_calibrator"):    # TRT 10+
        config.set_int8_calibrator(calib)
    else:                                         # TRT ≤ 9
        config.int8_calibrator = calib

# ---- optimisation profile ----
profile = builder.create_optimization_profile()
profile.set_shape(all_inputs_names[0],
                  min=(1, 1, 16, 60, 104),
                  opt=(1, 1, 16, 60, 104),
                  max=(1, 1, 16, 60, 104))
profile.set_shape("use_cache",               
                  min=(1,), opt=(1,), max=(1,))
for name, tensor in zip(all_inputs_names[2:], dummy_cache_input):
    profile.set_shape(name, tensor.shape, tensor.shape, tensor.shape)

config.add_optimization_profile(profile)

# Build the engine  (API changed in TRT-10)
print("⚙️  Building engine … (can take a minute)")

if hasattr(builder, "build_serialized_network"):          # TRT 10+
    serialized_engine = builder.build_serialized_network(network, config)
    assert serialized_engine is not None, "build_serialized_network() failed"
    plan_path = Path("checkpoints/vae_decoder_int8.trt")
    plan_path.write_bytes(serialized_engine)
    engine_bytes = serialized_engine
else:                                                     # TRT ≤ 9
    engine = builder.build_engine(network, config)
    assert engine is not None, "build_engine() returned None"
    plan_path = Path("checkpoints/vae_decoder_int8.trt")
    plan_path.write_bytes(engine.serialize())
    engine_bytes = engine.serialize()

print(f"✅  TensorRT engine written to {plan_path.resolve()}")

# Check the TensorRT engine.
with trt.Runtime(TRT_LOGGER) as rt:
    engine = rt.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream().cuda_stream

    device_buffers, outputs = {}, []
    dtype_map = {trt.float32: torch.float32,
                 trt.float16: torch.float16,
                 trt.int8:    torch.int8,
                 trt.int32:   torch.int32}

    for name, tensor in zip(all_inputs_names, inputs):
        if -1 in engine.get_tensor_shape(name):
            context.set_input_shape(name, tensor.shape)
        context.set_tensor_address(name, int(tensor.data_ptr()))
        device_buffers[name] = tensor

    context.infer_shapes()
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(context.get_tensor_shape(name))
            dtype = dtype_map[engine.get_tensor_dtype(name)]
            out = torch.empty(shape, dtype=dtype, device="cuda")
            context.set_tensor_address(name, int(out.data_ptr()))
            outputs.append(out)
            print(f"output {name} shape: {shape}")

    context.execute_async_v3(stream_handle=stream)
    torch.cuda.current_stream().synchronize()
    print("✅  TRT execution OK – first output shape:", outputs[0].shape)
