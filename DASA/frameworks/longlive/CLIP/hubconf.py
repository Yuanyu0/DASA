from clip.clip import tokenize as _tokenize, load as _load, available_models as _available_models
import re
import string

dependencies = ["torch", "torchvision", "ftfy", "regex", "tqdm"]

# For compatibility (cannot include special characters in function name)
model_functions = { model: re.sub(f'[{string.punctuation}]', '_', model) for model in _available_models()}

def _create_hub_entrypoint(model):
    def entrypoint(**kwargs):
        return _load(model, **kwargs)

    entrypoint.__doc__ = f"Load {model} CLIP and its image transform."
    return entrypoint

def tokenize():
    return _tokenize

_entrypoints = {model_functions[model]: _create_hub_entrypoint(model) for model in _available_models()}

globals().update(_entrypoints)
