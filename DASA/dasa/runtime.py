"""Repository paths and isolated upstream imports, with no ML dependencies."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
FRAMEWORKS = ("self_forcing", "longlive")


def framework_dir(framework):
    if framework not in FRAMEWORKS:
        raise ValueError(f"Unknown framework: {framework}")
    return ROOT / "frameworks" / framework


def config_path(framework, name="default.yaml"):
    framework_dir(framework)
    return ROOT / "configs" / framework / name


def prepare_runtime(framework):
    """Run one upstream import namespace per process, using repository-root paths."""
    selected = os.environ.get("DASA_ACTIVE_FRAMEWORK")
    if selected and selected != framework:
        raise RuntimeError("Run Self-Forcing and LongLive in separate Python processes")
    directory = framework_dir(framework)
    for name in ("wan", "model", "pipeline", "trainer", "utils", "demo_utils"):
        module = sys.modules.get(name)
        location = getattr(module, "__file__", None)
        if location and directory not in Path(location).resolve().parents:
            raise RuntimeError(f"The upstream namespace {name!r} is already loaded from {location}")
    os.environ["DASA_ACTIVE_FRAMEWORK"] = framework
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(directory))
    os.chdir(ROOT)


def load_config(path, framework):
    """Merge canonical framework defaults with the selected YAML file."""
    from omegaconf import OmegaConf
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return OmegaConf.merge(OmegaConf.load(config_path(framework)), OmegaConf.load(path))
