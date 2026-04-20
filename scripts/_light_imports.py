from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def load_audio_and_timing_modules():
    try:
        from muq_beat_weaver.model.audio import load_manifest
        from muq_beat_weaver.model.audio import interpolate_muq_to_beat_grid
        from muq_beat_weaver.model.timing import (
            estimate_song_timing,
            load_metadata_dict,
            load_timing_metadata,
            save_timing_metadata,
            timing_fingerprint,
            timing_metadata_path,
        )

        return {
            "load_manifest": load_manifest,
            "interpolate_muq_to_beat_grid": interpolate_muq_to_beat_grid,
            "estimate_song_timing": estimate_song_timing,
            "load_metadata_dict": load_metadata_dict,
            "load_timing_metadata": load_timing_metadata,
            "save_timing_metadata": save_timing_metadata,
            "timing_fingerprint": timing_fingerprint,
            "timing_metadata_path": timing_metadata_path,
        }
    except ModuleNotFoundError as exc:
        # The timing-only environment may intentionally omit the training stack
        # (notably torch), which is imported transitively by
        # muq_beat_weaver.model.__init__. In that case, load only the modules we
        # need directly from source files.
        if exc.name not in {"muq_beat_weaver", "torch"}:
            raise

    repo_root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("muq_beat_weaver")
    package.__path__ = [str(repo_root / "muq_beat_weaver")]
    sys.modules["muq_beat_weaver"] = package
    model_pkg = types.ModuleType("muq_beat_weaver.model")
    model_pkg.__path__ = [str(repo_root / "muq_beat_weaver" / "model")]
    sys.modules["muq_beat_weaver.model"] = model_pkg

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    audio_mod = load("muq_beat_weaver.model.audio", repo_root / "muq_beat_weaver" / "model" / "audio.py")
    timing_mod = load("muq_beat_weaver.model.timing", repo_root / "muq_beat_weaver" / "model" / "timing.py")
    return {
        "load_manifest": audio_mod.load_manifest,
        "interpolate_muq_to_beat_grid": audio_mod.interpolate_muq_to_beat_grid,
        "estimate_song_timing": timing_mod.estimate_song_timing,
        "load_metadata_dict": timing_mod.load_metadata_dict,
        "load_timing_metadata": timing_mod.load_timing_metadata,
        "save_timing_metadata": timing_mod.save_timing_metadata,
        "timing_fingerprint": timing_mod.timing_fingerprint,
        "timing_metadata_path": timing_mod.timing_metadata_path,
    }
