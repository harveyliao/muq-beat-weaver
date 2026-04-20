from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

_AUDIO_EXTENSIONS = {".ogg", ".egg", ".wav", ".mp3", ".flac"}


def load_audio(path: Path, sr: int = 24000) -> tuple[np.ndarray, int]:
    path = Path(path)
    audio, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio.astype(np.float32), sr


def pad_waveforms(waveforms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not waveforms:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=bool)
    max_len = max(len(w) for w in waveforms)
    batch = np.zeros((len(waveforms), max_len), dtype=np.float32)
    mask = np.zeros((len(waveforms), max_len), dtype=bool)
    for idx, wav in enumerate(waveforms):
        batch[idx, : len(wav)] = wav
        mask[idx, : len(wav)] = True
    return batch, mask


def save_manifest(manifest: dict[str, str], path: Path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _find_audio_in_folder(folder: Path, info_file: Path) -> Path | None:
    try:
        info = json.loads(Path(info_file).read_text(encoding="utf-8-sig"))
        audio_name = (
            info.get("_songFilename")
            or info.get("audio", {}).get("songFilename")
            or info.get("song", {}).get("songFilename")
        )
        if audio_name:
            audio_path = Path(folder) / audio_name
            if audio_path.exists():
                return audio_path
    except Exception:
        pass

    for ext in _AUDIO_EXTENSIONS:
        matches = sorted(Path(folder).glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def compute_mel_spectrogram(audio: np.ndarray, sr: int = 22050, n_mels: int = 80, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann')
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def compute_onset_envelope(audio: np.ndarray, sr: int = 22050, hop_length: int = 512) -> np.ndarray:
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    return onset.astype(np.float32).reshape(1, -1)


def compute_mel_with_onset(audio: np.ndarray, sr: int = 22050, n_mels: int = 80, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    mel = compute_mel_spectrogram(audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    onset = compute_onset_envelope(audio, sr=sr, hop_length=hop_length)
    min_len = min(mel.shape[1], onset.shape[1])
    return np.vstack([mel[:, :min_len], onset[:, :min_len]])


def beat_align_spectrogram(
    mel: np.ndarray,
    sr: int,
    hop_length: int,
    bpm: float,
    subdivisions_per_beat: int = 16,
    beat_offset_seconds: float = 0.0,
) -> np.ndarray:
    n_mels, n_frames = mel.shape
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    duration = frame_times[-1] if n_frames > 0 else 0.0
    start_time = max(0.0, float(beat_offset_seconds))
    if start_time >= duration:
        return np.zeros((n_mels, 0), dtype=np.float32)
    beats_per_second = bpm / 60.0
    subs_per_second = beats_per_second * subdivisions_per_beat
    total_subs = int(np.ceil((duration - start_time) * subs_per_second))
    if total_subs == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)
    sub_times = start_time + (np.arange(total_subs) / subs_per_second)
    frame_indices = np.interp(sub_times, frame_times, np.arange(n_frames))
    return np.stack([np.interp(frame_indices, np.arange(n_frames), mel_bin) for mel_bin in mel]).astype(np.float32)


def interpolate_muq_to_beat_grid(
    muq_features: np.ndarray,
    bpm: float,
    muq_hz: float = 25.0,
    subdivisions_per_beat: int = 16,
    beat_offset_seconds: float = 0.0,
) -> np.ndarray:
    if muq_features.ndim != 2:
        raise ValueError("MuQ features must be a 2D array.")
    if muq_features.shape[1] == 1024:
        features = muq_features.transpose()
    elif muq_features.shape[0] == 1024:
        features = muq_features
    elif muq_features.shape[1] < muq_features.shape[0]:
        features = muq_features.transpose()
    else:
        features = muq_features
    hidden, n_frames = features.shape
    duration = n_frames / muq_hz
    start_time = max(0.0, float(beat_offset_seconds))
    if start_time >= duration:
        return np.zeros((hidden, 0), dtype=np.float32)
    beats_per_second = bpm / 60.0
    subs_per_second = beats_per_second * subdivisions_per_beat
    total_subs = int(np.ceil((duration - start_time) * subs_per_second))
    if total_subs == 0:
        return np.zeros((hidden, 0), dtype=np.float32)
    src_times = np.arange(n_frames) / muq_hz
    tgt_times = start_time + (np.arange(total_subs) / subs_per_second)
    return np.stack([np.interp(tgt_times, src_times, row) for row in features]).astype(np.float32)

