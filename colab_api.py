from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torchaudio


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class ValidationReport:
    total_files: int
    invalid_files: Tuple[ValidationIssue, ...]
    expected_sample_rate: int
    expected_channels: int

    @property
    def ok(self) -> bool:
        return not self.invalid_files


@dataclass(frozen=True)
class EgsPaths:
    train_dir: str
    valid_dir: str
    test_dir: str
    test_noisy_dir: Optional[str]


def download_if_missing(
    url: str,
    dest_path: Path,
    verify_zip: bool = True,
    overwrite_invalid: bool = True,
) -> Path:
    if not url:
        raise ValueError("url must be a non-empty string.")
    if dest_path.exists():
        if verify_zip and not zipfile.is_zipfile(dest_path):
            if overwrite_invalid:
                dest_path.unlink()
            else:
                raise ValueError(
                    f"Existing file is not a zip: {dest_path}"
                )
        else:
            return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with open(dest_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
    if verify_zip and not zipfile.is_zipfile(dest_path):
        raise ValueError(f"Downloaded file is not a zip: {dest_path}")
    return dest_path


def download_with_megadl(
    url: str,
    dest_path: Path,
    verify_zip: bool = True,
    overwrite_invalid: bool = True,
) -> Path:
    if not url:
        raise ValueError("url must be a non-empty string.")
    if shutil.which("megadl") is None:
        raise RuntimeError(
            "megadl is not available. Install megatools first "
            "(e.g., apt-get install -y megatools)."
        )
    if dest_path.exists():
        if verify_zip and not zipfile.is_zipfile(dest_path):
            if overwrite_invalid:
                dest_path.unlink()
            else:
                raise ValueError(
                    f"Existing file is not a zip: {dest_path}"
                )
        else:
            return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["megadl", "--path", str(dest_path), url])
    if verify_zip and not zipfile.is_zipfile(dest_path):
        raise ValueError(f"Downloaded file is not a zip: {dest_path}")
    return dest_path


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def list_wavs(root: Path) -> Tuple[Path, ...]:
    if not root.exists():
        raise FileNotFoundError(f"Audio root not found: {root}")
    return tuple(sorted(root.rglob("*.wav")))


def validate_wavs(
    wav_paths: Sequence[Path],
    expected_sample_rate: int,
    expected_channels: int = 1,
) -> ValidationReport:
    invalid: list[ValidationIssue] = []
    for path in wav_paths:
        info = _get_audio_info(path)
        sample_rate = info.sample_rate
        channels = info.channels
        if sample_rate != expected_sample_rate or channels != expected_channels:
            invalid.append(
                ValidationIssue(
                    path=str(path),
                    sample_rate=sample_rate,
                    channels=channels,
                )
            )
    return ValidationReport(
        total_files=len(wav_paths),
        invalid_files=tuple(invalid),
        expected_sample_rate=expected_sample_rate,
        expected_channels=expected_channels,
    )


def split_paths(
    wav_paths: Sequence[Path],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Tuple[Path, ...], Tuple[Path, ...], Tuple[Path, ...]]:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")
    rng = random.Random(seed)
    shuffled = list(wav_paths)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    train_paths = tuple(shuffled[:train_end])
    valid_paths = tuple(shuffled[train_end:valid_end])
    test_paths = tuple(shuffled[valid_end:])
    return train_paths, valid_paths, test_paths


def build_egs_from_wavs(
    wav_paths: Sequence[Path],
    egs_dir: Path,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    test_noisy_dir: Optional[Path] = None,
    test_clip_log10_range: Tuple[float, float] = (-2.0, -0.9),
) -> EgsPaths:
    train_paths, valid_paths, test_paths = split_paths(
        wav_paths=wav_paths,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    egs_dir.mkdir(parents=True, exist_ok=True)
    test_noisy_paths: Optional[Tuple[Path, ...]] = None
    if test_noisy_dir is not None:
        test_noisy_paths = _create_clipped_test_set(
            clean_paths=test_paths,
            output_dir=test_noisy_dir,
            seed=seed,
            clip_log10_range=test_clip_log10_range,
        )
    train_dir = _write_split(egs_dir / "tr", train_paths)
    valid_dir = _write_split(egs_dir / "cv", valid_paths)
    test_dir = _write_split(
        egs_dir / "ts",
        test_paths,
        noisy_paths=test_noisy_paths,
    )
    return EgsPaths(
        train_dir=str(train_dir),
        valid_dir=str(valid_dir),
        test_dir=str(test_dir),
        test_noisy_dir=str(test_noisy_dir) if test_noisy_dir else None,
    )


def run_hifigan_training(
    train_dir: Path,
    valid_dir: Path,
    test_dir: Path,
    output_dir: Path,
    sample_rate: int,
    save_every: int,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        "python",
        "train_hifigan.py",
        "+augmentor=shift_only",
        "+tr_loader=clippedclean",
        "+cv_loader=clippedclean",
        "+ts_loader=setting_1",
        "+loss=setting_1",
        "+model=setting_1",
        "+optimizer=adamw_1e4",
        "+experiment=setting_3_bs2",
        "+solver=hifiganaudiotoaudio",
        f"+data.train={train_dir}",
        f"+data.valid={valid_dir}",
        f"+data.test={test_dir}",
        f"+experiment.output_path={output_dir}",
        f"+experiment.save_every={save_every}",
        f"+tr_loader.parameters.sample_rate={sample_rate}",
        f"+cv_loader.parameters.sample_rate={sample_rate}",
        f"+ts_loader.parameters.sample_rate={sample_rate}",
        f"+model.parameters.sample_rate={sample_rate}",
    ]
    if extra_args:
        train_cmd.extend(extra_args)
    subprocess.check_call(train_cmd)


def _file_length(path: Path) -> int:
    info = _get_audio_info(path)
    return info.length


@dataclass(frozen=True)
class _AudioInfo:
    length: int
    sample_rate: int
    channels: int


def _get_audio_info(path: Path) -> _AudioInfo:
    if hasattr(torchaudio, "info"):
        info = torchaudio.info(str(path))
        if hasattr(info, "num_frames"):
            return _AudioInfo(
                length=info.num_frames,
                sample_rate=info.sample_rate,
                channels=info.num_channels,
            )
        siginfo = info[0]
        return _AudioInfo(
            length=siginfo.length // siginfo.channels,
            sample_rate=siginfo.rate,
            channels=siginfo.channels,
        )
    backend = getattr(torchaudio, "backend", None)
    if backend and hasattr(backend, "sox_io_backend"):
        info = backend.sox_io_backend.info(str(path))
        return _AudioInfo(
            length=info.num_frames,
            sample_rate=info.sample_rate,
            channels=info.num_channels,
        )
    raise RuntimeError("torchaudio does not provide an audio info backend.")


def _write_split(
    split_dir: Path,
    clean_paths: Sequence[Path],
    noisy_paths: Optional[Sequence[Path]] = None,
) -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    noisy_paths = noisy_paths if noisy_paths is not None else clean_paths
    clean_meta = [(str(path.resolve()), _file_length(path)) for path in clean_paths]
    noisy_meta = [(str(path.resolve()), _file_length(path)) for path in noisy_paths]
    with open(split_dir / "clean.json", "w") as handle:
        json.dump(clean_meta, handle, indent=2)
    with open(split_dir / "noisy.json", "w") as handle:
        json.dump(noisy_meta, handle, indent=2)
    return split_dir


def _create_clipped_test_set(
    clean_paths: Sequence[Path],
    output_dir: Path,
    seed: int,
    clip_log10_range: Tuple[float, float],
) -> Tuple[Path, ...]:
    import torch

    if not clean_paths:
        raise ValueError("clean_paths must contain at least one path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    root_dir = Path(os.path.commonpath([str(path.resolve()) for path in clean_paths]))
    if root_dir.suffix:
        root_dir = root_dir.parent
    noisy_paths: list[Path] = []
    for src_path in clean_paths:
        src_path = src_path.resolve()
        rel = src_path.relative_to(root_dir)
        dst_path = output_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        wav, sr = torchaudio.load(str(src_path))
        threshold = 10 ** rng.uniform(clip_log10_range[0], clip_log10_range[1])
        wav = torch.clamp(wav, min=-threshold, max=threshold)
        torchaudio.save(str(dst_path), wav, sr)
        noisy_paths.append(dst_path)
    return tuple(noisy_paths)
