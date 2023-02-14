from pathlib import Path
from typing import Optional, Union

import torch
from demucs.apply import BagOfModels, apply_model
from demucs.audio import save_audio as _save_audio
from demucs.pretrained import get_model
from demucs.separate import load_track as _load_track
from loguru import logger


def init_model(
    name: str = "htdemucs",
    device: Optional[Union[str, torch.device]] = None,
    segment: Optional[int] = None,
) -> torch.nn.Module:
    """
    Initialize the model

    Args:
        name: Name of the model
        device: Device to use
        segment: Set split size of each chunk. This can help save memory of graphic card.

    Returns:
        The model
    """

    model = get_model(name)
    model.eval()

    if device is not None:
        model.to(device)

    logger.info(f"Model {name} loaded on {device}")

    if isinstance(model, BagOfModels) and len(model.models) > 1:
        logger.info(
            f"Selected model is a bag of {len(model.models)} models. "
            f"You will see {len(model.models)} progress bars per track."
        )

    if segment is not None:
        if isinstance(model, BagOfModels):
            for m in model.models:
                m.segment = segment
        else:
            model.segment = segment

    return model


def load_track(
    model: torch.nn.Module,
    path: Union[str, Path],
) -> torch.Tensor:
    """
    Load audio track

    Args:
        model: The model
        path: Path to the audio file

    Returns:
        The audio
    """

    return _load_track(path, model.audio_channels, model.samplerate)


def separate_audio(
    model: torch.nn.Module,
    audio: torch.Tensor,
    shifts: int = 1,
    num_workers: int = 0,
    progress: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Separate audio into sources

    Args:
        model: The model
        audio: The audio
        shifts: Run the model N times, larger values will increase the quality but also the time
        num_workers: Number of workers to use
        progress: Show progress bar

    Returns:
        The separated tracks
    """

    device = next(model.parameters()).device

    ref = audio.mean(0)
    audio = (audio - ref.mean()) / audio.std()

    sources = apply_model(
        model,
        audio[None],
        device=device,
        shifts=shifts,
        split=True,
        overlap=0.25,
        progress=progress,
        num_workers=num_workers,
    )[0]

    sources = sources * ref.std() + ref.mean()

    return dict(zip(model.sources, sources))


def save_audio(
    model: torch.nn.Module,
    path: Union[str, Path],
    track: torch.Tensor,
) -> None:
    """
    Save audio track

    Args:
        model: The model
        path: Path to save the audio file
        track: The audio tracks
    """

    _save_audio(
        track,
        path,
        model.samplerate,
        clip="rescale",
        as_float=False,
        bits_per_sample=16,
    )


def merge_tracks(
    tracks: dict[str, torch.Tensor],
    filter: Optional[list[str]] = None,
) -> torch.Tensor:
    """
    Merge tracks into one audio

    Args:
        tracks: The separated audio tracks
        filter: The tracks to merge

    Returns:
        The merged audio
    """

    if filter is None:
        filter = list(tracks.keys())

    merged = torch.zeros_like(next(iter(tracks.values())))

    for key in tracks:
        if key in filter:
            merged += tracks[key]

    return merged


if __name__ == "__main__":
    model = init_model("htdemucs", device="cuda:0")

    audio = load_track(
        model,
        "data/sources/其他素材/鸡你太美原曲《只因你太美》完整版-.wav",
    )

    tracks = separate_audio(model, audio, shifts=1, num_workers=0, progress=True)
    print(tracks.keys())
    print(tracks["vocals"])

    merged = merge_tracks(tracks, filter=None)
    save_audio(model, "revovered.wav", merged)

    merged = merge_tracks(tracks, filter=["vocals"])
    save_audio(model, "revovered_vocals.wav", merged)

    merged = merge_tracks(tracks, filter=["drums", "bass", "other"])
    save_audio(model, "revovered_instrumental.wav", merged)
