import math
from pathlib import Path
from typing import Iterable, Union

import librosa
import numpy as np
import soundfile as sf


def slice_by_max_duration(
    gen: np.ndarray, slice_max_duration: float, rate: int
) -> Iterable[np.ndarray]:
    """Slice audio by max duration

    Args:
        gen: audio data, in shape (samples, channels)
        slice_max_duration: maximum duration of each slice
        rate: sample rate

    Returns:
        generator of sliced audio data
    """

    if len(gen) > slice_max_duration * rate:
        # Evenly split _gen into multiple slices
        n_chunks = math.ceil(len(gen) / (slice_max_duration * rate))
        chunk_size = math.ceil(len(gen) / n_chunks)

        for i in range(0, len(gen), chunk_size):
            yield gen[i : i + chunk_size]
    else:
        yield gen


def slice_audio(
    audio: np.ndarray,
    rate: int,
    min_duration: float = 6.0,
    max_duration: float = 30.0,
    pad_silence: float = 0.4,
    top_db: int = 60,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Iterable[np.ndarray]:
    """Slice audio by silence

    Args:
        audio: audio data, in shape (samples, channels)
        rate: sample rate
        min_duration: minimum duration of each slice
        max_duration: maximum duration of each slice
        pad_silence: pad silence between each non-silent slice
        top_db: top_db of librosa.effects.split
        frame_length: frame_length of librosa.effects.split
        hop_length: hop_length of librosa.effects.split

    Returns:
        Iterable of sliced audio
    """

    if len(audio) / rate < min_duration:
        yield from slice_by_max_duration(audio, max_duration, rate)
        return

    intervals = librosa.effects.split(
        audio.T, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    arr, duration = [], 0

    for start, end in intervals:
        time = (end - start) / rate

        duration += time
        arr.append(audio[start:end])

        if duration >= min_duration:
            _gen = np.concatenate(arr)
            arr, duration = [], 0
            yield from slice_by_max_duration(_gen, max_duration, rate)
            continue

        if len(audio.shape) == 1:
            silent_shape = int(rate * pad_silence)
        else:
            silent_shape = (int(rate * pad_silence), audio.shape[1])

        arr.append(np.zeros(silent_shape, dtype=audio.dtype))


def slice_audio_file(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    min_duration: float = 6.0,
    max_duration: float = 30.0,
    pad_silence: float = 0.4,
    top_db: int = 60,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> None:
    """
    Slice audio by silence and save to output folder

    Args:
        input_file: input audio file
        output_dir: output folder
        min_duration: minimum duration of each slice
        max_duration: maximum duration of each slice
        pad_silence: pad silence between each non-silent slice
        top_db: top_db of librosa.effects.split
        frame_length: frame_length of librosa.effects.split
        hop_length: hop_length of librosa.effects.split
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, rate = sf.read(str(input_file))
    for idx, sliced in enumerate(
        slice_audio(
            audio,
            rate,
            min_duration=min_duration,
            max_duration=max_duration,
            pad_silence=pad_silence,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    ):
        sf.write(str(output_dir / f"{idx:04d}.wav"), sliced, rate)
