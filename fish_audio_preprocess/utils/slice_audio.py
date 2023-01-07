import math
from pathlib import Path
from typing import Iterable, Union

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from loguru import logger


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
        return audio

    intervals = librosa.effects.split(
        audio.T, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    arr = []
    duration = 0
    idx = 0

    for start, end in intervals:
        time = (end - start) / rate

        duration += time
        arr.append(audio[start:end])

        if duration >= min_duration:
            duration = 0
            _gen = np.concatenate(arr)
            arr = []

            if len(_gen) > max_duration * rate:
                # Evenly split _gen into multiple slices
                n_chunks = math.ceil(len(_gen) / (max_duration * rate))
                chunk_size = len(_gen) // n_chunks

                # logger.warning(
                #     f"Audio {idx} is too long: {len(_gen) / rate:.2f}s, "
                #     f"split into {n_chunks} chunks, each chunk is {chunk_size / rate:.2f}s"
                # )

                for i in range(0, len(_gen), chunk_size):
                    idx += 1
                    yield _gen[i : i + chunk_size]
            else:
                idx += 1
                yield _gen

        if len(audio.shape) == 1:
            silent_shape = int(rate * pad_silence)
        else:
            silent_shape = (int(rate * pad_silence), audio.shape[1])

        arr.append(np.zeros(silent_shape, dtype=audio.dtype))


def slice_audio_file(
    input_file: Union[str, Path],
    output_folder: Union[str, Path],
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
        output_folder: output folder
        min_duration: minimum duration of each slice
        max_duration: maximum duration of each slice
        pad_silence: pad silence between each non-silent slice
        top_db: top_db of librosa.effects.split
        frame_length: frame_length of librosa.effects.split
        hop_length: hop_length of librosa.effects.split
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    audio, rate = sf.read(input_file)
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
        sf.write(output_folder / f"{idx:04d}.wav", sliced, rate)
