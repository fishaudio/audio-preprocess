# This file is edited from https://github.com/openvpi/audio-slicer/blob/main/slicer2.py

from pathlib import Path
from typing import Iterable, Union

import librosa
import numpy as np
import soundfile as sf

from fish_audio_preprocess.utils.slice_audio import slice_by_max_duration


def merge_short_chunks(chunks, min_length, max_duration, rate):
    merging_chunk = np.array([])
    merged_chunks = []
    # 将 chunks 安装长度排序, 但是好像没啥用，删了删了
    # chunks = sorted(chunks, key=lambda x: len(x))
    for chunk in chunks:
        if (len(merging_chunk) + len(chunk)) > max_duration * rate and len(
            merging_chunk
        ) > 0:
            merged_chunks.append(merging_chunk)
            merging_chunk = np.array([])
        else:
            merging_chunk = np.concatenate((merging_chunk, chunk))
    if len(merging_chunk) > 0:
        merged_chunks.append(merging_chunk)
    return merged_chunks


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 10,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )

        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )

        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        if samples.shape[0] <= self.min_length:
            return [waveform]

        rms_list = librosa.feature.rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0

        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue

            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue

            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )

            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start

                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))

                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )

                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )

                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))

                clip_start = pos_r
            silence_start = None

        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []

            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))

            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )

            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )

            return chunks


def slice_audio_v2(
    audio: np.ndarray,
    rate: int,
    min_duration: float = 5.0,
    max_duration: float = 30.0,
    min_silence_duration: float = 0.3,
    top_db: int = -40,
    hop_length: int = 10,
    max_silence_kept: float = 0.5,
    useMergeShort: bool = False,
) -> Iterable[np.ndarray]:
    """Slice audio by silence

    Args:
        audio: audio data, in shape (samples, channels)
        rate: sample rate
        min_duration: minimum duration of each slice
        max_duration: maximum duration of each slice
        min_silence_duration: minimum duration of silence
        top_db: threshold to detect silence
        hop_length: hop length to detect silence
        max_silence_kept: maximum duration of silence to be kept

    Returns:
        Iterable of sliced audio
    """

    if len(audio) / rate < min_duration:
        sliced_by_max_duration_chunk = slice_by_max_duration(chunk, max_duration, rate)
        yield from merge_short_chunks(
            sliced_by_max_duration_chunk, min_duration, max_duration, rate
        ) if useMergeShort else sliced_by_max_duration_chunk
        return

    slicer = Slicer(
        sr=rate,
        threshold=top_db,
        min_length=min_duration * 1000,
        min_interval=min_silence_duration * 1000,
        hop_size=hop_length,
        max_sil_kept=max_silence_kept * 1000,
    )

    sliced_audio = slicer.slice(audio)
    if useMergeShort:
        sliced_audio = merge_short_chunks(
            sliced_audio, min_duration, max_duration, rate
        )

    for chunk in sliced_audio:
        sliced_by_max_duration_chunk = slice_by_max_duration(chunk, max_duration, rate)
        yield from sliced_by_max_duration_chunk


def slice_audio_file_v2(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    min_duration: float = 5.0,
    max_duration: float = 30.0,
    min_silence_duration: float = 0.3,
    top_db: int = -40,
    hop_length: int = 10,
    max_silence_kept: float = 0.5,
    useFlat: bool = False,
    useMergeShort: bool = False,
) -> None:
    """
    Slice audio by silence and save to output folder

    Args:
        input_file: input audio file
        output_dir: output folder
        min_duration: minimum duration of each slice
        max_duration: maximum duration of each slice
        min_silence_duration: minimum duration of silence
        top_db: threshold to detect silence
        hop_length: hop length to detect silence
        max_silence_kept: maximum duration of silence to be kept
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, rate = librosa.load(str(input_file), sr=None, mono=True)
    for idx, sliced in enumerate(
        slice_audio_v2(
            audio,
            rate,
            min_duration=min_duration,
            max_duration=max_duration,
            min_silence_duration=min_silence_duration,
            top_db=top_db,
            hop_length=hop_length,
            max_silence_kept=max_silence_kept,
            useMergeShort=useMergeShort,
        )
    ):
        if useFlat:
            sf.write(str(output_dir) + f"_{idx:04d}.wav", sliced, rate)
        else:
            sf.write(str(output_dir / f"{idx:04d}.wav"), sliced, rate)
