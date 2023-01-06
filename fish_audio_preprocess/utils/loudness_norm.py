from pathlib import Path
from typing import Union

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def loudness_norm(audio: np.ndarray, rate: int, peak=-1.0, loudness=-23.0):
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.
    :param audio: audio data
    :param rate: sample rate
    :param peak: peak normalize audio to N dB, default -1.0 dB
    :param loudness: loudness normalize audio to N dB LUFS, default -23.0 dB LUFS

    :return: loudness normalized audio
    """

    # peak normalize audio to [peak] dB
    audio = pyln.normalize.peak(audio, peak)

    # measure the loudness first
    meter = pyln.Meter(rate)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    # loudness normalize audio to [loudness] dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(audio, _loudness, loudness)

    return loudness_normalized_audio


def loudness_norm_file(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    peak=-1.0,
    loudness=-23.0,
) -> None:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.
    :param input_file: input audio file
    :param output_file: output audio file
    :param peak: peak normalize audio to N dB, default -1.0 dB
    :param loudness: loudness normalize audio to N dB LUFS, default -23.0 dB LUFS
    """

    audio, rate = sf.read(input_file)
    audio = loudness_norm(audio, rate, peak, loudness)
    sf.write(output_file, audio, rate)
