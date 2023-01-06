import numpy as np
import pyloudnorm as pyln


def loudness_norm(audio: np.ndarray, rate: int, peak=-1.0, loudness=-12.0):
    # peak normalize audio to -1 dB
    audio = pyln.normalize.peak(audio, peak)

    # measure the loudness first
    meter = pyln.Meter(rate)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(audio, _loudness, loudness)

    return loudness_normalized_audio
