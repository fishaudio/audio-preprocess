from collections import Counter
from pathlib import Path

import click
import librosa
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from fish_audio_preprocess.utils.file import list_files


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--visualize/--no-visualize", default=True, help="Visualize the distribution"
)
def frequency(
    input_dir: str,
    recursive: bool,
    visualize: bool,
):
    """
    Get the frequency of all audio files in a directory
    """

    import parselmouth as pm

    input_dir = Path(input_dir)
    files = list_files(input_dir, {".wav"}, recursive=recursive)
    logger.info(f"Found {len(files)} files, calculating frequency")

    counter = Counter()

    for file in tqdm(files, desc="Collecting infos"):
        pitch_ac = pm.Sound(str(file)).to_pitch_ac()
        f0 = pitch_ac.selected_array["frequency"]

        for i in f0:
            if np.isinf(i) or np.isnan(i) or i == 0:
                continue

            counter[librosa.hz_to_note(i)] += 1

    data = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)

    for note, count in data:
        logger.info(f"{note}: {count}")

    if visualize is False:
        return

    x_axis_order = librosa.midi_to_note(list(range(0, 300)))
    data = sorted(counter.items(), key=lambda kv: x_axis_order.index(kv[0]))

    plt.rcParams["figure.figsize"] = [10, 4]
    plt.rcParams["figure.autolayout"] = True
    plt.bar([x[0] for x in data], [x[1] for x in data])
    plt.xticks(rotation=90)
    plt.title("Notes distribution")
    plt.xlabel("Notes")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    frequency()
