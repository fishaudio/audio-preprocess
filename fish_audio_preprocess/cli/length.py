from pathlib import Path

import click
import soundfile as sf
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--visualize/--no-visualize", default=False, help="Visualize the distribution"
)
def length(
    input_dir: str,
    recursive: bool,
    visualize: bool,
):
    """
    Get the length of all audio files in a directory
    """

    input_dir = Path(input_dir)
    files = list_files(input_dir, AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, calculating length")

    infos = []
    for file in tqdm(files, desc="Collecting infos"):
        sound = sf.SoundFile(str(file))
        infos.append((len(sound), sound.samplerate, len(sound) / sound.samplerate))

    # Duration
    total_duration = sum([i[2] for i in infos])
    avg_duration = total_duration / len(infos)
    logger.info(f"Total duration: {total_duration / 3600:.2f} hours")
    logger.info(f"Average duration: {avg_duration:.2f} seconds")
    logger.info(f"Max duration: {max([i[2] for i in infos]):.2f} seconds")
    logger.info(f"Min duration: {min([i[2] for i in infos]):.2f} seconds")

    # Sample Rate
    total_samplerate = sum([i[1] for i in infos])
    avg_samplerate = total_samplerate / len(infos)
    logger.info(f"Average samplerate: {avg_samplerate:.2f}")

    if visualize is False:
        return

    # Visualize
    plt.hist([i[2] for i in infos], bins=100)
    plt.title(
        f"Distribution of audio lengths (Total: {len(infos)} files, {total_duration / 3600:.2f} hours)"
    )
    plt.xlabel("Length (seconds)")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    length()
