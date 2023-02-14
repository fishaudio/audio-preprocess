from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--visualize/--no-visualize", default=False, help="Visualize the distribution"
)
@click.option("-l", "--long-threshold", default=20, help="Threshold for long files")
@click.option("-s", "--short-threshold", default=2, help="Threshold for short files")
def length(
    input_dir: str,
    recursive: bool,
    visualize: bool,
    long_threshold: int,
    short_threshold: int,
):
    """
    Get the length of all audio files in a directory
    """

    import soundfile as sf
    from matplotlib import pyplot as plt

    input_dir = Path(input_dir)
    files = list_files(input_dir, AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, calculating length")

    infos = []
    for file in tqdm(files, desc="Collecting infos"):
        sound = sf.SoundFile(str(file))
        infos.append(
            (
                len(sound),
                sound.samplerate,
                len(sound) / sound.samplerate,
                file.relative_to(input_dir),
            )
        )

    # Duration
    total_duration = sum(i[2] for i in infos)
    avg_duration = total_duration / len(infos)
    logger.info(f"Total duration: {total_duration / 3600:.2f} hours")
    logger.info(f"Average duration: {avg_duration:.2f} seconds")
    logger.info(f"Max duration: {max(i[2] for i in infos):.2f} seconds")
    logger.info(f"Min duration: {min(i[2] for i in infos):.2f} seconds")

    # Too Long or Too Short
    long_files = [i for i in infos if i[2] > long_threshold]
    short_files = [i for i in infos if i[2] < short_threshold]
    # sort by duration
    if long_files:
        long_files = sorted(long_files, key=lambda x: x[2], reverse=True)
        logger.warning(
            f"Found {len(long_files)} files longer than {long_threshold} seconds"
        )
        for i in [f"{i[3]}: {i[2]:.2f}" for i in long_files]:
            logger.warning(f"    {i}")
    if short_files:
        short_files = sorted(short_files, key=lambda x: x[2], reverse=False)
        logger.warning(
            f"Found {len(short_files)} files shorter than {short_threshold} seconds"
        )
        for i in [f"{i[3]}: {i[2]:.2f}" for i in short_files]:
            logger.warning(f"    {i}")

    # Sample Rate
    total_samplerate = sum(i[1] for i in infos)
    avg_samplerate = total_samplerate / len(infos)
    logger.info(f"Average samplerate: {avg_samplerate:.2f}")

    if not visualize:
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
