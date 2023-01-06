from pathlib import Path

import click
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--overwrite/--no-overwrite", default=False, help="Overwrite existing files"
)
@click.option(
    "--clean/--no-clean", default=False, help="Clean output directory before processing"
)
@click.option(
    "--peak",
    help="Peak normalize audio to -1 dB",
    default=-1.0,
    show_default=True,
    type=float,
)
@click.option(
    "--loudness",
    help="Loudness normalize audio to -23 dB LUFS",
    default=-23.0,
    show_default=True,
    type=float,
)
def loudness_norm(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    peak: float,
    loudness: float,
):
    """Perform loudness normalization (ITU-R BS.1770-4) on audio files."""

    from fish_audio_preprocess.utils.loudness_norm import (
        loudness_norm as _loudness_norm,
    )

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    make_dirs(output_dir, clean)

    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, converting to wav")

    skipped = 0
    for file in tqdm(files):
        # Get relative path to input_dir
        relative_path = file.relative_to(input_dir)
        new_file = output_dir / relative_path

        if new_file.parent.exists() is False:
            new_file.parent.mkdir(parents=True)

        if new_file.exists() and overwrite is False:
            skipped += 1
            continue

        audio, rate = sf.read(file)
        audio = _loudness_norm(audio, rate, peak, loudness)
        sf.write(new_file, audio, rate)

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    loudness_norm()
