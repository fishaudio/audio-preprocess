import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs


def resample_file(
    input_file: Path, output_file: Path, overwrite: bool, samping_rate: int, mono: bool
):
    import librosa
    import soundfile as sf

    if overwrite is False and output_file.exists():
        return

    audio, _ = librosa.load(str(input_file), sr=samping_rate, mono=mono)

    if audio.ndim == 2:
        audio = audio.T

    sf.write(str(output_file), audio, samping_rate)


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
    "--num-workers",
    help="Number of workers to use for processing, defaults to number of CPU cores",
    default=os.cpu_count(),
    show_default=True,
    type=int,
)
@click.option(
    "--sampling-rate",
    "-sr",
    help="Sampling rate to resample to",
    default=44100,
    show_default=True,
    type=int,
)
@click.option(
    "--mono/--no-mono",
    default=True,
    help="Resample to mono (1 channel)",
)
def resample(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    num_workers: int,
    sampling_rate: int,
    mono: bool,
):
    """
    Resample all audio files in input_dir to output_dir.
    """

    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)

    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, resampling to {sampling_rate} Hz")

    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []

        for file in tqdm(files, desc="Preparing tasks"):
            # Get relative path to input_dir
            relative_path = file.relative_to(input_dir)
            new_file = output_dir / relative_path

            if new_file.parent.exists() is False:
                new_file.parent.mkdir(parents=True)

            if new_file.exists() and not overwrite:
                skipped += 1
                continue

            tasks.append(
                executor.submit(
                    resample_file, file, new_file, overwrite, sampling_rate, mono
                )
            )

        for i in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            assert i.exception() is None, i.exception()

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    resample()
