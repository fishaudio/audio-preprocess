import shutil
import subprocess as sp
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    list_files,
)


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
def to_wav(
    input_dir: str, output_dir: str, recursive: bool, overwrite: bool, clean: bool
):
    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if output_dir.exists():
        if clean:
            logger.info(f"Cleaning output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            logger.info(f"Output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_files(
        input_dir, extensions=VIDEO_EXTENSIONS | AUDIO_EXTENSIONS, recursive=recursive
    )
    logger.info(f"Found {len(files)} files, converting to wav")

    skipped = 0
    for idx, file in enumerate(files):
        # Get relative path to input_dir
        relative_path = file.relative_to(input_dir)
        new_file = (
            output_dir
            / relative_path.parent
            / relative_path.name.replace(file.suffix, ".wav")
        )

        if new_file.parent.exists() is False:
            new_file.parent.mkdir(parents=True)

        if new_file.exists() and overwrite is False:
            skipped += 1
            logger.info(f"Skipping existing file: {new_file}")
            continue

        sp.check_call(
            ["ffmpeg", "-i", str(file), "-y", str(new_file)],
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )

        logger.info(f"Processed {idx + 1}/{len(files)} files")

    logger.info(f"Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    to_wav()
