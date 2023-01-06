import shutil
import subprocess as sp
from pathlib import Path

import click
import torch
from loguru import logger

from fish_audio_preprocess.utils.file import list_files, make_dirs
from fish_audio_preprocess.utils.separate_audio import (
    init_model,
    load_track,
    merge_tracks,
    save_audio,
    separate_audio,
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
@click.option(
    "--track", "-t", multiple=True, help="Name of track to keep", default=["vocals"]
)
@click.option("--model", help="Name of model to use", default="htdemucs")
@click.option(
    "--shifts", help="Number of shifts, improves separation quality a bit", default=1
)
@click.option("--num_workers", help="Number of workers", default=0)
def separate(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    track: list[str],
    model: str,
    shifts: int,
    num_workers: int,
):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    make_dirs(output_dir, clean)

    files = list_files(input_dir, extensions={".wav"}, recursive=recursive)
    logger.info(f"Found {len(files)} files, separating audio")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = init_model(model, device)

    skipped = 0
    for idx, file in enumerate(files):
        # Get relative path to input_dir
        relative_path = file.relative_to(input_dir)
        new_file = output_dir / relative_path

        if new_file.parent.exists() is False:
            new_file.parent.mkdir(parents=True)

        if new_file.exists() and overwrite is False:
            skipped += 1
            logger.info(f"Skipping existing file: {new_file}")
            continue

        source = load_track(_model, file)
        separated = separate_audio(
            _model, source, shifts=shifts, num_workers=num_workers
        )
        merged = merge_tracks(separated, track)
        save_audio(_model, new_file, merged)

        logger.info(f"Processed {idx + 1}/{len(files)} files")

    logger.info(f"Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    separate()
