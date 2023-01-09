import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs

if TYPE_CHECKING:
    import torch


def worker(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    track: list[str],
    model: str,
    shifts: int,
    device: "torch.device",
    shard_idx: int = -1,
    total_shards: int = 1,
):
    from fish_audio_preprocess.utils.separate_audio import (
        init_model,
        load_track,
        merge_tracks,
        save_audio,
        separate_audio,
    )

    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)

    if shard_idx >= 0:
        files = [f for i, f in enumerate(files) if i % total_shards == shard_idx]

    shard_name = f"[Shard {shard_idx + 1}/{total_shards}]"
    logger.info(f"{shard_name} Found {len(files)} files, separating audio")

    _model = init_model(model, device)

    skipped = 0
    for file in tqdm(
        files,
        desc=f"{shard_name} Separating audio",
        position=0 if shard_idx < 0 else shard_idx,
        leave=False,
    ):
        # Get relative path to input_dir
        relative_path = file.relative_to(input_dir)
        new_file = output_dir / relative_path

        if new_file.parent.exists() is False:
            new_file.parent.mkdir(parents=True)

        if new_file.exists() and overwrite is False:
            skipped += 1
            continue

        source = load_track(_model, file)
        separated = separate_audio(_model, source, shifts=shifts, num_workers=0)
        merged = merge_tracks(separated, track)
        save_audio(_model, new_file, merged)

    logger.info(f"Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


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
@click.option("--num_workers_per_gpu", help="Number of workers per GPU", default=2)
def separate(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    track: list[str],
    model: str,
    shifts: int,
    num_workers_per_gpu: int,
):
    """
    Separates audio in input_dir using model and saves to output_dir.
    """

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    make_dirs(output_dir, clean)

    base_args = (
        input_dir,
        output_dir,
        recursive,
        overwrite,
        track,
        model,
        shifts,
    )

    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        logger.info(f"Device has {torch.cuda.device_count()} GPUs, let's use them!")

        mp.set_start_method("spawn")

        processes = []
        shards = torch.cuda.device_count() * num_workers_per_gpu
        for shard_idx in range(shards):
            p = mp.Process(
                target=worker,
                args=(
                    *base_args,
                    torch.device(f"cuda:{shard_idx % torch.cuda.device_count()}"),
                    shard_idx,
                    shards,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return

    worker(
        *base_args,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


if __name__ == "__main__":
    separate()
