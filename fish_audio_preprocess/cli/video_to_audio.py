import shutil
import subprocess as sp
from pathlib import Path

import click
from tqdm import tqdm

from fish_audio_preprocess.utils.file import VIDEO_EXTENSIONS, list_files


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
def video_to_audio(
    input_dir: str, output_dir: str, recursive: bool, overwrite: bool, clean: bool
):
    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if output_dir.exists() and clean:
        click.echo(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_files(input_dir, extensions=VIDEO_EXTENSIONS, recursive=recursive)
    click.echo(f"Found {len(files)} video files")

    skipped = 0
    for file in tqdm(files, desc="Converting videos to audio"):
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
            continue

        sp.check_call(
            ["ffmpeg", "-i", str(file), "-y", str(new_file)],
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )

    click.echo(f"Done!")
    click.echo(f"Total: {len(files)}, Skipped: {skipped}")
    click.echo(f"Output directory: {output_dir}")


if __name__ == "__main__":
    video_to_audio()
