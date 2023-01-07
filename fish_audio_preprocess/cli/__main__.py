import click
from loguru import logger

from .convert_to_wav import to_wav
from .length import length
from .loudness_norm import loudness_norm
from .separate_audio import separate
from .slice_audio import slice_audio


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug: bool):
    if debug:
        logger.info(f"Debug mode is on")


# Register subcommands
cli.add_command(to_wav)
cli.add_command(separate)
cli.add_command(loudness_norm)
cli.add_command(length)
cli.add_command(slice_audio)


if __name__ == "__main__":
    to_wav()
