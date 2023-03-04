import click
import richuru
from loguru import logger

from .convert_to_wav import to_wav
from .frequency import frequency
from .length import length
from .loudness_norm import loudness_norm
from .separate_audio import separate
from .slice_audio import slice_audio
from .so_vits_svc import so_vits_svc
from .to_ipa import to_ipa


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug: bool):
    """An audio preprocessing CLI."""

    if debug:
        richuru.install()
        logger.info("Debug mode is on")


# Register subcommands
cli.add_command(length)
cli.add_command(frequency)

cli.add_command(to_wav)
cli.add_command(separate)
cli.add_command(loudness_norm)
cli.add_command(slice_audio)

cli.add_command(so_vits_svc)
cli.add_command(to_ipa)


if __name__ == "__main__":
    to_wav()
