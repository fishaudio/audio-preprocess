import click

from .convert_to_wav import to_wav
from .separate_audio import separate


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug: bool):
    if debug:
        click.echo(f"Debug mode is on")


# Register subcommands
cli.add_command(to_wav)
cli.add_command(separate)


if __name__ == "__main__":
    to_wav()
