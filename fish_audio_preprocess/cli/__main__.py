import click

from .convert_to_wav import to_wav


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


cli.add_command(to_wav)


if __name__ == "__main__":
    to_wav()
