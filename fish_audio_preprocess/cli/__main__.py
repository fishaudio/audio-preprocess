import click

from .video_to_audio import video_to_audio


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


cli.add_command(video_to_audio)


if __name__ == "__main__":
    cli()
