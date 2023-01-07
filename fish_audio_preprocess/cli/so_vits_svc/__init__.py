import click

from .preprocess import preprocess


@click.group()
def so_vits_svc():
    """A command group for the so_vits_svc model."""


so_vits_svc.add_command(preprocess)
