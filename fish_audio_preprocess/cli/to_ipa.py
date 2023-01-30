import click

from fish_audio_preprocess.utils.to_ipa import chinese2p


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("pinyin_labels_file", type=click.Path(exists=False))
@click.argument("phoneme_labels_file", type=click.Path(exists=False))
@click.option(
    "--num_workers",
    "-n",
    type=int,
    default=8,
    help="Number of workers to use for processing.",
)
def to_ipa(input_file, pinyin_labels_file, phoneme_labels_file, num_workers):
    """Convert Chinese characters to pinyin and phonemes."""
    chinese2p(input_file, pinyin_labels_file, phoneme_labels_file, num_workers)
