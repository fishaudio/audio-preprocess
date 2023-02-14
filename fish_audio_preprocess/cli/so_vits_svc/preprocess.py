from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock, Value
from pathlib import Path

import click
import numpy as np
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import list_files


def resize2d(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source) * target_len, len(source)) / target_len,
        np.arange(0, len(source)),
        source,
    )
    return np.nan_to_num(target)


def compute_f0(path, c_len):
    import librosa
    from pyworld import pyworld

    x, sr = librosa.load(path, sr=32000)
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=800,
        frame_period=1000 * 320 / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, 32000)
    f0 = np.around(f0, 1)

    assert abs(c_len - x.shape[0] // 320) < 3, (c_len, f0.shape)

    return resize2d(f0, c_len)


HUBERT_MODEL = None


def init_hubert(worker_id: Value, lock: Lock):
    global HUBERT_MODEL

    import torch

    with lock:
        current_id = worker_id.value
        worker_id.value += 1

    device = torch.device(f"cuda:{current_id % torch.cuda.device_count()}")

    HUBERT_MODEL = torch.hub.load("bshall/hubert:main", "hubert_soft")
    HUBERT_MODEL.eval()
    HUBERT_MODEL.to(device)

    logger.info("Loaded Hubert model")


def process(filename: Path, overwrite: bool = False):
    import librosa
    import torch

    device = next(HUBERT_MODEL.parameters()).device

    # Process Hubert
    hubert_path = filename.parent / f"{filename.name}.soft.pt"
    if hubert_path.exists() is False or overwrite:
        wav, _ = librosa.load(filename, sr=16000)
        wav = torch.from_numpy(wav)[None, None].to(device)

        with torch.no_grad():
            c = HUBERT_MODEL.units(wav).transpose(1, 2)

        torch.save(c.cpu(), hubert_path)
    else:
        c = torch.load(hubert_path)

    # Process F0
    f0_path = filename.parent / f"{filename.name}.f0.npy"
    if f0_path.exists() is False or overwrite:
        f0 = compute_f0(filename, c.shape[-1] * 2)
        np.save(f0_path, f0)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--overwrite/--no-overwrite", default=False, help="Overwrite existing files"
)
@click.option(
    "--num-workers",
    help="Number of workers to use for processing, defaults to number of CPU cores",
    default=4,
    show_default=True,
    type=int,
)
def preprocess(
    input_dir: str,
    recursive: bool,
    overwrite: bool,
    num_workers: int,
):
    """Preprocess hubert and f0 for so_vits_svc"""

    input_dir = Path(input_dir)

    files = list_files(input_dir, extensions={".wav"}, recursive=recursive)
    logger.info(f"Found {len(files)} files, processing...")

    worker_id = Value("i", 0)
    lock = Lock()

    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=init_hubert, initargs=(worker_id, lock)
    ) as executor:
        tasks = [
            executor.submit(
                process,
                filename=file,
                overwrite=overwrite,
            )
            for file in tqdm(files, desc="Preparing tasks")
        ]
        for i in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            assert i.exception() is None, i.exception()

    logger.info("Done!")
    logger.info(f"Total: {len(files)}")


if __name__ == "__main__":
    preprocess()
