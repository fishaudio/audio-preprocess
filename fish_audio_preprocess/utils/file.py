import shutil
from pathlib import Path
from typing import Union

from loguru import logger

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".flv",
    ".mov",
    ".wmv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".m4v",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}


def list_files(
    path: Union[Path, str],
    extensions: set[str] = None,
    recursive: bool = False,
    sort: bool = True,
) -> list[Path]:
    """List files in a directory.

    Args:
        path (Path): Path to the directory.
        extensions (set, optional): Extensions to filter. Defaults to None.
        recursive (bool, optional): Whether to search recursively. Defaults to False.
        sort (bool, optional): Whether to sort the files. Defaults to True.

    Returns:
        list: List of files.
    """

    if isinstance(path, str):
        path = Path(path)

    files = path.glob("**/*") if recursive else path.glob("*")
    files = [f for f in files if f.is_file()]

    if extensions is not None:
        files = [f for f in files if f.suffix in extensions]

    if sort:
        files = sorted(files)

    return files


def make_dirs(path: Union[Path, str], clean: bool = False):
    """Make directories.

    Args:
        path (Union[Path, str]): Path to the directory.
        clean (bool, optional): Whether to clean the directory. Defaults to False.
    """
    if isinstance(path, str):
        path = Path(path)

    if path.exists():
        if clean:
            logger.info(f"Cleaning output directory: {path}")
            shutil.rmtree(path)
        else:
            logger.info(f"Output directory already exists: {path}")

    path.mkdir(parents=True, exist_ok=True)
