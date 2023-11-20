from pathlib import Path

from PIL import Image


def make_gif(
    save_path: str | Path,
    image_dir: str | Path,
    suffix: str = "png",
    duration: int = 100,
    loop: int = 0,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
):
    """
    concrate images into one gif

    Args:
        save_path (str | Path): path to save gif
        image_dir (str | Path): directory path of images
        suffix (str, optional): image suffix. Defaults to "png".
        duration (int, optional): duration to display each image. Defaults to 100.
        loop (int, optional): replay times when finishing. Defaults to 0.
        start (int, optional): start index of image list. Defaults to 0.
        end (int | None, optional): end index of image list. Defaults to None.
        step (int, optional): skip images with step. Defaults to 1.
    """
    image_list: list[Image.Image] = [
        Image.open(x) for x in Path(image_dir).glob(f"*.{suffix}")
    ]
    if end is None:
        end = len(image_list)
    image_list = image_list[start:end:step]
    image_list[0].save(
        save_path,
        format="GIF",
        save_all=True,
        append_images=image_list[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
