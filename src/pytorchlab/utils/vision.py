from pathlib import Path
from PIL import Image


def make_gif(save_path: str | Path, image_dir: str | Path, suffix: str = "png",duration:int=100, loop: int=0, step:int = 1):
    """use image below image_dir path with suffix to make gif

    Args:
        save_path (str | Path): gif path to save
        image_dir (str | Path): sourse images directory
        suffix (str, optional): image suffix. Defaults to 'png'.
        step (int, optional): skip images with step. Defaults to 1.
    """
    image_list: list[Image.Image] = [Image.open(x) for x in Path(image_dir).glob(f"*.{suffix}")][::step]
    image_list[0].save(
        save_path,
        format="GIF",
        save_all=True,
        append_images=image_list[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
