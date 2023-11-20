from jsonargparse import CLI

from pytorchlab.utils.vision import make_gif


def make_gif_cli():
    CLI(make_gif)
