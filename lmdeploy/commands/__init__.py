# Copyright (c) OpenMMLab. All rights reserved.

import click

from .convert import convert
from .inference import inference
from .serve import serve


@click.group()
def cli():
    """LMDeploy Command Line Interface.

    The CLI provides a unified API for converting, serving and testing large
    language models.
    """

    pass


cli.add_command(convert)
cli.add_command(inference)
cli.add_command(serve)

__all__ = ['cli']
