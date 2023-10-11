# Copyright (c) OpenMMLab. All rights reserved.
import click

from lmdeploy.model import MODELS
from lmdeploy.serve.turbomind.deploy import supported_formats


@click.command('convert')
@click.argument(
    'model_path',
    type=click.Path(exists=True, readable=True),
)
@click.option('--model_name',
              type=click.Choice(list(MODELS.module_dict)),
              default='internlm-chat-7b',
              help='Input model name. Default is internlm-chat-7b')
@click.option(
    '--model_format',
    type=click.Choice(supported_formats),
    default=None,
    help="The format of the model, fb or hf. 'fb' stands for \
        META's llama format, and 'hf' means huggingface format",
)
@click.option('--tokenizer_path',
              type=click.Path(exists=True, readable=True),
              default=None,
              help='The path of tokenizer model')
@click.option('--workspace',
              type=click.Path(),
              default='./workspace',
              help='Destination path to save the output model.')
@click.option('--tp',
              type=int,
              default=1,
              help='The number of GPUs used for tensor parallelism.')
@click.option('--quant_path',
              type=click.Path(),
              default=None,
              help='The path of quantized model. Used for awq models.')
@click.option(
    '--group_size',
    type=int,
    default=0,
    help='The parameter used in AWQ to quantize fp16 weights to 4 bits.')
def convert(
    model_path: str,
    model_name: str,
    model_format: str = None,
    tokenizer_path: str = None,
    workspace: str = './workspace',
    tp: int = 1,
    quant_path: str = None,
    group_size: int = 0,
) -> None:
    """Convert models to llmdeploy model format.

    \b
    Example:
        > lmdeploy convert ~/internlm-chat-7b \\
        >   --model_name internlm-chat-7b
    """
    from lmdeploy.serve.turbomind.deploy import main as run_convert

    run_convert(model_name,
                model_path,
                model_format=model_format,
                tokenizer_path=tokenizer_path,
                dst_path=workspace,
                tp=tp,
                quant_path=quant_path,
                group_size=group_size)
