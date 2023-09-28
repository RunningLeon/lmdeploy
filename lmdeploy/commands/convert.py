# Copyright (c) OpenMMLab. All rights reserved.
import os
import re

import click

from lmdeploy.model import MODELS
from lmdeploy.serve.turbomind.deploy import main as run_convert
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
@click.option('--kv_int8',
              is_flag=True,
              default=False,
              help='Indicator for whether to do KV int8 quantization.')
@click.option('--calib_dataset',
              type=click.Choice(['c4', 'ptb', 'wujutext2', 'pileval']),
              default=None,
              help='Calibration dataset when do kv_int8.')
@click.option(
    '--calib_samples',
    type=int,
    default=128,
    help='Number of samples in the calibration set, if the memory is not '
    'enough, it can be adjusted appropriately')
@click.option(
    '--calib_seqlen',
    type=int,
    default=2048,
    help='Length of a single text, if the memory is not enough, you can '
    'adjust it appropriately')
@click.option('--kv_sym',
              is_flag=True,
              default=False,
              help='Symmetric or asymmetric quantization, default is False.')
def convert(model_path: str,
            model_name: str,
            model_format: str = None,
            tokenizer_path: str = None,
            workspace: str = './workspace',
            tp: int = 1,
            quant_path: str = None,
            group_size: int = 0,
            kv_int8: bool = False,
            calib_dataset: str = None,
            calib_samples: int = 128,
            calib_seqlen: int = 2048,
            kv_sym: bool = False) -> None:
    """Convert models to llmdeploy model format.

    \b
    Example 1:
        > lmdeploy convert ~/internlm-chat-7b \\
        >   --model_name internlm-chat-7b

    \b
    Example 2:
        > lmdeploy convert ~/internlm-chat-7b \\
        >   --model_name internlm-chat-7b \\
        >   --kv_int8 \\
        >   --calib_dataset c4 \\
        >   --calib_samples 128 \\
        >   --calib_seqlen 2048
    """

    run_convert(model_name,
                model_path,
                model_format=model_format,
                tokenizer_path=tokenizer_path,
                dst_path=workspace,
                tp=tp,
                quant_path=quant_path,
                group_size=group_size)

    # do kv int8
    if kv_int8:
        from lmdeploy.lite.apis.calibrate import calibrate
        from lmdeploy.lite.apis.kv_qparams import main as run_kv_qparams
        calibrate(model_path,
                  calib_dataset=calib_dataset,
                  calib_samples=calib_samples,
                  calib_seqlen=calib_seqlen,
                  work_dir=workspace)
        run_kv_qparams(workspace,
                       turbomind_dir=os.path.join(workspace,
                                                  'triton_models/weights'),
                       kv_sym=kv_sym,
                       num_tp=tp)
        # update the values in `config.ini`
        config_path = os.path.join(workspace,
                                   'triton_models/weights/config.ini')
        with open(config_path, 'r') as f:
            content = f.read()
            content = re.sub(r'use_context_fmha = \d+', 'use_context_fmha = 1',
                             content)
            content = re.sub(r'quant_policy = \d+', 'quant_policy = 4',
                             content)
        with open(config_path, 'w') as f:
            f.write(content)
