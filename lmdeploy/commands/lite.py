# Copyright (c) OpenMMLab. All rights reserved.

import os
import re

import click


@click.command('lite')
@click.argument(
    'model_path',
    type=click.Path(exists=True, readable=True),
)
@click.option('--method',
              type=click.Choice(['kvint8', 'awq']),
              default='kvint8',
              help='Choose which quant method. Default is kvint8.')
@click.option('--workspace',
              type=click.Path(),
              default='./workspace',
              help='Destination path to save the quantization results.')
@click.option('--turbomind_dir',
              type=click.Path(),
              default=None,
              help='Destination path to the quantization parameters.')
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
@click.option('--num_bits',
              type=int,
              default=8,
              help='Number of bits for quantization. Defaults to 8.')
@click.option('--symmetric',
              is_flag=True,
              default=False,
              help='Symmetric or asymmetric quantization, default is False.')
@click.option('--tp',
              type=int,
              default=1,
              help='The number of GPUs used for tensor parallelism.')
@click.option(
    '--group_size',
    type=int,
    default=128,
    help='The parameter used in AWQ to quantize fp16 weights to 4 bits.')
def lite(model_path: str,
         method: str = 'kvint8',
         workspace: str = './workspace',
         turbomind_dir: str = None,
         calib_dataset: str = None,
         calib_samples: int = 128,
         calib_seqlen: int = 2048,
         num_bits: int = 8,
         symmetric: bool = False,
         tp: int = 1,
         group_size: int = 128) -> None:
    """Run quantization on HF models.

    \b
    Example (kv-int8):
        > lmdeploy lite $HF_MODEL \\
        >   --workspace ./workspace \\
        >   --method kvint8 \\
        >   --num_bits 8 \\
        >   --calib_dataset c4 \\
        >   --calib_samples 128 \\
        >   --calib_seqlen 2048 \\
        >   --turbomind_dir ./workspace/triton_models/weights \\
        >   --tp 1

    \b
    Example (awq):
        > lmdeploy lite $HF_MODEL \\
        >   --workspace ./workspace \\
        >   --method awq \\
        >   --calib_dataset c4 \\
        >   --calib_samples 128 \\
        >   --num_bits 4 \\
        >   --group_size 128
    """
    from lmdeploy.lite.apis.calibrate import calibrate

    calibrate(model_path,
              calib_dataset=calib_dataset,
              calib_samples=calib_samples,
              calib_seqlen=calib_seqlen,
              work_dir=workspace)
    if method == 'kvint8':
        from lmdeploy.lite.apis.kv_qparams import main as run_kv_qparams
        assert num_bits == 8
        run_kv_qparams(workspace,
                       turbomind_dir,
                       kv_bits=num_bits,
                       kv_sym=symmetric,
                       num_tp=tp)
        # update the values in `config.ini`
        config_path = os.path.join(turbomind_dir, 'config.ini')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                content = f.read()
                content = re.sub(r'use_context_fmha = \d+',
                                 'use_context_fmha = 1', content)
                content = re.sub(r'quant_policy = \d+', 'quant_policy = 4',
                                 content)
            with open(config_path, 'w') as f:
                f.write(content)
            print(f'Config file is updated: {config_path}')
    elif method == 'awq':
        from lmdeploy.lite.apis.auto_awq import auto_awq
        assert num_bits == 4
        auto_awq(model_path,
                 work_dir=workspace,
                 w_bits=num_bits,
                 w_sym=symmetric,
                 w_group_size=group_size)
