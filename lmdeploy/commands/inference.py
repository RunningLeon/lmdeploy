# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional

import click


@click.group('infer')
def inference():
    """Run inference in one session on local machine.

    \b
    Help:
        > lmdeploy infer --help
    """
    pass


@click.command('chat')
@click.argument(
    'model_path',
    type=click.Path(exists=True, readable=True),
)
@click.option('--engine',
              type=click.Choice(['pytorch', 'turbomind']),
              default='turbomind',
              help='Choose which engine to run inference. Default is turbomind'
              )
@click.option('--tokenizer_path',
              type=click.Path(),
              default=None,
              help='Input tokenizer path. Default is None.')
@click.option('--accel',
              type=click.Choice(['deepspeed']),
              default=None,
              help='Model accelerator.')
@click.option('--max_new_tokens',
              type=int,
              default=128,
              help='Input tokenizer path. Default is None.')
@click.option('--temperature',
              type=float,
              default=0.8,
              help='Temperature for sampling.')
@click.option('--top_p', type=float, default=0.95, help='Top p for sampling.')
@click.option('--seed', type=int, default=0, help='Random seed.')
@click.option('--use_fast_tokenizer',
              is_flag=True,
              default=True,
              help="Whether to use fast tokenizer. This argument is directly\
               pass to transformer's ``AutoTokenizer.from_pretrained``. \
               Generally, user should choose to use fast tokenizers. \
               But if using fast raise some error, try to force using \
               a slow one.")
@click.option('--max_alloc',
              type=int,
              default=2048,
              help='Maximum memory to allocate (for deepspeed).')
@click.option(
    '--max_session_len',
    type=int,
    default=None,
    help='Maximum number of tokens allowed for all inference sessions. \
            This include both history and current session.')
@click.option('--log_file',
              type=click.Path(),
              default=None,
              help='Path to log file')
@click.option('--debug',
              is_flag=True,
              default=False,
              help='Whether to enable debug mode.')
@click.option('--adapter', type=str, default=None, help='')
@click.option('--session_id', type=int, default=1, help='')
@click.option('--cap',
              type=click.Choice(['completion', 'infilling', 'chat', 'python']),
              default='chat',
              help='')
@click.option('--sys_instruct', type=str, default=None, help='')
@click.option('--tp', type=int, default=1, help='')
@click.option('--stream_output',
              is_flag=True,
              default=True,
              help='Indicator for streaming output or not. Default is False')
def _chat(model_path: str,
          engine: str = 'turbomind',
          tokenizer_path: Optional[str] = None,
          accel: Optional[str] = None,
          max_new_tokens: int = 128,
          temperature: float = 0.8,
          top_p: float = 0.95,
          seed: int = 0,
          use_fast_tokenizer: bool = True,
          max_alloc: int = 2048,
          max_session_len: int = None,
          log_file: Optional[str] = None,
          debug: bool = False,
          adapter: Optional[str] = None,
          session_id: int = 1,
          cap: str = 'chat',
          sys_instruct: str = None,
          tp: int = 1,
          stream_output: bool = True,
          **kwargs) -> None:
    """Run inference with chat models.

    \b
    Example:
        > lmdeploy infer chat $MODEL_PATH \\
        >   --engine turbomind \\
        >   --max_new_tokens 64 \\
        >   --temperture 0.8 \\
        >   --top_p 0.95 \\
        >   --seed 0
    """
    if engine == 'turbomind':
        from lmdeploy.turbomind.chat import main as run_turbomind_chat
        run_turbomind_chat(model_path,
                           session_id=session_id,
                           cap=cap,
                           sys_instruct=sys_instruct,
                           tp=tp,
                           stream_output=stream_output,
                           **kwargs)
    elif engine == 'pytorch':
        from lmdeploy.pytorch.chat import main as run_pytorch_chat
        run_pytorch_chat(model_path,
                         tokenizer_path=tokenizer_path,
                         accel=accel,
                         max_new_tokens=max_new_tokens,
                         temperature=temperature,
                         top_p=top_p,
                         seed=seed,
                         use_fast_tokenizer=use_fast_tokenizer,
                         max_alloc=max_alloc,
                         max_session_len=max_session_len,
                         log_file=log_file,
                         debug=debug,
                         adapter=adapter)
    else:
        raise RuntimeError(f'Unsupport engine type: {engine}')


inference.add_command(_chat)
