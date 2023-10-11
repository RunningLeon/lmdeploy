# Copyright (c) OpenMMLab. All rights reserved.

import click


@click.group()
def serve():
    """Serve models with gradio or fastapi.

    \b
    Help:
        > lmdeploy serve --help
    """
    pass


@click.command('run')
@click.argument('model_path', type=str)
@click.option('--type',
              type=click.Choice(['gradio', 'fastapi']),
              default='gradio',
              help='Choose which type of server to run.')
@click.option('--name',
              type=str,
              default='localhost',
              help='the ip address of gradio server')
@click.option('--port', type=int, default=23333, help='server port')
@click.option('--concur_num',
              type=int,
              default=32,
              help='Number of concurrency or instances for turbomind.')
@click.option('--tp',
              type=int,
              default=1,
              help='tensor parallel for Turbomind')
def _run(model_path: str,
         type: str = 'gradio',
         name: str = 'localhost',
         port: int = 23333,
         concur_num: int = 32,
         tp: int = 1) -> None:
    """Start serving with gradio or fastapi.

    \b
    Example 1:
        > lmdeploy serve run $MODEL_PATH \\
        > --type gradio \\
        > --name 0.0.0.0 \\
        > --port 23333 \\
        > --concur_num 32 \\
        > --tp 1

    \b
    Example 2:
        > lmdeploy serve run $MODEL_PATH \\
        > --type fastapi \\
        > --name 0.0.0.0 \\
        > --port 23333 \\
        > --concur_num 32 \\
        > --tp 1
    """
    if type == 'gradio':
        from lmdeploy.serve.gradio.app import run as run_gradio
        run_gradio(model_path,
                   server_name=name,
                   server_port=port,
                   batch_size=concur_num,
                   tp=tp,
                   restful_api=False)
    elif type == 'fastapi':
        from lmdeploy.serve.openai.api_server import main as run_fastapi
        run_fastapi(model_path,
                    server_name=name,
                    server_port=port,
                    instance_num=concur_num,
                    tp=tp)


@click.command('test')
@click.argument(
    'url',
    type=str,
)
@click.option('--type',
              type=click.Choice(['gradio', 'fastapi', 'triton']),
              default='gradio',
              help='Choose which type of server to run.')
@click.option('--restful_api',
              is_flag=True,
              default=False,
              help='Whether to enable restful api')
@click.option('--session_id',
              type=int,
              default=0,
              help='The identical id of a session. Default is 0')
@click.option('--gradio_server',
              type=str,
              default='0.0.0.0',
              help='the ip address of gradio server')
@click.option('--gradio_port', type=int, default=6006, help='server port')
@click.option('--cap',
              type=click.Choice(['completion', 'infilling', 'chat', 'python']),
              default='chat',
              help='the capability of a model.')
@click.option('--sys_instruct', type=str, default=None, help='')
@click.option('--tp',
              type=int,
              default=1,
              help='The number of GPUs used for tensor parallelism.')
@click.option('--stream_output',
              is_flag=True,
              default=True,
              help='Indicator for streaming output or not. Default is True')
def _test(
    url: str,
    type: str = 'gradio',
    restful_api: bool = False,
    session_id: int = 0,
    gradio_server: str = '0.0.0.0',
    gradio_port: int = 6006,
    tp: int = 1,
    cap: str = 'chat',
    sys_instruct: str = None,
    stream_output: bool = True,
) -> None:
    """Call server on terminal or webui.

    \b
    Example 1:
        > lmdeploy serve test --type fastapi $RESTFUL_API_URL

    \b
    Example 2:
        > lmdeploy serve test $RESTFUL_API_URL \\
        >   --type gradio \\
        >   --restful_api \\
        >   --gradio_server 0.0.0.0 \\
        >   --gradio_port 6006

    \b
    Example 3:
        > lmdeploy serve test $TRITON_SERVER_URL --type gradio

    \b
    Example 4:
        > lmdeploy serve test $TRITON_SERVER_URL --type triton \\
        >   --session_id 1 \\
        >   --cap chat \\
        >   --sys_instruct '' \\
        >   --tp 1
    """
    if type == 'gradio':
        from lmdeploy.serve.gradio.app import run as run_gradio
        run_gradio(url,
                   server_name=gradio_server,
                   server_port=gradio_port,
                   restful_api=restful_api,
                   tp=tp)
    elif type == 'fastapi':
        from lmdeploy.serve.openai.api_client import main as run_fastapi
        run_fastapi(url, session_id=session_id)
    elif type == 'triton':
        from lmdeploy.serve.client import main as run_triton
        run_triton(url,
                   session_id=session_id,
                   cap=cap,
                   sys_instruct=sys_instruct,
                   stream_output=stream_output)


serve.add_command(_run)
serve.add_command(_test)
