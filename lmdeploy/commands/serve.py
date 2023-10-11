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
@click.option('--name',
              type=str,
              default='localhost',
              help='the ip address of gradio server')
@click.option('--port', type=int, default=23333, help='server port')
@click.option('--instance_num',
              type=int,
              default=32,
              help='number of instances of turbomind model')
@click.option('--batch_size',
              type=int,
              default=32,
              help='batch size for running Turbomind directly for gradio')
@click.option('--tp',
              type=int,
              default=1,
              help='tensor parallel for Turbomind')
@click.option('--restful_api',
              is_flag=True,
              default=False,
              help='Whether to enable restful api')
@click.option('--use_gradio',
              is_flag=True,
              default=False,
              help='Whether to use gradio or fastapi to start serving')
def _run(model_path: str,
         name: str = 'localhost',
         port: int = 23333,
         instance_num: int = 32,
         batch_size: int = 32,
         tp: int = 1,
         restful_api: bool = False,
         use_gradio: bool = False) -> None:
    """Run serve with gradio or fastapi.

    \b
    Example:
        > lmdeploy serve run $MODEL_PATH \\
        > --max_new_tokens 64 \\
        > --temperture 0.8 \\
        > --top_p 0.95 \\
        > --seed 0
    """
    if use_gradio:
        from lmdeploy.serve.gradio.app import run as run_gradio
        run_gradio(model_path,
                   server_name=name,
                   server_port=port,
                   batch_size=batch_size,
                   tp=tp,
                   restful_api=restful_api)
    else:
        from lmdeploy.serve.openai.api_server import main as run_fastapi
        run_fastapi(model_path,
                    server_name=name,
                    server_port=port,
                    instance_num=instance_num,
                    tp=tp)


@click.command('test')
@click.argument(
    'url',
    type=str,
)
@click.option('--session_id', type=int, default=0, help='')
def _test(
    url: str,
    session_id: int = 0,
) -> None:
    """Inference with openai api client.

    \b
    Example:
        > lmdeploy serve test $URL --session_id 0
    """
    from lmdeploy.serve.openai.api_client import main
    main(url, session_id=session_id)


serve.add_command(_run)
serve.add_command(_test)
