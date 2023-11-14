import os
import subprocess
import tempfile
import time
from subprocess import Popen

import fire

import docker


def print_log(log_file: str, head: str = ''):
    print(head)
    with open(log_file, 'r') as f:
        print(f.read())


def test(model_path: str, workdir: str = None):
    """Start triton server and test triton client
    Args:
        model_path (str): The turbomind model directory.
        workdir (str): The working directory to save results.
    """
    if workdir is None:
        workdir = tempfile.TemporaryDirectory().name
    workdir = os.path.abspath(workdir)
    bash_path = os.path.join(model_path, 'service_docker_up.sh')
    os.makedirs(workdir, exist_ok=True)
    server_cmd = [f'bash {bash_path}']
    current_dir = os.path.abspath(os.path.dirname(__file__))
    server_log = os.path.join(workdir, 'triton_server.log')
    client_log = os.path.join(workdir, 'triton_client.log')
    client_cmd = [
        f'docker run -it --rm --gpus \'"device=7"\' '
        '--network host '
        f'-v {workdir}:/root/workspace/workdir '
        f'-v {current_dir}/test_triton_client.py:/opt/test_triton_client.py '
        f'openmmlab/lmdeploy:debug-ci '
        f'python3 /opt/test_triton_client.py '
        f'--workdir /root/workspace/workdir'
    ]
    with open(server_log, 'w') as f_server, open(client_log, 'w') as f_client:
        print('Starting triton server ...')
        proc_server = Popen(server_cmd,
                            stdout=f_server,
                            stderr=f_server,
                            shell=True,
                            encoding='utf-8',
                            text=True)
        time.sleep(60)  # wait triton server to start up
        print('Starting running triton client ...')
        ret = subprocess.run(client_cmd,
                             stdout=f_client,
                             stderr=f_client,
                             shell=True,
                             text=True,
                             check=False)
        print(f'Return code from triton client: {ret.returncode}')
        success = (ret.returncode == 0) and (proc_server.returncode is None)

    docker_client = docker.from_env()
    # 通过容器name获取容器，在service_docker_up.sh中设置的
    try:
        server_container = docker_client.containers.get('lmdeploy')
        server_container.stop()
    except Exception:
        pass

    print_log(server_log,
              '\n============== Triton Server Log ==============\n')
    print_log(client_log,
              '\n============== Triton Client Log ==============\n')
    print(f'Finish with success = {success}')
    assert success


if __name__ == '__main__':
    fire.Fire(test)
