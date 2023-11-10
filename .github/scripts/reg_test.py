from subprocess import PIPE, Popen
from typing import Dict, List

DIALOGUE_SEPARATOR = 'double enter to end input >>>'
PROMPTS = ['Hello', '你好', 'How many days does a week have?', '一周有多少天']
prompts_multi = ['李白是谁？', '他有哪些作品？', '其中哪个作品最受欢迎？']


def parse_dialogue(inputs: str):
    dialogues = inputs.strip()
    if dialogues.endswith(DIALOGUE_SEPARATOR):
        dialogues = dialogues[:-len(DIALOGUE_SEPARATOR)]
    dialogues = dialogues.strip()
    dialogues = dialogues.split(DIALOGUE_SEPARATOR)
    dialogues = [d.strip() for d in dialogues]
    return dialogues[1:]


def run_chat(cmd: List[str], prompts: List[str], multi_rounds: bool = False):
    outputs, errors = [], []
    sep = '\n\n'
    end = sep + 'exit\n\n\n'
    if multi_rounds:
        inputs = '\n\n'.join(prompts) + end
        with Popen(cmd,
                   stdin=PIPE,
                   stdout=PIPE,
                   stderr=PIPE,
                   shell=True,
                   text=True,
                   encoding='utf-8') as proc:
            out, err = proc.communicate(input=inputs)
            outputs = parse_dialogue(out)
            errors = len(outputs) * [err]
            print(proc.returncode)
    else:
        for quest in prompts:
            inputs = quest + end
            with Popen(cmd,
                       stdin=PIPE,
                       stdout=PIPE,
                       stderr=PIPE,
                       shell=True,
                       text=True,
                       encoding='utf-8') as proc:
                out, err = proc.communicate(input=inputs)
                proc.wait()
                if proc.returncode == 0:
                    out = parse_dialogue(out)[0]
                outputs.append(out)
                errors.append(err)
    for q, o, e in zip(prompts, outputs, errors):
        print(20 * '--')
        print(q)
        print(o)
        print(f'err: {e}')
    return outputs, errors


def test_restful_api(model_path: str, prompts, server_kwargs: {},
                     client_kwargs: {}):
    # server 启动成功或失败， 日志
    # client 调用成功否
    # client 回答是否正确 日志
    port = 23444
    server_cmd = [
        f'lmdeploy serve api_server {model_path}', f'--server_port {port}'
    ]
    server_cmd += [f'--{k} {v}' for k, v in server_kwargs.items()]
    client_cmd = [f'lmdeploy serve api_client http://127.0.0.1:{port}']
    client_cmd += [f'--{k} {v}' for k, v in client_kwargs.items()]

    with Popen(server_cmd,
               stdin=PIPE,
               stdout=PIPE,
               stderr=PIPE,
               shell=True,
               text=True) as server_proc:
        outputs, errors = run_chat(client_cmd, prompts_single)
        out, err = server_proc.communicate()
        print()
        print(out, err)


def test_gradio():
    pass

def test_lite():
    # choose model, lite the model, then use cli chat to test
    pass



cmd = ['lmdeploy chat torch /nvme/shared_data/InternLM/internlm-chat-7b']
prompts_single = ['Hello', '你好', 'How many days does a week have?', '一周有多少天']
prompts_multi = ['李白是谁？', '他有哪些作品？', '其中哪个作品最受欢迎？']
# outputs, errors = run_chat(cmd, prompts_single)

model_path = '/mnt/models-new/ningsheng/internlm-chat-7b-turbomind'
test_restful_api(model_path, prompts_single)

server_name = 'x'
server_port = 2
cmd = [f'curl http://{server_name}:{server_port}/v1/chat/interactive',
       '-H "Content-Type: application/json"',
       '-d \'{"prompt": "Hello! How are you?", "session_id": 1, "interactive_mode": true}\''
       ]