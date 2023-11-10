# Copyright (c) OpenMMLab. All rights reserved.
import os
from subprocess import PIPE, Popen
from typing import List
import json
import tqdm
import yaml


DIALOGUE_SEPARATOR = 'double enter to end input >>>'
PROMPTS = ['Hello', '你好', 'How many days does a week have?', '一周有多少天']
ASSERTS = []
prompts_multi = ['李白是谁？', '他有哪些作品？', '其中哪个作品最受欢迎？']


def parse_dialogue(inputs: str):
    dialogues = inputs.strip()
    if dialogues.endswith(DIALOGUE_SEPARATOR):
        dialogues = dialogues[:-len(DIALOGUE_SEPARATOR)]
    dialogues = dialogues.strip()
    dialogues = dialogues.split(DIALOGUE_SEPARATOR)
    dialogues = [d.strip() for d in dialogues]
    return dialogues[1:]


class OneRoundPromptValidator:

    def __init__(self, prompts):
        self.prompts = prompts

    def validate(self, cmd: str, log_file: str, ):
        ret = True
        with open(log_file, 'a') as f:
            f.write('Run OneRoundPromptValidator\n')
            f.flush()
            questions = [p['question'] for p in self.prompts]
            outputs, errors = run_chat(cmd, questions, multi_rounds=False)
            for out, err, pro in zip(outputs, errors, self.prompts):
                success = all([k in out for k in pro['keywords']])
                f.write(f'Question: {pro["question"]}\n')
                f.write(f'Answer: {out}\n')
                f.write(f'Ref Answer: {pro["answer"]}')
                f.write(f'Key words: {pro["keywords"]}')
                f.write(f'Passed: {success}')
                if not success:
                    ret = False
        return ret


def test_chat_oneround_en(cmd, prompts, log_file):
    ret = True
    with open(log_file, 'a') as f:
        f.write('=== Test One Round test ===\n')
        f.flush()
        questions = [p['question'] for p in prompts]
        outputs, errors = run_chat(cmd, questions, multi_rounds=False)
        for out, err, pro in zip(outputs, errors, prompts):
            # validate key words
            success = all([k in out for k in pro['keywords']])
            f.write('=== test one round dialogue ===')
            f.write(f'Question: {pro["question"]}\n')
            f.write(f'Answer: {out}\n')
            f.write(f'Ref Answer: {pro["answer"]}')
            f.write(f'Key words: {pro["keywords"]}')
            f.write(f'Passed: {success}')
            if not success:
                f.write(f'Error: {err}')
                ret = False
    return ret



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
                proc.poll()
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


def test_triton_client(prompts, workdir):
    os.makedirs(workdir, exist_ok=True)
    cmd = ['lmdeploy serve triton_client localhost:33337']

    outputs, errors = run_chat(cmd, prompts)
    out_file = os.path.join(workdir, 'dialogues.json')
    results = [
        dict(question=q, answer=a) for q, a in zip(prompts, outputs)
    ]
    results = {'results': results, 'pass': True}

    with open(out_file, 'w') as f:
        json.dump(results, f)
    print(f'Saved to {out_file}')


