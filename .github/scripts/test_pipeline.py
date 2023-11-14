# Copyright (c) OpenMMLab. All rights reserved.
import os
import subprocess
from collections import OrderedDict
from subprocess import PIPE, Popen
from typing import List

import fire
import tqdm
import yaml
from tqdm import tqdm


def parse_dialogue(inputs: str):
    sep = 'double enter to end input >>>'
    dialogues = inputs.strip()
    if dialogues.endswith(sep):
        dialogues = dialogues[:-len(sep)]
    dialogues = dialogues.strip()
    dialogues = dialogues.split(sep)
    dialogues = [d.strip() for d in dialogues]
    return dialogues[1:]


def validate_chat_one_round_en(cmd, prompts, log_file):
    ret = True
    with open(log_file, 'a') as f:
        f.write('=== Run validate_chat_one_round_en ===\n')
        f.flush()
        questions = [p['question'] for p in prompts]
        outputs, errors = run_chat(cmd, questions)
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


def run_cmd(cmd, log_file, msg=''):
    with open(log_file, 'a') as f:
        f.write(f'========  {msg}  ========' + '\n')
        f.write('\\'.join(cmd) + '\n')
        f.flush()
        ret = subprocess.run(cmd,
                             stdout=f,
                             stderr=f,
                             shell=True,
                             text=True,
                             encoding='utf-8',
                             check=True)
        return ret.returncode


def filter_pipelines(model_configs, models, precisions, engines):
    model_pipelines = []
    for model_cfg in model_configs:
        model_name = model_cfg['model_name']
        if isinstance(models, list) and model_name not in models:
            print(f'Skip all pipelines for model [{model_name}]')
            continue
        model_path = model_cfg['model_path']
        for pipeline in model_cfg['pipelines']:
            prec = pipeline['precision']
            if prec not in precisions:
                print(f'Skip pipeline[{prec}] for model [{model_name}]')
                continue
            eng = pipeline['engine']
            if eng not in engines:
                print(f'Skip pipeline[{eng}] for model [{model_name}]')
                continue
            pipeline['model_name'] = model_name
            pipeline['model_path'] = model_path
            model_pipelines.append(pipeline)
    return model_pipelines


class RunPipeline(object):

    def __init__(self, pipeline, workspace, model_root):
        self.pipeline = pipeline
        self.model_name = pipeline['model_name']
        self.model_path = os.path.join(model_root, pipeline['model_path'])
        self.engine = pipeline['engine']
        self.precision = pipeline['precision']
        self.workspace = os.path.join(workspace, self.model_name, self.engine,
                                      self.precision)
        self.source_model_path = self.model_path
        self.results = OrderedDict()
        os.makedirs(self.workspace, exist_ok=True)

    def run_lite(self) -> bool:
        lite_cfg = self.pipeline.get('lite', {})
        lite_log = os.path.join(self.workspace, 'lite.txt')
        if not lite_cfg:
            self.results['lite'] = '-'
            return True
        if self.engine == 'turbomind':
            # temp dir and later used to convert
            workdir = os.path.join(self.workspace, 'lite')
            self.source_model_path = workdir
        else:
            workdir = self.workspace

        if 'w4a16' in lite_cfg:
            calibrate_args = lite_cfg['w4a16']['calibrate']
            autoawq_args = lite_cfg['w4a16']['auto_awq']
            calib_cmd = [
                f'lmdeploy lite calibrate {self.model_path}',
                f'--work_dir {workdir}'
            ]
            calib_cmd += [f'--{k} {v}' for k, v in calibrate_args.items()]
            ret = run_cmd(calib_cmd, lite_log, msg='Run w4a16 calibrate step')
            if ret != 0:
                self['lite'] = False
                return False
            autoawq_cmd = [
                f'lmdeploy lite auto_awq {self.model_path} {workdir}'
            ]
            autoawq_cmd += [f'--{k} {v}' for k, v in autoawq_args.items()]
            ret = run_cmd(autoawq_cmd, lite_log, msg='Run w4a16 auto_awq step')
            if ret != 0:
                self['lite'] = False
                return False
        else:
            raise RuntimeError(f'Not supported lite: {lite_cfg}')
        self.results['lite'] = True
        return True

    def run_convert(self):
        convert_cfg = self.pipeline.get('convert', {})
        convert_log = os.path.join(self.workspace, 'convert.txt')
        if not convert_cfg:
            self.results['convert'] = '-'
            return True

        convert_cmd = [
            f'lmdeploy convert {self.model_name} {self.source_model_path}',
            f'--dst_path {self.workspace}'
        ]
        convert_cmd += [f'--{k} {v}' for k, v in convert_cfg.items()]
        ret = run_cmd(convert_cmd, convert_log, msg='Run convert step')
        success = False
        if ret == 0:
            self.results['convert'] = True
        else:
            self.results['convert'] = success = True
        return success

    def run_chat(self):
        chat_cfg = self.pipeline.get('chat', {})
        chat_log = os.path.join(self.workspace, 'chat.txt')
        if not chat_cfg:
            self.results['chat'] = '-'
            return True
        chat_args = chat_cfg['chat_args']
        validators = chat_cfg['validators']

        model_path = self.workspace
        if self.engine == 'torch' and self.precision == 'fp16':
            model_path = self.model_path
        chat_cmd = [f'lmdeploy chat {self.engine} {model_path}']
        chat_cmd += [f'--{k} {v}' for k, v in chat_args.items()]
        success = True
        for idx, validator in enumerate(validators):
            f_name = validator.pop('name')
            f_kwargs = validator
            f_kwargs['log_file'] = chat_log
            f = eval(f_name)
            ret = f(chat_cmd, **f_kwargs)
            success = success and ret
        self.results['chat'] = success
        return success

    def run_restful_api(self):
        restfulapi_cfg = self.pipeline.get('serve', {}).get('restful_api', {})
        api_server_log = os.path.join(self.workspace, 'api_server.txt')
        api_client_log = os.path.join(self.workspace, 'api_client.txt')
        if not restfulapi_cfg:
            self.results['restfulapi'] = '-'
            return True
        api_server_args = restfulapi_cfg['api_server']
        api_client_args = restfulapi_cfg['api_client']
        server_port = api_server_args.pop('server_port', 23333)
        server_cmd = [
            f'lmdeploy serve api_server {self.workspace} --server_port {server_port}'
        ]
        server_cmd += [f'--{k} {v}' for k, v in api_server_args.items()]
        url = f'http://127.0.0.1:{server_port}'
        client_cmd = [f'lmdeploy serve api_client {url}']
        client_cmd += [f'--{k} {v}' for k, v in api_client_args.items()]
        success = True
        validators = restfulapi_cfg['validators']
        with open(api_server_log, 'w') as f_server:
            with Popen(server_cmd,
                       stdout=f_server,
                       stderr=f_server,
                       shell=True,
                       encoding='utf-8',
                       text=True) as server:
                for idx, validator in enumerate(validators):
                    f_name = validator.pop('name')
                    f_kwargs = validator
                    f_kwargs['log_file'] = api_client_log
                    f = eval(f_name)
                    ret = f(client_cmd, **f_kwargs)
                    success = success and ret
                assert server.returncode is None
                server.kill()

        self.results['restfulapi'] = success
        return success

    def run_gradio(self):
        gradio_cfg = self.pipeline.get('serve', {}).get('gradio', {})
        gradio_log = os.path.join(self.workspace, 'gradio.txt')
        if not gradio_cfg:
            self.results['gradio'] = '-'
            return True
        gradio_args = gradio_cfg['gradio']
        success = True
        gradio_cmd = [f'lmdeploy serve gradio {self.workspace}']
        gradio_cmd += [f'--{k} {v}' for k, v in gradio_args.items()]

        self.results['gradio'] = success
        return success

    def run(self):

        try:
            ret = self.run_lite()
            if not ret:
                return False
            ret = self.run_convert()
            if not ret:
                return False
        except Exception as e:
            print(e)
            return False

        success = True

        try:
            ret = self.run_chat()
            if not ret:
                success = False
        except Exception as e:
            print(e)

        try:
            ret = self.run_restful_api()
            if not ret:
                success = False
        except Exception as e:
            print(e)

        try:
            ret = self.run_gradio()
            if not ret:
                success = False
        except Exception as e:
            print(e)
        return success

    def report(self):
        output = [
            self.model_name,
            self.engine,
            self.precision,
        ]
        targets = ['lite', 'convert', 'chat', 'restfulapi', 'gradio']
        targets_results = [self.results.get(it, '-') for it in targets]
        all_pass = all([t in ['-', True] for t in targets_results])
        output.extend(targets_results)
        output.append(all_pass)
        return output


def test(config_file: str,
         select_model: List[str] = None,
         precision=['fp16'],
         engine: str = None):

    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    models_root = cfg['models_root']
    workspace_root = cfg['workspace_root']
    host_models_root = cfg['host_models_root']
    host_workspace_root = cfg['host_workspace_root']

    test_models = cfg['models']
    if select_model is None:
        select_model = ['torch', 'turbomind']
    model_pipelines = filter_pipelines(test_models, select_model, precision,
                                       engine)
    for pipeline in tqdm.tqdm(model_pipelines):
        runner = RunPipeline(pipeline, workspace_root, models_root)
        ret = runner.run()
        if not ret:
            print(
                f'Failed to run {runner.model_name} {runner.engine} {runner.precision}'
            )
        res = runner.report()
        print(res)

    # csv_path = os.path.join(workspace_root, 'convert.csv')
    # with open(csv_path, 'w') as f:
    #     header = ['model_name', 'engine', 'precision', 'lite', 'convert', 'all_pass']
    #     f.write(','.join(header) + '\n')
    #     for res in results:
    #         f.write(','.join([res[_] for _ in header]) + '\n')
    # print(f'Saved to {csv_path}')


if __name__ == '__main__':
    fire.Fire(test)
