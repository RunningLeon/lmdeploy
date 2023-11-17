import os
import subprocess

import allure
import pytest


@pytest.mark.smoke
@pytest.mark.convert
@pytest.mark.internlm_chat_7b
@allure.story('internlm-chat-7b模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_model_convert_internlm_chat_7b(config):
    result, convert_log = convert(config, 'internlm-chat-7b',
                                  'internlm-chat-7b')

    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.convert
@pytest.mark.internlm_chat_20b
@allure.story('internlm-chat-20b模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_model_convert_internlm_chat_20b(config):
    result, convert_log = convert(config, 'internlm-chat-20b',
                                  'internlm-chat-20b')

    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.convert
@pytest.mark.Qwen_7B_Chat
@allure.story('Qwen-7B-Chat模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_model_convert_Qwen_7B_Chat(config):
    result, convert_log = convert(config, 'Qwen-7B-Chat', 'qwen-7b')

    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.convert
@pytest.mark.Qwen_14B_Chat
@allure.story('Qwen-14B-Chat模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_model_convert_Qwen_14B_Chat(config):
    result, convert_log = convert(config, 'Qwen-14B-Chat', 'qwen-14b')

    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.convert
@pytest.mark.Qwen_7B_Chat
@allure.story('Qwen-7B-Chat-inner-w4模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_model_convert_Qwen_7B_Chat_inner_w4(config):
    result, convert_log = convert(config, 'Qwen-7B-Chat-inner-w4', 'qwen-7b')

    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.convert
@pytest.mark.Baichuan2_7B_Chat
@allure.story('Baichuan2-7B-Chat-inner-w4模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_model_convert_Baichuan2_7B_Chat_inner_w4(config):
    result, convert_log = convert(config, 'Baichuan2-7B-Chat-inner-w4',
                                  'baichuan2-7b')

    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.convert
@pytest.mark.CodeLlama_7b_hf
@allure.story('CodeLlama-7b-hf模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_model_convert_CodeLlama_7b_hf(config):
    result, convert_log = convert(config, 'CodeLlama-7b-hf', 'codellama')

    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert (result)


def convert(config, model_case, model_name):
    model_path = config.get('model_path')
    dst_path = config.get('dst_path')
    log_path = config.get('log_path')

    if 'w4' in model_case:
        cmd = [
            'lmdeploy convert ' + model_name + ' ' + model_path + '/' +
            model_case + ' --model-format awq --group-size 128 --dst_path ' +
            dst_path + '/workspace_' + model_case
        ]
    else:
        cmd = [
            'lmdeploy convert ' + model_name + ' ' + model_path + '/' +
            model_case + ' --dst_path ' + dst_path + '/workspace_' + model_case
        ]

    convert_log = os.path.join(log_path, 'convert_' + model_case + '.log')

    with open(convert_log, 'w') as f:
        subprocess.run(['pwd'],
                       stdout=f,
                       stderr=f,
                       shell=True,
                       text=True,
                       encoding=True)
        f.writelines('commondLine: ' + ' '.join(cmd) + '\n')

        # convert
        convertRes = subprocess.run(cmd,
                                    stdout=f,
                                    stderr=f,
                                    shell=True,
                                    text=True,
                                    encoding=True)
        # 命令校验
        result = convertRes.returncode == 0

    return result, convert_log
