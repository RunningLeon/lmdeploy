import os
import subprocess

import allure
import pytest


@pytest.mark.convert
@pytest.mark.llama2_chat_7b_w4
@allure.story('llama2-chat-7b-w4')
def test_model_convert_llama2_chat_7b_w4(config):
    result, convert_log = convert(config, 'llama2-chat-7b-w4', 'llama2')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.internlm_chat_7b
@allure.story('internlm-chat-7b')
def test_model_convert_internlm_chat_7b(config):
    result, convert_log = convert(config, 'internlm-chat-7b',
                                  'internlm-chat-7b')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.internlm_chat_20b
@allure.story('internlm-chat-20b')
def test_model_convert_internlm_chat_20b(config):
    result, convert_log = convert(config, 'internlm-chat-20b',
                                  'internlm-chat-20b')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.internlm_chat_20b
@allure.story('internlm-chat-20b')
def test_model_convert_internlm_chat_20b_inner_w4(config):
    result, convert_log = convert(config, 'internlm-chat-20b-inner-w4',
                                  'internlm-chat-20b')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.Qwen_7B_Chat
@allure.story('Qwen-7B-Chat')
def test_model_convert_Qwen_7B_Chat(config):
    result, convert_log = convert(config, 'Qwen-7B-Chat', 'qwen-7b')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.Qwen_14B_Chat
@allure.story('Qwen-14B-Chat')
def test_model_convert_Qwen_14B_Chat(config):
    result, convert_log = convert(config, 'Qwen-14B-Chat', 'qwen-14b')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.Qwen_7B_Chat
@allure.story('Qwen-7B-Chat-inner-w4')
def test_model_convert_Qwen_7B_Chat_inner_w4(config):
    result, convert_log = convert(config, 'Qwen-7B-Chat-inner-w4', 'qwen-7b')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.Qwen_14B_Chat
@allure.story('Qwen-14B-Chat-inner-w4')
def test_model_convert_Qwen_14B_Chat_inner_w4(config):
    result, convert_log = convert(config, 'Qwen-14B-Chat-inner-w4', 'qwen-7b')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.Baichuan2_7B_Chat
@allure.story('Baichuan2-7B-Chat')
def test_model_convert_Baichuan2_7B_Chat(config):
    result, convert_log = convert(config, 'Baichuan2-7B-Chat', 'baichuan2-7b')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.Baichuan2_7B_Chat
@allure.story('Baichuan2-7B-Chat-inner-w4')
def test_model_convert_Baichuan2_7B_Chat_inner_w4(config):
    result, convert_log = convert(config, 'Baichuan2-7B-Chat-inner-w4',
                                  'baichuan2-7b')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.CodeLlama_7b_Instruct_hf
@allure.story('CodeLlama-7b-Instruct-hf')
def test_model_convert_CodeLlama_7b_Instruct_hf(config):
    result, convert_log = convert(config, 'CodeLlama-7b-Instruct-hf',
                                  'codellama')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.CodeLlama_7b_Instruct_hf
@allure.story('CodeLlama-7b-Instruct-hf-inner-w4')
def test_model_convert_CodeLlama_7b_Instruct_hf_inner_w4(config):
    result, convert_log = convert(config, 'CodeLlama-7b-Instruct-hf-inner-w4',
                                  'codellama')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.Llama_2_7b_chat_hf
@allure.story('Llama-2-7b-chat-hf')
def test_model_convert_Llama_2_7b_chat_hf(config):
    result, convert_log = convert(config, 'Llama-2-7b-chat-hf', 'llama2')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.convert
@pytest.mark.Llama_2_7b_chat_hf
@allure.story('Llama-2-7b-chat-hf-inner-w4')
def test_model_convert_Llama_2_7b_chat_hf_inner_w4(config):
    result, convert_log = convert(config, 'Llama-2-7b-chat-hf-inner-w4',
                                  'llama2')
    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result


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
                       encoding='utf-8')
        f.writelines('commondLine: ' + ' '.join(cmd) + '\n')
        # convert
        convertRes = subprocess.run(cmd,
                                    stdout=f,
                                    stderr=f,
                                    shell=True,
                                    text=True,
                                    encoding='utf-8')
        # check result
        result = convertRes.returncode == 0

    return result, convert_log
