import os
import subprocess

import allure
import pytest


@pytest.mark.quantization
@pytest.mark.CodeLlama_7b_Instruct_hf
@allure.story('CodeLlama-7b-Instruct-hf')
def test_quantization_CodeLlama_7b_Instruct_hf(config):
    result, quantization_param_log, quantization_log = quantization(
        config, 'CodeLlama-7b-Instruct-hf-inner-w4',
        'CodeLlama-7b-Instruct-hf')
    allure.attach.file(quantization_param_log,
                       attachment_type=allure.attachment_type.TEXT)
    if quantization_log is not None:
        allure.attach.file(quantization_log,
                           attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.quantization
@pytest.mark.Llama_2_7b_chat_hf
@allure.story('Llama-2-7b-chat-hf')
def test_quantization_Llama_2_7b_chat_hf(config):
    result, quantization_param_log, quantization_log = quantization(
        config, 'Llama-2-7b-chat-hf-inner-w4', 'Llama-2-7b-chat-hf')
    allure.attach.file(quantization_param_log,
                       attachment_type=allure.attachment_type.TEXT)
    if quantization_log is not None:
        allure.attach.file(quantization_log,
                           attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.quantization
@pytest.mark.internlm_chat_20b
@allure.story('internlm-chat-20b')
def test_quantization_internlm_chat_20b(config):
    result, quantization_param_log, quantization_log = quantization(
        config, 'internlm_chat_20b-inner-w4', 'internlm_chat_20b')
    allure.attach.file(quantization_param_log,
                       attachment_type=allure.attachment_type.TEXT)
    if quantization_log is not None:
        allure.attach.file(quantization_log,
                           attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.quantization
@pytest.mark.Qwen_7B_Chat
@allure.story('Qwen-14B-Chat')
def test_quantization_Qwen_14B_Chat(config):
    result, quantization_param_log, quantization_log = quantization(
        config, 'Qwen-14B-Chat-inner-w4', 'Qwen-14B-Chat')
    allure.attach.file(quantization_param_log,
                       attachment_type=allure.attachment_type.TEXT)
    if quantization_log is not None:
        allure.attach.file(quantization_log,
                           attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.quantization
@pytest.mark.Baichuan2_7B_Chat
@allure.story('Baichuan2-7B-Chat')
def test_quantization_Baichuan2_7B_Chat(config):
    result, quantization_param_log, quantization_log = quantization(
        config, 'Baichuan2-7B-Chat-inner-w4', 'Baichuan2-7B-Chat')
    allure.attach.file(quantization_param_log,
                       attachment_type=allure.attachment_type.TEXT)
    if quantization_log is not None:
        allure.attach.file(quantization_log,
                           attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.quantization
@pytest.mark.Qwen_7B_Chat
@allure.story('Qwen-7B-Chat')
def test_quantization_Qwen_7B_Chat(config):
    result, quantization_param_log, quantization_log = quantization(
        config, 'Qwen-7B-Chat-inner-w4', 'Qwen-7B-Chat')
    allure.attach.file(quantization_param_log,
                       attachment_type=allure.attachment_type.TEXT)
    if quantization_log is not None:
        allure.attach.file(quantization_log,
                           attachment_type=allure.attachment_type.TEXT)
    assert (result)


def quantization(config, w4_model_name, origin_model_name):
    model_path = config.get('model_path')
    log_path = config.get('log_path')

    get_param_cmd = [
        'lmdeploy lite calibrate --model ' + model_path + '/' +
        origin_model_name + ' --calib_dataset \'c4\' --calib_samples 128' +
        ' --calib_seqlen 2048 --work_dir ' + model_path + '/' + w4_model_name
    ]
    quantization_cmd = [
        'lmdeploy lite auto_awq --model ' + model_path + '/' +
        origin_model_name + ' --w_bits 4  --w_group_size 128  ' +
        '--work_dir ' + model_path + '/' + w4_model_name
    ]
    quantization_param_log = os.path.join(
        log_path, 'quantization_param_' + w4_model_name + '.log')
    quantization_log = os.path.join(log_path,
                                    'quantization_' + w4_model_name + '.log')

    with allure.step('step1 - get quantization params'):
        with open(quantization_param_log, 'w') as f:
            subprocess.run(['pwd'],
                           stdout=f,
                           stderr=f,
                           shell=True,
                           text=True,
                           encoding=True)
            f.writelines('commondLine get_param_cmd: ' +
                         ' '.join(get_param_cmd) + '\n')
            # get params
            getParamsRes = subprocess.run(get_param_cmd,
                                          stdout=f,
                                          stderr=f,
                                          shell=True,
                                          text=True,
                                          encoding=True)
            result = getParamsRes.returncode == 0
        if result is False:
            return result, quantization_param_log, None

    with allure.step('step2 - quantization'):
        with open(quantization_log, 'w') as f:
            subprocess.run(['pwd'],
                           stdout=f,
                           stderr=f,
                           shell=True,
                           text=True,
                           encoding=True)
            f.writelines('commondLine quantization_cmd: ' +
                         ' '.join(quantization_cmd) + '\n')
            # convert
            quantizationRes = subprocess.run(quantization_cmd,
                                             stdout=f,
                                             stderr=f,
                                             shell=True,
                                             text=True,
                                             encoding=True)
            # 命令校验
            result = quantizationRes.returncode == 0

    return result, quantization_param_log, quantization_log
