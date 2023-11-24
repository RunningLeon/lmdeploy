import os
import subprocess
from time import sleep

import allure
import pytest
from utils.run_client_chat import commondLineTest
from utils.run_restful_chat import restfulOpenAiChatTest

TYPE = 'api_client'
HTTP_PREFIX = 'http://0.0.0.0:'


@pytest.mark.smoke
@pytest.mark.restful_api
@pytest.mark.internlm_chat_7b
@allure.story('internlm-chat-7b模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_restful_internlm_chat_7b(config):
    model = 'internlm-chat-7b'
    port = 60006

    result, start_log, chat_log, restful_log, kill_log = run_all_step(
        config, model, port)
    allure.attach.file(start_log, attachment_type=allure.attachment_type.TEXT)
    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    allure.attach.file(restful_log,
                       attachment_type=allure.attachment_type.TEXT)
    allure.attach.file(kill_log, attachment_type=allure.attachment_type.TEXT)
    assert result


@pytest.mark.smoke
@pytest.mark.restful_api
@pytest.mark.Qwen_14B_Chat
@allure.story('Qwen-14B-Chat模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_restful_Qwen_14B_Chat(config):
    model = 'Qwen-14B-Chat'
    port = 60007

    result, start_log, chat_log, restful_log, kill_log = run_all_step(
        config, model, port)
    allure.attach.file(start_log, attachment_type=allure.attachment_type.TEXT)
    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    allure.attach.file(restful_log,
                       attachment_type=allure.attachment_type.TEXT)
    allure.attach.file(kill_log, attachment_type=allure.attachment_type.TEXT)
    assert result


def run_all_step(config, model, port):

    with allure.step('step1 - Launch'):
        start_log, pid = startRestfulApi(config, model, port)
        allure.attach.file(start_log,
                           attachment_type=allure.attachment_type.TEXT)
        result = pid > 0
        if not result:
            return result, None, None, None, None
    sleep(20)

    with allure.step('step2 - command chat regression'):
        chat_result, chat_log = commondLineTest(config, model, TYPE,
                                                HTTP_PREFIX + str(port))
        result = result & chat_result

    with allure.step('step4 - restful_test - openai chat'):
        restful_result, restful_log = restfulOpenAiChatTest(
            config, model, HTTP_PREFIX + str(port))
        result = result & restful_result

    # with allure.step("step5 - restful_test - lmdeploy chat"):
    #  restful_result, restful_log = restfulOpenAiChatTest(config, model, pid)
    #  result = result & restful_result

    with allure.step('step5 - kill'):
        kill_result, kill_log = killByPid(config, model, pid)
        result = result & kill_result

    return result, start_log, chat_log, restful_log, kill_log


def startRestfulApi(config, model, port):
    dst_path = config.get('dst_path')
    log_path = config.get('log_path')

    cmd = [
        'lmdeploy serve api_server ' + dst_path + '/workspace_' + model +
        ' --server_name 0.0.0.0 --server_port ' + str(port) +
        ' --instance_num 32 --tp 1'
    ]
    start_log = os.path.join(log_path, 'start_restful_' + model + '.log')

    with open(start_log, 'w') as f:
        subprocess.run(['pwd'],
                       stdout=f,
                       stderr=f,
                       shell=True,
                       text=True,
                       encoding='utf-8')
        f.writelines('commondLine: ' + ' '.join(cmd) + '\n')

        # convert
        convertRes = subprocess.Popen(cmd,
                                      stdout=f,
                                      stderr=f,
                                      shell=True,
                                      text=True,
                                      encoding='utf-8')
        pid = convertRes.pid

    return start_log, pid


def killByPid(config, model, pid):
    log_path = config.get('log_path')

    cmd = ['kill -9 ' + str(pid)]
    kill_log = os.path.join(log_path, 'kill_' + model + '.log')

    with open(kill_log, 'w') as f:
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

        # 命令校验
        success = convertRes.returncode == 0

    return success, kill_log
