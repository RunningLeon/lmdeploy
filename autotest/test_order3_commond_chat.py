import allure
import pytest
from utils.run_client_chat import commondLineTest


@pytest.mark.smoke
@pytest.mark.command_chat
@pytest.mark.internlm_chat_7b
@allure.story('internlm-chat-7b模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_chat_internlm_chat_7b(config):
    result, chat_log = commondLineTest(config, 'internlm-chat-7b', 'turbomind',
                                       None)

    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.command_chat
@pytest.mark.internlm_chat_20b
@allure.story('internlm-chat-20b模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_chat_internlm_chat_20b(config):
    result, chat_log = commondLineTest(config, 'internlm-chat-20b',
                                       'turbomind', None)

    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.command_chat
@pytest.mark.Qwen_7B_Chat
@allure.story('Qwen-7B-Chat模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_chat_Qwen_7B_Chat(config):
    result, chat_log = commondLineTest(config, 'Qwen-7B-Chat', 'turbomind',
                                       None)

    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.command_chat
@pytest.mark.Qwen_14B_Chat
@allure.story('Qwen-14B-Chat模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_chat_Qwen_14B_Chat(config):
    result, chat_log = commondLineTest(config, 'Qwen-14B-Chat', 'turbomind',
                                       None)

    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.command_chat
@allure.story('Baichuan2-7B-Chat-inner-w4模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_chat_Baichuan2_7B_Chat_inner_w4(config):
    result, chat_log = commondLineTest(config, 'Baichuan2-7B-Chat-inner-w4',
                                       'turbomind', None)

    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    assert (result)


@pytest.mark.smoke
@pytest.mark.command_chat
@pytest.mark.CodeLlama_7b_hf
@allure.story('CodeLlama-7b-hf模型')
@allure.tag('ceshitag')
@allure.label('ceshilabel')
def test_chat_CodeLlama_7b_hf(config):
    result, chat_log = commondLineTest(config, 'CodeLlama-7b-hf', 'turbomind',
                                       None)

    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    assert (result)
