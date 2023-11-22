import allure
import pytest
import yaml
from utils.run_client_chat import commondLineTest


@pytest.fixture(scope='class', autouse=True)
def case_config(request):
    case_path = './autotest/chat_prompt_case.yaml'
    print(case_path)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config


def getList():
    case_path = './autotest/chat_prompt_case.yaml'
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return list(case_config.keys())


@pytest.mark.usefixtures('case_config')
@pytest.mark.command_chat
class Test_command_chat:

    @pytest.mark.llama2_chat_7b_w4
    @allure.story('llama2-chat-7b-w4')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_llama2_chat_7b_w4(self, config, case_config, usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'llama2-chat-7b-w4', 'turbomind',
                                           None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.internlm_chat_7b
    @allure.story('internlm-chat-7b')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_internlm_chat_7b(self, config, case_config, usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'internlm-chat-7b', 'turbomind',
                                           None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.internlm_chat_20b
    @allure.story('internlm-chat-20b')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_internlm_chat_20b(self, config, case_config, usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'internlm-chat-20b', 'turbomind',
                                           None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.internlm_chat_20b
    @allure.story('internlm-chat-20b-inner-w4')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_internlm_chat_20b_inner_w4(self, config, case_config,
                                             usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'internlm-chat-20b-inner-w4',
                                           'turbomind', None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.Qwen_7B_Chat
    @allure.story('Qwen-7B-Chat')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_Qwen_7B_Chat(self, config, case_config, usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'Qwen-7B-Chat', 'turbomind', None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.Qwen_7B_Chat
    @allure.story('Qwen-7B-Chat-inner-w4')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_Qwen_7B_Chat_inner_w4(self, config, case_config, usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'Qwen-7B-Chat-inner-w4',
                                           'turbomind', None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.Qwen_14B_Chat
    @allure.story('Qwen-14B-Chat')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_Qwen_14B_Chat(self, config, case_config, usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'Qwen-14B-Chat', 'turbomind', None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.Qwen_14B_Chat_inner_w4
    @allure.story('Qwen-14B-Chat-inner-w4')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_Qwen_14B_Chat_inner_w4(self, config, case_config, usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'Qwen-14B-Chat-inner-w4',
                                           'turbomind', None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.Baichuan2_7B_Chat
    @allure.story('Baichuan2-7B-Chat')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_Baichuan2_7B_Chat(self, config, case_config, usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'Baichuan2-7B-Chat', 'turbomind',
                                           None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.Baichuan2_7B_Chat
    @allure.story('Baichuan2-7B-Chat-inner-w4')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_Baichuan2_7B_Chat_inner_w4(self, config, case_config,
                                             usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'Baichuan2-7B-Chat-inner-w4',
                                           'turbomind', None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.CodeLlama_7b_Instruct_hf
    @allure.story('CodeLlama-7b-Instruct-hf')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_CodeLlama_7b_Instruct_hf(self, config, case_config,
                                           usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'CodeLlama-7b-Instruct-hf',
                                           'turbomind', None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.CodeLlama_7b_Instruct_hf
    @allure.story('CodeLlama-7b-Instruct-hf-inner-w4')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_CodeLlama_7b_Instruct_hf_inner_w4(self, config, case_config,
                                                    usercase):
        result, chat_log = commondLineTest(
            config, usercase, case_config.get(usercase),
            'CodeLlama-7b-Instruct-hf-inner-w4', 'turbomind', None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.Llama_2_7b_chat_hf
    @allure.story('Llama-2-7b-chat-hf')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_Llama_2_7b_chat_hf(self, config, case_config, usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'Llama-2-7b-chat-hf', 'turbomind',
                                           None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)

    @pytest.mark.Llama_2_7b_chat_hf
    @allure.story('Llama-2-7b-chat-hf-inner-w4')
    @pytest.mark.parametrize('usercase', getList())
    def test_chat_Llama_2_7b_chat_hf_inner_w4(self, config, case_config,
                                              usercase):
        result, chat_log = commondLineTest(config, usercase,
                                           case_config.get(usercase),
                                           'Llama-2-7b-chat-hf-inner-w4',
                                           'turbomind', None)
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
        assert (result)
