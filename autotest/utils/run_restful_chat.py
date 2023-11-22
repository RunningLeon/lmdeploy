import json
import os

from utils.rule_condition_assert import assert_result

from lmdeploy.serve.openai.api_client import APIClient


def restfulOpenAiChatTest(config, model, url):
    log_path = config.get('log_path')

    restful_log = os.path.join(log_path, 'restful_' + model + '.log')

    file = open(restful_log, 'w')

    testfile = open('./autotest/restful_prompt_file.json', 'r')
    # 读取JSON数据
    data = json.load(testfile)
    testfile.close()

    result = True

    api_client = APIClient(url)
    file.writelines('available_models:' +
                    ','.join(api_client.available_models) + '\n')
    model_name = model

    for case_detail in sorted(data, key=lambda x: x['order']):
        file.writelines('--- case:' + str(case_detail.get('order')) + '---\n')

        messages = []
        for prompt_detail in sorted(case_detail.get('case'),
                                    key=lambda x: x['order']):
            new_prompt = {
                'role': 'user',
                'content': prompt_detail.get('prompt')
            }
            messages.append(new_prompt)
            file.writelines('prompt:' + prompt_detail.get('prompt') + '\n')

            for output in api_client.chat_completions_v1(model=model_name,
                                                         messages=messages):
                output_message = output.get('choices')[0].get('message')
                messages.append(output_message)

                file.writelines('output:' + output_message.get('content') +
                                '\n')

                case_result, reason = assert_result(
                    output, prompt_detail.get('assert'))
                file.writelines('result:' + str(case_result) + ',reason:' +
                                reason + '\n')
                result = result & case_result

    file.close()
    return result, restful_log


if __name__ == '__main__':
    url = 'http://10.140.0.187:60006'
    config = {'log_path': '/home/PJLAB/zhulin1/code/qa_lmdeploy'}
    restfulOpenAiChatTest(config, 'test', url)
