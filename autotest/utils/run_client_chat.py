import json
import os
from subprocess import Popen, PIPE
from utils.rule_condition_assert import assert_result


def commondLineTest(config, model, type, extra): 
    dst_path = config.get("dst_path")
    log_path = config.get("log_path")

    if type == 'api_client':
        cmd = ['lmdeploy serve api_client ' + extra] 
    elif type == 'triton_client':
        cmd = ['lmdeploy serve triton_client ' + extra] 
    else:
        cmd = ['lmdeploy chat turbomind ' + dst_path + '/workspace_' + model]
    chat_log = os.path.join(log_path, 'chat_' + type + '_' + model + '.log')

    file = open(chat_log, 'w')
    
    testfile = open('./autotest/chat_prompt_file.json', 'r')
    # 读取JSON数据
    data = json.load(testfile)
    testfile.close()

    returncode = -1
    result = True

    file.writelines('commondLine: ' + ' '.join(cmd) + '\n')

    for case_detail in sorted(data, key=lambda x: x['order']):
        # join prompt together
        sorted_case_list = sorted(case_detail.get('case'), key=lambda x: x['order'])
        prompt = "\n\n".join([item["prompt"] for item in sorted_case_list]) + "\n\nexit\n\n"

        with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, text=True, encoding='utf-8') as proc:
            file.writelines("--- case:" + str(case_detail.get('order')) + '---\n')
            file.writelines("prompt:" + prompt + '\n')

            outputs, errors = proc.communicate(input=prompt)
            returncode = proc.returncode
            if returncode != 0:
                file.writelines("error:" + errors + '\n')
                result = False
                return result, chat_log

            outputDialogs = parse_dialogue(outputs)
            file.writelines("answersize:" + str(len(outputDialogs)) + '\n')

            # 结果判断
            index = 0
            for prompt_detail in sorted_case_list:
                case_result, reason = assert_result(outputDialogs[index], prompt_detail.get('assert'))
                file.writelines("prompt:" + prompt_detail.get('prompt') + '\n')
                file.writelines("output:" + outputDialogs[index] + '\n')
                file.writelines("result:" + str(case_result) + ",reason:" + reason + '\n')
                index += 1
                result = result & case_result

    file.close()
    return result, chat_log


# 从输出中解析模型输出的对话内容
def parse_dialogue(inputs: str):
    dialogues = inputs.strip()
    sep = 'double enter to end input >>>'
    dialogues = dialogues.strip()
    dialogues = dialogues.split(sep)
    dialogues = [d.strip() for d in dialogues]
    return dialogues[1:-1] # 去除首尾无用字符


