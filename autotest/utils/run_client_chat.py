import os
from subprocess import PIPE, Popen

from utils.rule_condition_assert import assert_result


def commondLineTest(config, case, case_info, model, type, extra):
    dst_path = config.get('dst_path')
    log_path = config.get('log_path')

    if type == 'api_client':
        cmd = ['lmdeploy serve api_client ' + extra]
    elif type == 'triton_client':
        cmd = ['lmdeploy serve triton_client ' + extra]
    else:
        cmd = ['lmdeploy chat turbomind ' + dst_path + '/workspace_' + model]

    if case == 'session_len_error':
        cmd[0] = cmd[0] + ' --session_len 20'
    chat_log = os.path.join(log_path, 'chat_' + type + '_' + model + '.log')

    file = open(chat_log, 'w')

    returncode = -1
    result = True

    file.writelines('commondLine: ' + ' '.join(cmd) + '\n')

    spliter = '\n\n'
    if model == 'CodeLlama-7b-Instruct-hf':
        spliter = '\n!!\n'
    # join prompt together
    prompt = ''
    for item in case_info:
        prompt += list(item.keys())[0] + spliter
    prompt += 'exit' + spliter

    with Popen(cmd,
               stdin=PIPE,
               stdout=PIPE,
               stderr=PIPE,
               shell=True,
               text=True,
               encoding='utf-8') as proc:
        file.writelines('prompt:' + prompt + '\n')

        outputs, errors = proc.communicate(input=prompt)
        returncode = proc.returncode
        if returncode != 0:
            file.writelines('error:' + errors + '\n')
            result = False
            return result, chat_log

        outputDialogs = parse_dialogue(outputs, model)
        file.writelines('answersize:' + str(len(outputDialogs)) + '\n')

        # 结果判断
        index = 0
        for prompt_detail in case_info:
            if type == 'turbomind':
                output = extract_output(outputDialogs[index], model)
            case_result, reason = assert_result(output, prompt_detail.values())
            file.writelines('prompt:' + list(prompt_detail.keys())[0] + '\n')
            file.writelines('output:' + outputDialogs[index] + '\n')
            file.writelines('result:' + str(case_result) + ',reason:' +
                            reason + '\n')
            index += 1
            result = result & case_result

    file.close()
    return result, chat_log


# 从输出中解析模型输出的对话内容
def parse_dialogue(inputs: str, model: str):
    dialogues = inputs.strip()
    sep = 'double enter to end input >>>'
    dialogues = dialogues.strip()
    dialogues = dialogues.split(sep)
    dialogues = [d.strip() for d in dialogues]
    if 'Llama' in model:
        return dialogues
    return dialogues[1:-1]  # 去除首尾无用字符


def extract_output(output: str, model: str):
    if 'internlm' in model:
        if len(output.split('<|Bot|>: ')) > 2:
            return output.split('<|Bot|>: ')[1]
    if 'Qwen' in model:
        if len(output.split('<|im_start|>assistant')) > 2:
            return output.split('<|im_start|>assistant')[1]
    if 'Baichuan2' in model:
        if len(output.split('<reserved_107> ')) > 2:
            return output.split('<reserved_107> ')[1]
    return output


if __name__ == '__main__':
    input = '成都的景点\n您好，以下是成都的景点推荐。'
    model = 'model'
    print(extract_output(input, model))
