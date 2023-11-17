
def assert_result(input, rule_condition):
    for rule in rule_condition:
        operator = rule.get('operator')
        value = rule.get('value')
        if operator == "in":
            if isinstance(value, list):
                tmpResult = False
                for word in value:
                    if word in input:
                        tmpResult = True
                if tmpResult == False:
                    return False, "keyword doesn't exist: " + ",".join(value)
            else:
                if value not in input:
                    return False, "keyword doesn't exist: " + value
        if operator == "not_in":
            if isinstance(value, list):
                for word in value:
                    if word in input:
                        return False, "keyword shouldn't exist: " + word
            else:
                if value not in input:
                    return False, "keyword doesn't exist: " + value
        if operator == "len_g":
            if len(input) < value:
                return False, "length: " + str(len(input)) + ", should greater than " + str(value)
    return True, ""
    
    
if __name__ == '__main__':
    input = "成都的景点\n您好，以下是成都的景点推荐。"
    condition = [{"operator":"len_g","value":10},{"operator":"not_in","value":["。"]},{"operator":"not_in","value":"。"},{"operator":"in","value":["，"]},{"operator":"in","value":["。"]},{"operator":"len_g","value":10}]
    print(assert_result(input, condition))