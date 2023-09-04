def full_to_half(s):
    """
    将全角字符转为半角字符
    """
    result = ''
    for c in s:
        if ord(c) == 0x3000:  # 全角空格特殊处理
            result += chr(0x0020)
        elif 0xFF01 <= ord(c) <= 0xFF5E:
            result += chr(ord(c) - 0xfee0)
        else:
            result += c
    return result