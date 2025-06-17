import re

def text_postprocess(text: str, language='en') -> str:
    # 1. 统一编码，清除非可打印字符
    text = re.sub(r'[^\x20-\x7E\n]', '', text)

    # 2. 修复 OCR 断词：处理 -\n + 缩进/空格 + 小写字母
    text = re.sub(r'-\n\s*([a-z])', r'\1', text)  # Ox-\nford -> Oxford
    text = re.sub(r'-\s*([a-z])', r'\1', text)  # 有时没换行，直接断词也合并

    # 3. 删除模板字段
    header_patterns = [
        r'^WO\s?\d{4}/\d{6}(?:\s?[A-Z]\d?)?',
        r'^\(?\d{1,3}\)?\s',
        r'^INTERNATIONAL\b.*$',
        r'^World Intellectual Property Organization.*$',
        r'^Organization International Bureau.*$',
        r'^PCT/.*$',
        r'^(C|G)[0-9][A-Z]?\s\d+/\d+\s?\(\d{4}',
    ]
    for pat in header_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE | re.MULTILINE)

    # 4. 删除纯数字或乱码行
    lines = text.splitlines()
    filtered = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r'\d{4,}', line):
            continue
        if re.fullmatch(r'[A-Z]{5,}', line):
            continue
        if re.fullmatch(r'[>a-zA-Z]{1,3}', line):
            continue
        filtered.append(line)

    # 5. 合并非标题行的换行
    merged = []
    for i, line in enumerate(filtered):
        if i < len(filtered) - 1 and not re.match(r'^[A-Z ]{5,}$', line) and not re.match(r'^[A-Z ]{5,}$',
                                                                                          filtered[i + 1]):
            merged.append(line + ' ')
        else:
            merged.append(line + '\n')

    text = ''.join(merged)

    # 6. OCR 拼写修复
    ocr_fixes = {
        r'\bW0\s?(\d{4}/\d+)\b': r'WO \1',
        r'\blnternational\b': 'International',
        r'\b1nternational\b': 'International',
        r'\bPub[’\'`]?n?lication\b': 'Publication',
        r'\bGOIN\b': 'G01N',
        r'\bOx\s*ford\b': 'Oxford',  # 加强保险
    }
    for pat, repl in ocr_fixes.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    # 7. 清理多余空格和换行
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +\n', '\n', text)
    text = text.strip()

    return text

def clean_docx_numbers(text: str) -> str:
    text = re.sub(r'\b[A-Z]=', '', text)

    # 处理连词断开（如 Ox- ford -> Oxford）
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # 把换行替换成空格
    text = text.replace('\n', ' ')

    # 多个空格合成一个
    text = re.sub(r'\s+', ' ', text)

    # 去掉前后空格
    text = text.strip()

    return text


import re

def clean_chinese_docx_text(text: str) -> str:

    patterns_to_remove = [
        r'项目名称：.*',
        r'项目编号：.*',
        r'页码',
        r'文件名称：.*',
        r'文件编号：.*',
        r'修订记录.*',
        r'目录.*',
        r'^\d+/\d+$',
        r'^\s*$',
    ]
    for pat in patterns_to_remove:
        text = re.sub(pat, '', text, flags=re.MULTILINE)

    text = re.sub(r'[ \t]+', ' ', text)

    # 3. 替换多余空行为一个换行
    text = re.sub(r'\n{2,}', '\n', text)

    text = re.sub(r'\s*\n\s*', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    # 4. 删除所有非中英文数字、标点符号及换行之外的字符（去除乱码）
    # 中英文数字和常用标点的unicode范围
    allowed_chars = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9，。！？、,.!?\n\r\-:：；;（）()\[\]《》“”"\'’‘\s]')
    text = allowed_chars.sub('', text)

    # 5. 合并断开的英文单词（如 Ox- ford => Oxford）
    text = re.sub(r'-\s*([A-Za-z])', r'\1', text)

    # 6. 行首尾空格清理
    lines = [line.strip() for line in text.splitlines() if line.strip() != '']

    # 7. 保留段落结构（每行视为一个段落），重新组合
    text = '\n'.join(lines)

    return text

