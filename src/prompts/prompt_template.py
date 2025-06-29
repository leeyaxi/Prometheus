PROMPTS = {
    "bio_qa_zh_question": """
    请基于以下对话上下文，简洁地把用户最新问题改写成完整问题：
    - 如果问题不相关可以不用改写问题，与最新问题保持一致即可
    - 无需解释改写后的问题从何而来
    - 生成一个改写的完整问题即可
    - 改写后的问题长度与原问题字数差距不应该太大
    - 无需说明

    对话上下文：
    {context_str}

    最新问题：
    {question}

    改写后的完整问题：
    
    """,

    "bio_qa_zh_refine": """
    参考已有答案，结合用户问题，给出合理的回答：
    - 已有答案不一定与问题高度相关，如果不相关可以自行回答
    - 回答一个结果即可，无需多次思考
    - 不要从已有答案中直接摘抄文字作为回答
    - 如果已有答案高度相关，可以重新组织语言生成回答
    - 如果不知道答案是什么，直接回答不知道即可
    - 无需说明
    
    用户问题：
    {question}
    
    已有答案：
    {existing_answer}
    
    新文档内容：
    {context_str}
    
    """
}
