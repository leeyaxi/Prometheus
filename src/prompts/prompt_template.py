PROMPTS = {
    # Prompt to rewrite follow-up question based on conversation history
    "bio_qa_zh_question": """你将把一个用户的追问问题，结合之前的对话内容，改写成一个完整、独立的问题，以便让 AI 更准确理解上下文。

请参考以下对话记录，并在最后的用户追问基础上进行改写：

【对话记录】
{context_str}

【用户的追问】
{question}

【改写后的完整问题】""",

    # Prompt to refine answer using new document content
    "bio_qa_zh_refine": """你是一位精准的知识问答助手。请结合已有回答与新的文档内容，对答案进行补充与完善，使其更加准确、全面。如果新文档提供的信息有冲突，请优先以新文档为准，并予以说明。

【已有回答】
{existing_answer}

【新文档内容】
{context_str}

【用户问题】
{question}

【改进后的答案】"""
}
