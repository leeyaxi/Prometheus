from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.core import Core
import logging

app = FastAPI()
core = Core(conversational=True)
qa_chain = core.qa_chain
chat_histories = {}
logger = logging.getLogger(__name__)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: Request, query: Query):
    client_ip = request.client.host
    history = chat_histories.get(client_ip, [])

    result = qa_chain({"question": query.question, "chat_history": history})
    answer = result["answer"]

    history.append((query.question, answer))
    chat_histories[client_ip] = history[-5:]

    logger.info(f"Client {client_ip} asked: {query.question}")
    logger.info(f"Answer: {answer}")

    return {"answer": answer}
