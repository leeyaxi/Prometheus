from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStore
from langchain.llms.base import BaseLLM
from src.utils.common import BaseModule
import logging

logger = logging.getLogger(__name__)

class Engine(BaseModule):
    def __init__(self, conf: dict, llm: BaseLLM, db: VectorStore):
        super().__init__(conf, "retriever")
        self.llm = llm
        self.db = db

    def build(self, conversational: bool = False):
        if conversational:
            logger.info("Building ConversationalRetrievalChain")
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.db.as_retriever(),
                return_source_documents=True,
            )
        else:
            chain_type = self.module_conf.get("chain_type", "stuff")
            logger.info("Building RetrievalQA chain")
            from langchain.chains import RetrievalQA
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=chain_type,
                retriever=self.db.as_retriever()
            )
