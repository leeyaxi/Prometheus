import yaml
import logging
from src.library.ingest import DocumentLoader
from src.embedding.manager import EmbeddingManager
from src.llm.llm import LlmLoader
from src.retriever.engine import Engine

def setup_logging(logfile: str):
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

from langchain.chains import ConversationalRetrievalChain

class Core:
    def __init__(self, config_path="config.yaml", conversational=False):
        self.conf = self.load_config(config_path)
        setup_logging(self.conf.get("logging", {}).get("logfile", "rag.log"))
        self.logger = logging.getLogger(__name__)
        self.logger.info("Core initialization started")

        docs = DocumentLoader(self.conf).load()
        embedding_mgr = EmbeddingManager(self.conf)

        self.db = embedding_mgr.build(docs)
        self.model = LlmLoader(self.conf)

        self.engine = Engine(self.conf, self.model, self.db)

        # 根据参数构建对应链
        if conversational:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.model,
                retriever=self.db.as_retriever(),
                return_source_documents=True,
            )
            self.logger.info("Conversational retrieval chain initialized")
        else:
            self.qa_chain = self.engine.build()
            self.logger.info("Retrieval QA chain initialized")

        self.logger.info("Core initialization finished")

    @staticmethod
    def load_config(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

