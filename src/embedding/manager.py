import os
import logging
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.utils.common import BaseModule

logger = logging.getLogger(__name__)

class EmbeddingManager(BaseModule):
    def __init__(self, conf: dict):
        super().__init__(conf, "embedding")
        self.model_name = self.module_conf.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.index_path = self.module_conf.get("index_path", "faiss_index")
        self.device = self.conf["infer"].get("device", "cpu")
        self.batch = self.conf["infer"].get("batch", 32)
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name,
                                                     model_kwargs={"device": self.device},
                                                     encode_kwargs={"batch_size": self.batch})

        # 内存索引对象
        self.db: Optional[FAISS] = None

    def build(self, documents: List[Document], reuse: bool = True) -> FAISS:
        """
        构建或加载FAISS索引，如果reuse=True则尝试复用已有索引并增量更新
        """
        if os.path.exists(self.index_path) and reuse:
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.db = FAISS.load_local(self.index_path, self.embedding_model, allow_dangerous_deserialization=True)
            logger.info("Perform incremental embedding and indexing for new docs")
            self._add_new_documents(documents)
        else:
            logger.info("Building FAISS index from scratch")
            self.db = FAISS.from_documents(documents, self.embedding_model)
            self.db.save_local(self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
        return self.db

    def _add_new_documents(self, documents: List[Document]):
        """
        向已加载的索引增量添加新文档（跳过已有文档）
        """
        if self.db is None:
            raise RuntimeError("FAISS index is not loaded, call build_or_load first")

        # 简单策略：通过文本完全匹配排除已有文档，避免重复计算
        existing_texts = set([doc.page_content for doc in self.db.docstore._dict.values()])
        new_docs = [doc for doc in documents if doc.page_content not in existing_texts]
        if not new_docs:
            logger.info("No new documents to add to FAISS index")
            return

        logger.info(f"Adding {len(new_docs)} new documents to FAISS index")
        texts = [doc.page_content for doc in new_docs]
        embeddings = []
        for i in range(0, len(texts), self.batch):
            batch_texts = texts[i:i+self.batch]
            batch_embeds = self.embedding_model.embed_documents(batch_texts)
            embeddings.extend(batch_embeds)

        self.db.add_documents(new_docs, embeddings)
