import os
import json
import glob
import logging
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PDFMinerLoader, CSVLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader,
    Docx2txtLoader
)

from src.utils.common import BaseModule

logger = logging.getLogger(__name__)

SUFFIX_TO_LOADER = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".md":   TextLoader,
    ".docx": Docx2txtLoader,
}


class DocumentLoader(BaseModule):
    def __init__(self, conf: dict):
        super().__init__(conf, "library")
        self.folder_path = self.module_conf.get("path", "docs")
        self.record_file = self.module_conf.get("record_file", "ingested.json")
        self.chunk_size = self.module_conf.get("chunk_size", 1000)
        self.chunk_overlap = self.module_conf.get("chunk_overlap", 200)

    def load(self) -> List[Document]:
        docs = []
        raw_docs = []
        ingested_files = set()

        if os.path.exists(self.record_file):
            with open(self.record_file, 'r') as f:
                ingested_files = set(json.load(f))

        all_files = glob.glob(self.folder_path + "/**/**.*", recursive=True)
        new_files = []

        for file in all_files:
            file = Path(file)
            filename = file.name
            if filename in ingested_files:
                continue

            suffix = file.suffix.lower()
            loader_cls = SUFFIX_TO_LOADER.get(suffix)
            if loader_cls:
                try:
                    loaded = loader_cls(str(file)).load()
                    raw_docs.extend(loaded)
                    new_files.append(filename)
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.warning(f"Unsupported file type {suffix} for file {filename}")

        with open(self.record_file, 'w') as f:
            json.dump(list(ingested_files.union(new_files)), f)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        for doc in raw_docs:
            docs.extend(splitter.split_documents([doc]))

        logger.info(f"Loaded {len(docs)} chunks from {len(new_files)} new files.")
        return docs
