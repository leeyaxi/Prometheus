import yaml
import logging

from src.library.ingest import DocumentLoader
from src.embedding.manager import EmbeddingManager
from src.llm.llm import LlmLoader
from src.prompts.prompt_template import PROMPTS
from src.retriever.engine import Engine

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering.chain import load_qa_chain
from langchain.chains.llm import LLMChain


def setup_logging(logfile: str):
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


class Core:
    def __init__(self, config_path="config.yaml", conversational=False):
        self.conf = self.load_config(config_path)
        setup_logging(self.conf.get("logging", {}).get("logfile", "rag.log"))
        self.logger = logging.getLogger(__name__)
        self.logger.info("Core initialization started")

        # Load and embed documents
        docs = DocumentLoader(self.conf).load()
        embedding_mgr = EmbeddingManager(self.conf)
        self.db = embedding_mgr.build(docs)

        # Load language model
        self.model = LlmLoader(self.conf)

        # Load retriever engine
        self.engine = Engine(self.conf, self.model, self.db)

        # Select prompt key
        prompt_key = self.conf["infer"].get("prompt_lang", "bio_qa_zh")
        self.logger.info(f"Using prompt language key: {prompt_key}")

        # Load prompts
        question_template_text = PROMPTS.get(f"{prompt_key}_question")
        refine_template_text = PROMPTS.get(f"{prompt_key}_refine")

        # Fallback if prompts not found
        if question_template_text is None or refine_template_text is None:
            self.logger.warning(f"Prompt keys '{prompt_key}_question' or '{prompt_key}_refine' not found. Falling back to 'bio_qa_zh'.")
            question_template_text = PROMPTS["bio_qa_zh_question"]
            refine_template_text = PROMPTS["bio_qa_zh_refine"]

        # Construct chain
        if conversational:
            # Prompt for rewriting questions
            question_prompt = PromptTemplate(
                input_variables=["context_str", "question"],
                template=question_template_text
            )

            # Prompt for refining answers
            refine_prompt = PromptTemplate(
                input_variables=["existing_answer", "context_str", "question"],
                template=refine_template_text
            )

            # Combine documents chain using refine
            combine_docs_chain = load_qa_chain(
                llm=self.model,
                chain_type="refine",
                question_prompt=question_prompt,
                refine_prompt=refine_prompt,
                document_variable_name="context_str"
            )

            question_generator = LLMChain(llm=self.model, prompt=question_prompt)
            self.qa_chain = ConversationalRetrievalChain(
                retriever=self.db.as_retriever(),
                question_generator=question_generator,
                combine_docs_chain=combine_docs_chain,
                verbose=True,
            )
            self.logger.info("Conversational retrieval chain initialized with refine mode.")
        else:
            self.qa_chain = self.engine.build()
            self.logger.info("Retrieval QA chain initialized (non-conversational).")

        self.logger.info("Core initialization finished")

    @staticmethod
    def load_config(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
