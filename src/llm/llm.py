from typing import Any

from langchain.llms.base import LLM
from pydantic import PrivateAttr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlmLoader(LLM):
    _conf: dict = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    temperature: float = 0.7
    trust_remote_code: bool = True
    device: str = "cuda:0"

    def __init__(self, conf: dict):
        super().__init__()
        self._conf = conf
        self._module_conf = conf.get("llm", {})
        self.model_name = self._module_conf.get("model_name", self.model_name)
        self.temperature = self._module_conf.get("temperature", self.temperature)
        self.trust_remote_code = self._module_conf.get("trust_remote_code", self.trust_remote_code)
        self.device = conf["infer"].get("device", "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code).to(torch.device(self.device))

    @property
    def _llm_type(self) -> str:
        return "transformers"

    def _call(self, prompt: str, stop: list = None) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(torch.device(self.device))
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=self.temperature,
            do_sample=True,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        if stop:
            for s in stop:
                decoded = decoded.split(s)[0]
        return decoded
