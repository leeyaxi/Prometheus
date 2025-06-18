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
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
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
            temperature=self._module_conf.get("temperature", self.temperature),
            do_sample=False,
            top_p=self._module_conf.get("top_p", self.top_p),
            top_k=self._module_conf.get("top_k", self.top_k),
            min_p=self._module_conf.get("min_p", self.min_p),
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,

        )
        input_len = inputs['input_ids'].shape[1]
        output_ids = outputs[0][input_len:]
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        decoded = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        if stop:
            for s in stop:
                decoded = decoded.split(s)[0]
        return decoded
