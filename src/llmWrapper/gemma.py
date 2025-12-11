from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
from langchain_core.language_models import LLM
from pydantic import ConfigDict
import torch
from typing import Optional, List, Any

class GemmaLLM:
    def __init__(self, role):
        self.model_id = "google/gemma-3-1b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model_config = ConfigDict(arbitrary_types_allowed=True)
        self.role = role
        self.model = Gemma3ForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.quantization_config,
            device_map="auto",
        )

    def call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": self.role,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return result.strip()