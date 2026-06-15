import json
from pathlib import Path

from tokenizers import Tokenizer


class SFTDataset:
    """
    Supervised Fine-Tuning dataset
    формат: instruction → response
    """

    def __init__(
            self,
            data_dir: str,
            tokenizer: Tokenizer,
            max_length: int = 2048
    ):

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data()

    def _load_data(self) -> list[dict]:
        samples = []

        for file in self.data_dir.rglob("*.jsonl"):
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)

                    if "instruction" in obj and "response" in obj:
                        samples.append(obj)

        return samples

    def _format_prompt(self, instruction: str) -> str:
        return f"<user>\n{instruction}\n<assistant>\n"


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, int]:
        sample = self.samples[idx]

        instruction = sample["instruction"]
        response = sample["response"]

        prompt = self._format_prompt(instruction)
        full_text = prompt + response

        tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        tokens = tokens[: self.max_length]

        labels = tokens.copy()
        prompt_len = len(prompt_tokens)

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": tokens,
            "labels": labels,
        }