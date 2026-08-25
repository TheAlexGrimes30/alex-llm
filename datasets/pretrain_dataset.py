import json
from pathlib import Path

from tokenizers import Tokenizer


class PretrainDataset:
    """
    Dataset для pretraining
    Формат: текст → токены
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

    def _load_data(self) -> list[str]:
        texts = []

        for file in self.data_dir.rglob("*"):
            if file.suffix == ".txt":
                texts.append(file.read_text(encoding="utf-8"))

            elif file.suffix == ".jsonl":
                with file.open("r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        if "text" in obj:
                            texts.append(obj["text"])

        return texts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, int]:
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[: self.max_length]

        return {
            "input_ids": tokens,
            "labels": tokens.copy(),
        }