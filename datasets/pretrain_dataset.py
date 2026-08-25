import json
from pathlib import Path

from tokenizers import Tokenizer
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    Dataset для causal language model pretraining.

    Pipeline:

        .txt / .jsonl
            ↓
        documents
            ↓
        tokenizer
            ↓
        [tokens_doc_1] + EOS
        [tokens_doc_2] + EOS
            ↓
        единый поток токенов
            ↓
        разбиение на последовательности max_length
            ↓
        input_ids / labels
    """

    def __init__(
            self,
            data_dir: str | Path,
            tokenizer: Tokenizer,
            context_length: int = 2048,
            eos_token: str = "<eos>",
            bos_token: str = "<bos>",
    ):

        super().__init__()

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer

        self.context_length = context_length

        if context_length <= 1:
            raise ValueError(
                "max_length должен быть больше 1."
            )

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Директория не существует: {self.data_dir}"
            )

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