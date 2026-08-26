import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import LLMTokenizer


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
            tokenizer: LLMTokenizer,
            context_length: int = 2048,
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

        self.tokens = self._tokenize_corpus()

        self.num_samples = (
                len(self.tokens)
                // self.context_length
        )

        if self.num_samples == 0:
            raise ValueError(
                "Dataset слишком маленький для "
                f"context_length={self.context_length}. "
                f"Всего токенов: {len(self.tokens)}"
            )

    def _load_documents(self) -> Iterator[str]:
        """
        Последовательно читает документы.
        Поддерживаемые форматы:
            .txt
            .jsonl

        Для JSONL ожидается:
            {"text": "..."}
        """

        files = sorted(
            file
            for file in self.data_dir.rglob("*")
            if file.is_file()
        )

        for file in files:
            if file.suffix.lower() == ".txt":
                text = file.read_text(
                    encoding="utf-8"
                ).strip()

                if text:
                    yield text

            elif file.suffix.lower() == ".jsonl":

                with file.open("r", encoding="utf-8") as f:
                    for line_number, line in enumerate(
                            f,
                            start=1,
                    ):
                        line = line.strip()

                        if not line:
                            continue

                        try:
                            obj = json.loads(line)

                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                "Некорректный JSONL:\n"
                                f"file={file}\n"
                                f"line={line_number}"
                            ) from exc

                        text = obj.get("text")

                        if (
                                isinstance(text, str)
                                and text.strip()
                        ):
                            yield text.strip()

    def _tokenize_corpus(
            self,
    ) -> torch.Tensor:
        """
        Создает единый поток токенов.

        Каждый документ:

            document tokens + EOS

        BOS для каждого документа не добавляется.
        """

        all_tokens: list[int] = []

        document_count = 0

        for text in self._load_documents():

            document_count += 1

            token_ids = self.tokenizer.encode(text)

            if not token_ids:
                continue

            all_tokens.extend(token_ids)

            all_tokens.append(
                self.tokenizer.eos_token_id
            )

        if document_count == 0:
            raise ValueError(
                f"В директории {self.data_dir} "
                "не найдено документов."
            )

        if not all_tokens:
            raise ValueError(
                "После токенизации corpus пуст."
            )

        return torch.tensor(
            all_tokens,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
            self,
            idx: int,
    ) -> dict[str, torch.Tensor]:

        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range."
            )

        start = (
                idx
                * self.context_length
        )

        end = (
                start
                + self.context_length
        )

        input_ids = self.tokens[
                    start:end
                    ]

        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }