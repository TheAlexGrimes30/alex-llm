import json
from pathlib import Path

from tokenizers import Tokenizer


class RewardDataset:
    """
    Reward model dataset (pairwise preference learning)

    format:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }
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

                    if all(k in obj for k in ["prompt", "chosen", "rejected"]):
                        samples.append(obj)

        return samples

    def _format(self, prompt: str, answer: str) -> str:
        return f"<user>\n{prompt}\n<assistant>\n{answer}"

    def _encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)[
               : self.max_length
               ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, int]:
        sample = self.samples[idx]

        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_text = self._format(prompt, chosen)
        rejected_text = self._format(prompt, rejected)

        chosen_tokens = self._encode(chosen_text)
        rejected_tokens = self._encode(rejected_text)

        return {
            "chosen_input_ids": chosen_tokens,
            "rejected_input_ids": rejected_tokens,
        }

