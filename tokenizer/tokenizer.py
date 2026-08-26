from pathlib import Path
from tokenizers import Tokenizer


class LLMTokenizer:
    def __init__(self):
        tokenizer_path = (
            Path(__file__).resolve().parent / "vocab" / "tokenizer.json"
        )

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                "Tokenizer not found. Run train_tokenizer.py first."
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.unk_token_id = self.tokenizer.token_to_id("<unk>")
        self.bos_token_id = self.tokenizer.token_to_id("<bos>")
        self.eos_token_id = self.tokenizer.token_to_id("<eos>")

        for name, tid in [
            ("<pad>", self.pad_token_id),
            ("<unk>", self.unk_token_id),
            ("<bos>", self.bos_token_id),
            ("<eos>", self.eos_token_id),
        ]:
            if tid is None:
                raise ValueError(f"Missing special token in vocab: {name}")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:

        ids = self.tokenizer.encode(text).ids

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:

        if skip_special_tokens:
            special_ids = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }

            ids = [x for x in ids if x not in special_ids]

        return self.tokenizer.decode(ids)

    def encode_chat(self, messages: list[dict]) -> list[int]:
        text = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            text += f"<{role}>\n{content}\n"

        return self.encode(text)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id)


if __name__ == "__main__":
    tokenizer = LLMTokenizer()

    text = "Какие цвета есть у радуги?"

    ids = tokenizer.encode(text)

    print("TEXT:")
    print(text)

    print("\nTOKEN IDS:")
    print(ids)

    print("\nDECODE:")
    print(tokenizer.decode(ids))