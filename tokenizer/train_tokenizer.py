from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

VOCAB_SIZE = 16000

ROOT_DIR = Path(__file__).resolve().parent.parent
PRETRAIN_DIR = ROOT_DIR / "data" / "pretrain"
VOCAB_DIR = Path(__file__).parent / "vocab"

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<user>",
    "<assistant>",
    "<system>",
]


def collect_training_files() -> list[str]:
    files = []

    for ext in ("*.txt", "*.jsonl"):
        files.extend(PRETRAIN_DIR.rglob(ext))

    return [str(f) for f in files]


def train_tokenizer() -> None:
    files = collect_training_files()

    if not files:
        raise RuntimeError(f"No training files found in {PRETRAIN_DIR}")

    VOCAB_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        BPE(unk_token="<unk>")
    )

    tokenizer.normalizer = Sequence([
        NFKC(),
    ])

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print(f"Training tokenizer on {len(files)} files...")

    tokenizer.train(files, trainer)

    save_path = VOCAB_DIR / "tokenizer.json"
    tokenizer.save(str(save_path))

    print(f"Tokenizer saved to {save_path}")


if __name__ == "__main__":
    train_tokenizer()