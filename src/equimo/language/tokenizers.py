from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

DEFAULT_TOKENIZER_REPOSITORY = (
    "https://huggingface.co/poiretclement/equimo/resolve/main/models/tokenizers"
)


def _require_tensorflow_text():
    try:
        import tensorflow as tf  # ty: ignore[unresolved-import]
        import tensorflow_text  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "`tensorflow` and `tensorflow_text` are required for language "
            "tokenizers. Install Equimo with the `language` extra."
        ) from exc
    return tf, tensorflow_text


class SentencePieceTokenizer:
    """SentencePiece tokenizer wrapper used by Equimo language checkpoints."""

    def __init__(
        self,
        identifier: Optional[str] = None,
        url: Optional[str] = None,
        path: Optional[str] = None,
        repository: str = DEFAULT_TOKENIZER_REPOSITORY,
    ):
        if not identifier and not path and not url:
            raise ValueError("At least one of identifier, path, or url is required.")

        if identifier:
            url = f"{repository}/{identifier}.model"
        if url:
            model_file_path = self.download(url)
        else:
            assert path
            model_file_path = Path(path).expanduser()
            if not model_file_path.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {path}")

        _, tensorflow_text = _require_tensorflow_text()
        with open(model_file_path, "rb") as f:
            model = f.read()
        self.tokenizer = tensorflow_text.SentencepieceTokenizer(
            model=model, add_eos=False, add_bos=False
        )

    def download(self, url: str) -> Path:
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("Tokenizer URL must start with http:// or https://.")

        parsed_url = urlparse(url)
        fname = Path(parsed_url.path).name
        cache_dir = Path("~/.cache/equimo/tokenizers").expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_file_path = cache_dir / fname

        if not model_file_path.exists():
            response = requests.get(url)
            response.raise_for_status()
            with open(model_file_path, "wb") as f:
                f.write(response.content)

        return model_file_path

    def encode(self, input_text, max_length: int = 64, lowercase: bool = True):
        """Return ``(token_ids, padding_mask)`` arrays.

        ``padding_mask`` uses ``1`` for padding tokens and ``0`` for valid
        tokens, matching :class:`TextTransformerEncoder`.
        """
        tf, _ = _require_tensorflow_text()
        text = tf.strings.lower(input_text) if lowercase else input_text
        tokens = self.tokenizer.tokenize(text).to_tensor()
        curr_len = tokens.shape[1]
        if curr_len > max_length:
            tokens = tokens[:, :max_length]
        else:
            padding_len = max_length - curr_len
            tokens = tf.pad(tokens, [[0, 0], [0, padding_len]], constant_values=0)
        padding_mask = tf.cast(tokens == 0, tf.int32)
        return tokens.numpy(), padding_mask.numpy()
