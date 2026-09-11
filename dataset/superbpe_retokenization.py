"""Document-boundary GPT-2 -> frozen SuperBPE conversion for the MoE corpus.

The production source stays the pinned GPT-2-tokenized ClimbMix JSONL.  This
module changes only the representation handed to the existing stratifier: each
accepted source document is decoded back to UTF-8 text and re-encoded with the
frozen 8k tokenizer before ``SourceDocument`` is constructed.  Consequently all
scheduler, rolling-mixture, stopping and attribution counters are measured in
SuperBPE tokens.

Installation is intentionally process-local and reversible.  It patches the
single validation hook imported by ``dataset.src.streaming`` and the EOD value
read by its existing packer; the rest of the production/sharding/HF durability
pipeline remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

from dataset import config
from dataset.src import records, streaming

SOURCE_TOKENIZER_ID = "gpt2"
SOURCE_EOD_TOKEN_ID = 50_256
TARGET_TOKENIZER_ID = "superbpe_8000"
TARGET_TOKENIZER_RELATIVE_PATH = "tokenizer/superbpe_8000.json"
TARGET_SEMANTIC_VOCAB_SIZE = 8_000
TARGET_SOURCE_VOCAB_SIZE = 7_992
TARGET_EOD_TOKEN_ID = 7_992
EXPECTED_SPECIAL_TOKENS = {
    7_992: "<|endoftext|>",
    7_993: "<think>",
    7_994: "</think>",
    7_995: "<answer>",
    7_996: "<|reserved_0|>",
    7_997: "<|reserved_1|>",
    7_998: "<|reserved_2|>",
    7_999: "<|reserved_3|>",
}

_DYNAMIC_CONFIG_FIELDS = (
    "CORPUS_SOURCE_TOKENIZER_ID",
    "CORPUS_OUTPUT_TOKENIZER_ID",
    "CORPUS_TOKENIZER_ARTIFACT",
    "CORPUS_TOKENIZER_SHA256",
    "CORPUS_SEMANTIC_VOCAB_SIZE",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_target_tokenizer(path: Path):
    """Load the frozen tokenizer while making reserved specials unreachable from source text.

    The eight reserved/special entries remain part of the semantic vocabulary,
    but ordinary pretraining text must not turn literal strings such as
    ``<think>`` into control-token IDs.  Removing ``added_tokens`` from the
    runtime copy leaves the frozen BPE model/merges untouched and restricts
    document encoding to IDs 0..7991; EOD 7992 is inserted only by the packer.
    """

    try:
        from tokenizers import Tokenizer
    except ImportError as error:  # pragma: no cover - provider packaging guard
        raise RuntimeError(
            "SuperBPE corpus production requires tokenizers==0.23.2"
        ) from error

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("frozen SuperBPE tokenizer JSON must contain an object")
    raw_added = payload.get("added_tokens")
    if not isinstance(raw_added, list):
        raise RuntimeError("frozen SuperBPE tokenizer has no added-token inventory")
    observed: dict[int, str] = {}
    for item in raw_added:
        if not isinstance(item, dict) or item.get("special") is not True:
            raise RuntimeError("frozen SuperBPE added-token inventory is malformed")
        token_id = item.get("id")
        content = item.get("content")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or not isinstance(content, str):
            raise RuntimeError("frozen SuperBPE added-token inventory is malformed")
        observed[token_id] = content
    if observed != EXPECTED_SPECIAL_TOKENS:
        raise RuntimeError(
            f"unexpected SuperBPE special-token inventory: {observed!r}"
        )

    model = payload.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if not isinstance(vocab, dict):
        raise RuntimeError("frozen SuperBPE tokenizer has no BPE vocabulary")
    ids = sorted(value for value in vocab.values() if isinstance(value, int) and not isinstance(value, bool))
    if ids != list(range(TARGET_SOURCE_VOCAB_SIZE)):
        raise RuntimeError(
            "frozen SuperBPE BPE vocabulary must be exactly IDs 0..7991 before specials"
        )

    source_text_payload = dict(payload)
    source_text_payload["added_tokens"] = []
    tokenizer = Tokenizer.from_str(
        json.dumps(source_text_payload, ensure_ascii=False, separators=(",", ":"))
    )
    return tokenizer


class SuperBPERetokenizer:
    """Thread-safe immutable document converter used by parallel source readers."""

    def __init__(self, tokenizer_path: Path) -> None:
        try:
            import tiktoken
        except ImportError as error:  # pragma: no cover - provider packaging guard
            raise RuntimeError(
                "SuperBPE corpus production requires tiktoken==0.14.0"
            ) from error
        self.tokenizer_path = tokenizer_path.resolve(strict=True)
        self.tokenizer_sha256 = _sha256(self.tokenizer_path)
        self._gpt2 = tiktoken.get_encoding("gpt2")
        self._target = _load_target_tokenizer(self.tokenizer_path)

    @property
    def contract(self) -> dict[str, object]:
        return {
            "source_tokenizer_id": SOURCE_TOKENIZER_ID,
            "source_eod_token_id": SOURCE_EOD_TOKEN_ID,
            "output_tokenizer_id": TARGET_TOKENIZER_ID,
            "tokenizer_artifact": TARGET_TOKENIZER_RELATIVE_PATH,
            "tokenizer_sha256": self.tokenizer_sha256,
            "semantic_vocab_size": TARGET_SEMANTIC_VOCAB_SIZE,
            "source_text_vocab_size": TARGET_SOURCE_VOCAB_SIZE,
            "eod_token_id": TARGET_EOD_TOKEN_ID,
            "retokenization": "gpt2_decode_utf8_then_superbpe_encode_before_scheduling",
        }

    def validate_record(self, record: records.ParsedRecord) -> records.ValidationResult:
        validated = records.validate_record(record)
        if not validated.valid or validated.cluster_id not in config.ACCEPTED_CLUSTER_IDS:
            return validated
        assert validated.tokens is not None

        source_ids = list(validated.tokens)
        if source_ids and source_ids[-1] == SOURCE_EOD_TOKEN_ID:
            source_ids.pop()
        if not source_ids:
            return records.ValidationResult(
                False, validated.cluster_id, None, "retokenized_document_empty"
            )

        try:
            source_bytes = self._gpt2.decode_bytes(source_ids)
            text = source_bytes.decode("utf-8", errors="strict")
        except (KeyError, UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(
                f"GPT-2 source document at record byte {record.record_start} failed exact UTF-8 reconstruction"
            ) from error

        target_ids = tuple(
            int(token_id)
            for token_id in self._target.encode(text, add_special_tokens=False).ids
        )
        if not target_ids:
            return records.ValidationResult(
                False, validated.cluster_id, None, "retokenized_document_empty"
            )
        invalid = next(
            (token_id for token_id in target_ids if not 0 <= token_id < TARGET_SOURCE_VOCAB_SIZE),
            None,
        )
        if invalid is not None:
            raise RuntimeError(
                f"ordinary source text produced reserved/out-of-range SuperBPE token {invalid}"
            )
        return records.ValidationResult(True, validated.cluster_id, target_ids, None)


@dataclass
class InstalledSuperBPERetokenization:
    retokenizer: SuperBPERetokenizer
    _original_validate_record: Callable[[records.ParsedRecord], records.ValidationResult]
    _original_eod: int
    _original_dynamic: dict[str, tuple[bool, object]]
    _active: bool = True

    @property
    def contract(self) -> dict[str, object]:
        return self.retokenizer.contract

    def restore(self) -> None:
        if not self._active:
            return
        streaming.validate_record = self._original_validate_record
        config.EOD_TOKEN_ID = self._original_eod
        for name, (existed, value) in self._original_dynamic.items():
            if existed:
                setattr(config, name, value)
            elif hasattr(config, name):
                delattr(config, name)
        self._active = False


def install_superbpe_retokenization(
    repo_root: Path | str,
) -> InstalledSuperBPERetokenization:
    """Install the accepted MoE corpus tokenizer contract for one producer process."""

    root = Path(repo_root).resolve()
    tokenizer_path = root / TARGET_TOKENIZER_RELATIVE_PATH
    retokenizer = SuperBPERetokenizer(tokenizer_path)
    original_dynamic = {
        name: (hasattr(config, name), getattr(config, name, None))
        for name in _DYNAMIC_CONFIG_FIELDS
    }
    installed = InstalledSuperBPERetokenization(
        retokenizer=retokenizer,
        _original_validate_record=streaming.validate_record,
        _original_eod=config.EOD_TOKEN_ID,
        _original_dynamic=original_dynamic,
    )
    streaming.validate_record = retokenizer.validate_record
    config.EOD_TOKEN_ID = TARGET_EOD_TOKEN_ID
    config.CORPUS_SOURCE_TOKENIZER_ID = SOURCE_TOKENIZER_ID
    config.CORPUS_OUTPUT_TOKENIZER_ID = TARGET_TOKENIZER_ID
    config.CORPUS_TOKENIZER_ARTIFACT = TARGET_TOKENIZER_RELATIVE_PATH
    config.CORPUS_TOKENIZER_SHA256 = retokenizer.tokenizer_sha256
    config.CORPUS_SEMANTIC_VOCAB_SIZE = TARGET_SEMANTIC_VOCAB_SIZE
    return installed


__all__ = [
    "EXPECTED_SPECIAL_TOKENS",
    "SOURCE_EOD_TOKEN_ID",
    "SOURCE_TOKENIZER_ID",
    "TARGET_EOD_TOKEN_ID",
    "TARGET_SEMANTIC_VOCAB_SIZE",
    "TARGET_SOURCE_VOCAB_SIZE",
    "TARGET_TOKENIZER_ID",
    "TARGET_TOKENIZER_RELATIVE_PATH",
    "InstalledSuperBPERetokenization",
    "SuperBPERetokenizer",
    "install_superbpe_retokenization",
]
