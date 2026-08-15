from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


RSFT_DIR = Path(__file__).resolve().parents[1] / "post_training" / "R-SFT"


def _load_tokenizer():
    module_name = "small_llm_rsft_tokenizer_test"
    path = RSFT_DIR / "tokenizer.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tokenizer = _load_tokenizer()


class _ByteEncoding:
    """Tiny injectable stand-in for GPT-2 used to keep this test dependency-free."""

    def encode(self, text: str, **kwargs) -> list[int]:
        del kwargs
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8")

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return bytes([token_id])


class ReasoningTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = tokenizer.ReasoningTokenSpec(
            reasoning_start="<R>",
            reasoning_end="</R>",
            answer_start="<A>",
        )

    def test_metadata_round_trip_preserves_strings_and_fixed_ids(self) -> None:
        metadata = self.spec.to_metadata()
        restored = tokenizer.ReasoningTokenSpec.from_metadata(metadata)
        self.assertEqual(restored, self.spec)
        self.assertEqual(
            restored.special_tokens,
            {
                "<R>": 50_257,
                "</R>": 50_258,
                "<A>": 50_259,
            },
        )

    def test_pipeline_state_is_required_and_fail_closed(self) -> None:
        pipeline = {tokenizer.TOKENIZER_METADATA_KEY: self.spec.to_metadata()}
        self.assertEqual(tokenizer.spec_from_pipeline_state(pipeline), self.spec)
        with self.assertRaisesRegex(ValueError, "missing"):
            tokenizer.spec_from_pipeline_state({})

    def test_wrong_promoted_id_is_rejected(self) -> None:
        metadata = self.spec.to_metadata()
        metadata["special_tokens"]["reasoning_start"]["id"] = 50_260
        with self.assertRaisesRegex(ValueError, "fixed token ID 50257"):
            tokenizer.ReasoningTokenSpec.from_metadata(metadata)

    def test_encode_decode_treats_control_markers_as_single_tokens(self) -> None:
        encoding = tokenizer.ReasoningGPT2Encoder(
            self.spec,
            base_encoding=_ByteEncoding(),
        )
        text = "Q<R>think</R><A>answer"
        token_ids = encoding.encode(text)
        self.assertIn(50_257, token_ids)
        self.assertIn(50_258, token_ids)
        self.assertIn(50_259, token_ids)
        self.assertEqual(token_ids.count(50_257), 1)
        self.assertEqual(token_ids.count(50_258), 1)
        self.assertEqual(token_ids.count(50_259), 1)
        self.assertEqual(encoding.decode(token_ids), text)
        self.assertEqual(encoding.decode_single_token_bytes(50_257), b"<R>")

    def test_non_rsft_padding_id_is_rejected_on_decode(self) -> None:
        encoding = tokenizer.ReasoningGPT2Encoder(
            self.spec,
            base_encoding=_ByteEncoding(),
        )
        with self.assertRaisesRegex(ValueError, "outside the R-SFT vocabulary"):
            encoding.decode([50_260])


if __name__ == "__main__":
    unittest.main()
