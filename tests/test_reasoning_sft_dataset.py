from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "post_training" / "R-SFT" / "dataset.py"
SPEC = importlib.util.spec_from_file_location("reasoning_sft_dataset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dataset
SPEC.loader.exec_module(dataset)

TEST_ENDPOINT = "https://teacher.invalid/v1/chat/completions"


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class ReasoningSFTGeminiTransportTests(unittest.TestCase):
    def test_complete_sends_only_model_and_messages_and_parses_output(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, *, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(
                {
                    "id": "chatcmpl-test",
                    "model": "gemini-3.7-flash",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Teacher output"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            )

        client = dataset.GeminiDistillationClient(
            api_key="test-key",
            endpoint=TEST_ENDPOINT,
        )
        with mock.patch.object(dataset, "urlopen", side_effect=fake_urlopen):
            response = client.complete(
                [
                    {"role": "system", "content": "Return one example."},
                    dataset.ChatMessage(role="user", content="Hello!"),
                ]
            )

        request = captured["request"]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            sent,
            {
                "model": "gemini-3.7-flash",
                "messages": [
                    {"role": "system", "content": "Return one example."},
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )
        self.assertEqual(request.full_url, TEST_ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(captured["timeout"], dataset.DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(response.content, "Teacher output")
        self.assertEqual(response.model, "gemini-3.7-flash")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage["completion_tokens"], 2)

    def test_api_key_and_endpoint_can_fall_back_to_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                'GEMR_API_KEY="dotenv-secret"\n'
                'LLM_ENDPOINT="https://private-router.example/v1/chat/completions"\n',
                encoding="utf-8",
            )
            with mock.patch.dict(dataset.os.environ, {}, clear=True):
                self.assertEqual(
                    dataset.resolve_api_key(env_path=env_path),
                    "dotenv-secret",
                )
                self.assertEqual(
                    dataset.resolve_endpoint(env_path=env_path),
                    "https://private-router.example/v1/chat/completions",
                )

    def test_environment_takes_precedence_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "GEMR_API_KEY=dotenv-key\nLLM_ENDPOINT=https://dotenv.invalid/v1\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                dataset.os.environ,
                {
                    "GEMR_API_KEY": "environment-key",
                    "LLM_ENDPOINT": "https://environment.invalid/v1",
                },
                clear=True,
            ):
                self.assertEqual(dataset.resolve_api_key(env_path=env_path), "environment-key")
                self.assertEqual(
                    dataset.resolve_endpoint(env_path=env_path),
                    "https://environment.invalid/v1",
                )

    def test_missing_api_key_fails_without_network_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            with mock.patch.dict(dataset.os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "GEMR_API_KEY is required"):
                    dataset.GeminiDistillationClient(
                        endpoint=TEST_ENDPOINT,
                        env_path=env_path,
                    )

    def test_missing_endpoint_fails_without_network_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            with mock.patch.dict(dataset.os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "LLM_ENDPOINT is required"):
                    dataset.GeminiDistillationClient(
                        api_key="test-key",
                        env_path=env_path,
                    )

    def test_malformed_response_fails_closed(self) -> None:
        client = dataset.GeminiDistillationClient(
            api_key="test-key",
            endpoint=TEST_ENDPOINT,
        )
        with mock.patch.object(
            dataset,
            "urlopen",
            return_value=_FakeResponse({"choices": []}),
        ):
            with self.assertRaisesRegex(RuntimeError, "no choices"):
                client.complete_text("Hello!")


if __name__ == "__main__":
    unittest.main()
