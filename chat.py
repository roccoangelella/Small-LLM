TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95
MAX_NEW_TOKENS = 256
SEED = 17

import argparse
import codecs
import gc
import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


_SFT_RUN_IDS = {
    (20_000_000, 500_000_000): "20m-500m-sft-s0-001",
    (20_000_000, 2_000_000_000): "20m-2b-sft-s0-001",
}
_QUANTITY_SUFFIXES = {
    "": 1,
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}


class _TokenTextStreamer:
    """Incrementally decode GPT-2 token bytes without corrupting split UTF-8."""

    def __init__(self, encoding: object) -> None:
        self.encoding = encoding
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        token_bytes = self.encoding.decode_single_token_bytes(token_id)
        return self.decoder.decode(token_bytes, final=False)

    def finish(self) -> str:
        return self.decoder.decode(b"", final=True)


def _parse_quantity(value: str) -> int:
    compact = value.strip().replace("_", "").replace(",", "").replace(" ", "")
    if not compact:
        raise argparse.ArgumentTypeError("size cannot be empty")
    suffix = compact[-1].upper() if compact[-1].isalpha() else ""
    number = compact[:-1] if suffix else compact
    if suffix not in _QUANTITY_SUFFIXES:
        raise argparse.ArgumentTypeError(f"unsupported size suffix in {value!r}")
    try:
        amount = float(number) * _QUANTITY_SUFFIXES[suffix]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}") from error
    if not math.isfinite(amount) or amount <= 0 or not amount.is_integer():
        raise argparse.ArgumentTypeError(f"size must resolve to a positive whole number: {value!r}")
    return int(amount)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a completed Small-LLM SFT checkpoint from Hugging Face and chat locally."
    )
    parser.add_argument(
        "--model_params",
        "--model-params",
        dest="model_params",
        type=_parse_quantity,
        required=True,
        help="model parameter profile, e.g. 20M",
    )
    parser.add_argument(
        "--num_tokens",
        "--num-tokens",
        dest="num_tokens",
        type=_parse_quantity,
        required=True,
        help="parent pretraining token profile, e.g. 500M or 2B",
    )
    return parser.parse_args(argv)


def _resolve_sft_run_id(model_params: int, num_tokens: int) -> str:
    try:
        return _SFT_RUN_IDS[(model_params, num_tokens)]
    except KeyError as error:
        supported = ", ".join(
            f"{params // 1_000_000}M/{tokens // 1_000_000}M"
            if tokens < 1_000_000_000
            else f"{params // 1_000_000}M/{tokens // 1_000_000_000}B"
            for params, tokens in _SFT_RUN_IDS
        )
        raise RuntimeError(
            f"no registered SFT profile for model_params={model_params}, num_tokens={num_tokens}; "
            f"supported: {supported}"
        ) from error


def _repo_id() -> str:
    repo_id = os.environ.get("SMALL_LLM_SFT_HF_REPO_ID") or os.environ.get(
        "SMALL_LLM_HF_REPO_ID"
    )
    if not repo_id:
        raise RuntimeError(
            "set SMALL_LLM_SFT_HF_REPO_ID (or SMALL_LLM_HF_REPO_ID when SFT checkpoints "
            "share the base checkpoint repository)"
        )
    return repo_id


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _load_completed_sft(checkpoint_root: Path, *, device: object):
    """Load one verified SFT checkpoint and reject partial SFT trajectories."""

    from dataset.src.joint_checkpoint import verify_local_manifest
    from model.config import ModelConfig
    from model.model import SmallLLM

    verify_local_manifest(checkpoint_root)
    checkpoint = _read_json(checkpoint_root / "checkpoint.json", label="checkpoint.json")
    pipeline = checkpoint.get("pipeline_state")
    sft_identity = pipeline.get("sft_identity") if isinstance(pipeline, Mapping) else None
    if not isinstance(sft_identity, Mapping) or sft_identity.get("stage") != "sft_s0":
        raise RuntimeError("the selected Hugging Face checkpoint is not an SFT S0 checkpoint")

    with (checkpoint_root / "trainer_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, dict) or state.get("version") != 1:
        raise RuntimeError("SFT trainer_state.pkl has an unsupported structure or version")

    trainer_config = state.get("config")
    if not isinstance(trainer_config, Mapping) or trainer_config.get("schedule") != "wsd":
        raise RuntimeError("SFT checkpoint has no valid WSD trainer configuration")
    schedule_parts: list[int] = []
    for name in ("warmup_tokens", "stable_tokens", "decay_tokens"):
        value = trainer_config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"SFT checkpoint has invalid {name}")
        schedule_parts.append(value)
    expected_consumed = sum(schedule_parts)
    consumed = state.get("consumed_tokens")
    if isinstance(consumed, bool) or not isinstance(consumed, int) or consumed < 0:
        raise RuntimeError("SFT checkpoint has an invalid consumed_tokens counter")
    if expected_consumed <= 0 or consumed != expected_consumed:
        raise RuntimeError(
            "SFT exists on Hugging Face but is not complete: "
            f"consumed_loss_targets={consumed:,}, full_schedule_targets={expected_consumed:,}"
        )

    raw_config = state.get("model_config")
    model_state = state.get("model")
    if not isinstance(raw_config, Mapping) or not isinstance(model_state, Mapping):
        raise RuntimeError("SFT checkpoint does not contain self-describing model weights")
    config_values = dict(raw_config)
    if isinstance(config_values.get("layer_pattern"), list):
        config_values["layer_pattern"] = tuple(config_values["layer_pattern"])
    config = ModelConfig(**config_values)  # type: ignore[arg-type]

    # The optimizer/scaler/RNG payloads are irrelevant for inference. Drop them
    # before allocating the live model so local RAM peaks stay modest.
    for name in (
        "optimizer",
        "scheduler",
        "scaler",
        "python_rng_state",
        "torch_rng_state",
        "cuda_rng_states",
    ):
        state.pop(name, None)
    gc.collect()

    model = SmallLLM(config)
    model.load_state_dict(model_state, strict=True)
    del model_state
    del state
    gc.collect()
    model.to(device)
    model.eval()
    return model, config, consumed


def _download_model(*, repo_id: str, run_id: str, device: object):
    from trainer.post_pretraining_prompt_suite import download_verified_checkpoint

    token = os.environ.get("HF_TOKEN")
    temporary = tempfile.TemporaryDirectory(prefix="small-llm-chat-")
    try:
        checkpoint_root, info = download_verified_checkpoint(
            repo_id=repo_id,
            run_id=run_id,
            token=token,
            revision=None,
            pointer_name="latest",
            destination=Path(temporary.name),
        )
        model, config, consumed = _load_completed_sft(checkpoint_root, device=device)
    except Exception:
        temporary.cleanup()
        raise
    return model, config, consumed, info, temporary


def _fit_generation_prompt(history, *, template, encoding, max_prompt_tokens: int):
    """Keep the newest complete turns while reserving room for the next answer."""

    while True:
        try:
            prompt_ids = template.encode_generation_prompt(history, encoding)
        except ValueError as error:
            if "generation prompt exceeds model context" not in str(error):
                raise
            prompt_ids = None
        if prompt_ids is not None and len(prompt_ids) <= max_prompt_tokens:
            return history, prompt_ids
        if len(history) <= 1:
            raise RuntimeError(
                f"message is too long; the chat prompt must fit within {max_prompt_tokens} tokens "
                f"after reserving {MAX_NEW_TOKENS} tokens for the answer"
            )
        history = history[2:]


def _stream_sample_token_ids(
    model,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    max_seq_len: int,
    eos_token_id: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    precision: str,
    encoding: object,
) -> list[int]:
    """Sample with the canonical logits policy and print each decoded token immediately."""

    import torch

    from trainer.post_pretraining_prompt_suite import _autocast_context, _filter_logits

    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if max_new_tokens < 0 or max_seq_len <= 0:
        raise ValueError("generation lengths are invalid")
    if temperature < 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    device = next(model.parameters()).device
    output = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    generated: list[int] = []
    streamer = _TokenTextStreamer(encoding)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            with _autocast_context(device, precision):
                logits = model(output[:, -max_seq_len:])[:, -1, :].float()

            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                filtered = _filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
                probabilities = torch.softmax(filtered, dim=-1)
                if not torch.isfinite(probabilities).all() or bool(
                    (probabilities.sum(dim=-1) <= 0).any()
                ):
                    raise FloatingPointError("sampling produced an invalid probability distribution")
                next_token = torch.multinomial(probabilities, 1, generator=generator)

            token_id = int(next_token.item())
            if token_id == eos_token_id:
                break

            generated.append(token_id)
            text = streamer.push(token_id)
            if text:
                print(text, end="", flush=True)
            output = torch.cat((output, next_token), dim=1)

    tail = streamer.finish()
    if tail:
        print(tail, end="", flush=True)
    return generated


def _chat(model, config, *, device) -> None:
    import tiktoken

    from post_training.sft.schema import ChatMessage
    from post_training.sft.template import GPT2ChatTemplate

    encoding = tiktoken.get_encoding("gpt2")
    template = GPT2ChatTemplate(
        eos_token_id=50_256,
        maximum_context_tokens=config.max_seq_len,
        maximum_assistant_tokens=min(MAX_NEW_TOKENS, 512),
    )
    precision = "fp16" if getattr(device, "type", None) == "cuda" else "fp32"
    max_prompt_tokens = config.max_seq_len - MAX_NEW_TOKENS
    if max_prompt_tokens <= 0:
        raise RuntimeError("MAX_NEW_TOKENS must be smaller than the model context length")

    history: list[ChatMessage] = []
    turn = 0
    print("Type /clear to clear history, or /quit to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.lower() in {"/quit", "/exit"}:
            return
        if text.lower() == "/clear":
            history.clear()
            print("history cleared")
            continue

        candidate = [*history, ChatMessage(role="user", content=text)]
        candidate, prompt_ids = _fit_generation_prompt(
            candidate,
            template=template,
            encoding=encoding,
            max_prompt_tokens=max_prompt_tokens,
        )
        print("assistant> ", end="", flush=True)
        try:
            generated = _stream_sample_token_ids(
                model,
                prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                max_seq_len=config.max_seq_len,
                eos_token_id=50_256,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                seed=SEED + turn,
                precision=precision,
                encoding=encoding,
            )
        except BaseException:
            print()
            raise
        turn += 1
        response = encoding.decode(generated).strip()
        if not response:
            print("[ended turn without text]")
            continue
        print()
        history = [*candidate, ChatMessage(role="assistant", content=response)]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = _resolve_sft_run_id(args.model_params, args.num_tokens)
    repo_id = _repo_id()

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "chat.py requires the model and post-training extras; install with "
            "`pip install -e '.[model,post-training]'`"
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, config, consumed, info, temporary = _download_model(
            repo_id=repo_id,
            run_id=run_id,
            device=device,
        )
    except Exception as error:
        raise RuntimeError(
            f"could not load a completed SFT checkpoint for {run_id} from {repo_id}: {error}"
        ) from error

    try:
        print(
            f"Loaded {run_id} checkpoint={info.get('checkpoint_id')} "
            f"loss_targets={consumed:,} device={device}."
        )
        _chat(model, config, device=device)
    finally:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
