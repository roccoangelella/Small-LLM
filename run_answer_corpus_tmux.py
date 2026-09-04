#!/usr/bin/env python3
"""Run ADR-0136 all-model native evaluations and maintain evidence docs.

This file is intentionally self-contained so it can be copied into a Kaggle or
VPS workspace and launched inside tmux.  It runs the full eval_core_v1 suite for
Small-LLM pretrained and ordinary SFT checkpoints under two native-budget prompt
protocols:

* greedy_native:  temperature=0,   top_p=1, top_k=0, seed=17
* sampled_native: temperature=1.0, top_p=1, top_k=0, seed=17

The ADR-0025 greedy-32 qualitative cap is deliberately excluded: this runner
never passes --max-new-tokens 32.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MODEL_REPOS = (
    "roccoangelella/small-llm-20m-qualification",
    "roccoangelella/small-llm-100m-qualification",
)

DEFAULT_MODELS: tuple["ModelSpec", ...] = ()


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    label: str
    kind: str
    source: str
    repo_id: str
    run_id: str
    pointer: str = "latest"
    note: str = ""

    def key(self) -> tuple[str, str, str]:
        # Stable and live namespaces for the same repo/run/kind are the same
        # logical model candidate. Prefer the checked-in/default row that
        # appears first, and do not evaluate duplicate transports into the
        # same evidence row.
        return (self.repo_id, self.run_id, self.kind)


@dataclasses.dataclass(frozen=True)
class DecodeMode:
    name: str
    temperature: float
    top_p: float
    top_k: int
    seed: int = 17
    samples_per_prompt: int = 1


MODES = (
    DecodeMode("greedy_native", temperature=0.0, top_p=1.0, top_k=0),
    DecodeMode("sampled_native", temperature=1.0, top_p=1.0, top_k=0),
)

FROZEN_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "20m": {
        "semantic_vocab_size": 50257,
        "padded_vocab_size": 50304,
        "max_seq_len": 2048,
        "d_model": 256,
        "n_layers": 8,
        "d_ff": 704,
        "n_heads": 4,
        "head_dim": 64,
        "gdn_num_key_heads": 4,
        "gdn_num_value_heads": 4,
        "gdn_key_dim": 64,
        "gdn_value_dim": 64,
        "gdn_conv_kernel_size": 4,
        "gdn_chunk_size": 64,
        "layer_pattern": ["gdn", "gdn", "gdn", "mha"],
        "architecture": "gdn2_hybrid",
        "rms_norm_eps": 1e-6,
        "rope_base": 10000.0,
        "attention_window": None,
        "dropout": 0.0,
    },
    "100m": {
        "semantic_vocab_size": 50257,
        "padded_vocab_size": 50304,
        "max_seq_len": 2048,
        "d_model": 512,
        "n_layers": 20,
        "d_ff": 1408,
        "n_heads": 8,
        "head_dim": 64,
        "gdn_num_key_heads": 8,
        "gdn_num_value_heads": 8,
        "gdn_key_dim": 64,
        "gdn_value_dim": 64,
        "gdn_conv_kernel_size": 4,
        "gdn_chunk_size": 64,
        "layer_pattern": ["gdn", "gdn", "gdn", "mha"],
        "architecture": "gdn2_hybrid",
        "rms_norm_eps": 1e-6,
        "rope_base": 10000.0,
        "attention_window": None,
        "dropout": 0.0,
    },
}


def ensure_model_config_jsons(root: Path, work_dir: Path) -> dict[str, Path]:
    config_dir = (work_dir if work_dir.is_absolute() else root / work_dir) / "model_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for size, payload in FROZEN_MODEL_CONFIGS.items():
        target = config_dir / f"{size}_gdn2_hybrid.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths[size] = target.resolve()
    return paths


def config_key_for_model(model: ModelSpec) -> str | None:
    probe = f"{model.label} {model.run_id}".lower()
    if "20m" in probe:
        return "20m"
    if "100m" in probe:
        return "100m"
    return None

# The checked-in default matrix mirrors the current project docs/evidence.  HF
# discovery can add newly-published ordinary pretrained/SFT runs at execution
# time, but these rows preserve the canonical intended scope when Hub listing is
# unavailable.
DEFAULT_MODELS = (
    ModelSpec(
        label="20m-qualification-pretrain",
        kind="pretrained",
        source="stable",
        repo_id="roccoangelella/small-llm-20m-qualification",
        run_id="20m-qualification-dataset-001",
    ),
    ModelSpec(
        label="20m-100m-pretrain",
        kind="pretrained",
        source="stable",
        repo_id="roccoangelella/small-llm-20m-qualification",
        run_id="20m-100m-dataset-001",
    ),
    ModelSpec(
        label="20m-500m-pretrain",
        kind="pretrained",
        source="stable",
        repo_id="roccoangelella/small-llm-20m-qualification",
        run_id="20m-500m-dataset-001",
    ),
    ModelSpec(
        label="20m-2b-pretrain",
        kind="pretrained",
        source="live",
        repo_id="roccoangelella/small-llm-20m-qualification",
        run_id="20m-2b-dataset-001",
    ),
    ModelSpec(
        label="100m-2b-pretrain",
        kind="pretrained",
        source="stable",
        repo_id="roccoangelella/small-llm-100m-qualification",
        run_id="100m-2b-data-001",
    ),
    ModelSpec(
        label="100m-10b-pretrain",
        kind="pretrained",
        source="bucket",
        repo_id="roccoangelella/small-llm-100m-qualification-checkpoints",
        run_id="100m-10b-deep-decay-from-step15500",
        note="live HF Storage Bucket latest pointer",
    ),
    ModelSpec(
        label="20m-500m-sft-s0",
        kind="sft",
        source="stable",
        repo_id="roccoangelella/small-llm-20m-qualification",
        run_id="20m-500m-sft-s0-001",
    ),
    ModelSpec(
        label="100m-2b-sft-s0",
        kind="sft",
        source="live",
        repo_id="roccoangelella/small-llm-100m-qualification",
        run_id="100m-2b-sft-s0-001",
    ),
    ModelSpec(
        label="100m-2b-sft-s0-10pct",
        kind="sft",
        source="live",
        repo_id="roccoangelella/small-llm-100m-qualification",
        run_id="100m-2b-sft-s0-10pct-001",
    ),
    ModelSpec(
        label="100m-2b-sft-s0-10pct-longpeak",
        kind="sft",
        source="live",
        repo_id="roccoangelella/small-llm-100m-qualification",
        run_id="100m-2b-sft-s0-10pct-longpeak-001",
        note="historical experiment; not canonical default after ADR-0132",
    ),
    ModelSpec(
        label="100m-2b-sft-s0-10pct-peak3000",
        kind="sft",
        source="live",
        repo_id="roccoangelella/small-llm-100m-qualification",
        run_id="100m-2b-sft-s0-10pct-peak3000-001",
    ),
)

EXCLUDE_RUN_ID_PARTS = (
    "rsft",
    "r-sft",
    "probe",
    "smoke",
    "triton-cache",
    "dataset-bucket",
)


@dataclasses.dataclass()
class RunRecord:
    model: ModelSpec
    mode: DecodeMode
    state: str
    output_path: Path
    log_path: Path
    status_path: Path
    payload: dict[str, Any] | None = None
    status: dict[str, Any] | None = None


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "trainer" / "eval_entrypoint.py").is_file():
        return cwd
    here = Path(__file__).resolve().parent
    current = here
    while current != current.parent:
        if (current / "trainer" / "eval_entrypoint.py").is_file():
            return current
        current = current.parent
    return cwd


def today_slug() -> str:
    return _dt.date.today().isoformat()


def default_work_dir(root: Path) -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "all_model_eval_adr0136_native"
    return root / "artifacts" / "all_model_eval_adr0136_native"


def default_doc_path(root: Path) -> Path:
    return root / "llm_docs" / "evidence" / "scaling" / f"adr0136_all_model_native_eval_{today_slug()}.md"


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip(".-_")
    if not cleaned:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return cleaned.lower()


def classify_run_id(run_id: str) -> str | None:
    lowered = run_id.lower()
    if any(part in lowered for part in EXCLUDE_RUN_ID_PARTS):
        return None
    if "sft" in lowered:
        return "sft"
    if "data" in lowered or "pretrain" in lowered or "qualification" in lowered:
        return "pretrained"
    return None


def label_for(run_id: str, *, kind: str) -> str:
    lowered = run_id.lower()
    label = lowered
    for suffix in ("-dataset-001", "-data-001", "-001"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    if kind == "pretrained" and not label.endswith("pretrain"):
        label = f"{label}-pretrain"
    return slugify(label)


def discover_hf_models(repo_ids: Sequence[str], *, token: str | None) -> list[ModelSpec]:
    """Best-effort discovery of ordinary pretrained/SFT model and live runs."""

    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        print("[discover] huggingface_hub is unavailable; using checked-in model matrix", flush=True)
        return []

    discovered: list[ModelSpec] = []
    api = HfApi(token=token)
    for repo_id in repo_ids:
        print(f"[discover] listing {repo_id}", flush=True)
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
        except Exception as error:  # noqa: BLE001 - discovery is best effort
            print(f"[discover] could not list {repo_id}: {error}", flush=True)
            continue

        stable_runs: set[str] = set()
        live_runs: set[str] = set()
        for path in files:
            if not isinstance(path, str):
                continue
            stable = re.fullmatch(r"models/([^/]+)/artifact\.json", path)
            if stable is not None:
                stable_runs.add(stable.group(1))
                continue
            live = re.fullmatch(r"run/([^/]+)/latest\.json", path)
            if live is not None:
                live_runs.add(live.group(1))

        for run_id in sorted(stable_runs):
            kind = classify_run_id(run_id)
            if kind is None:
                continue
            discovered.append(
                ModelSpec(
                    label=label_for(run_id, kind=kind),
                    kind=kind,
                    source="stable",
                    repo_id=repo_id,
                    run_id=run_id,
                )
            )
        for run_id in sorted(live_runs):
            kind = classify_run_id(run_id)
            if kind is None:
                continue
            discovered.append(
                ModelSpec(
                    label=label_for(run_id, kind=kind),
                    kind=kind,
                    source="live",
                    repo_id=repo_id,
                    run_id=run_id,
                )
            )
    return discovered


def merge_models(defaults: Iterable[ModelSpec], discovered: Iterable[ModelSpec]) -> list[ModelSpec]:
    ordered: list[ModelSpec] = []
    seen: set[tuple[str, str, str]] = set()
    used_labels: set[str] = set()
    for model in list(defaults) + list(discovered):
        key = model.key()
        if key in seen:
            continue
        seen.add(key)
        candidate = model
        if candidate.label in used_labels:
            candidate = dataclasses.replace(
                candidate,
                label=slugify(f"{candidate.label}-{candidate.run_id}-{candidate.source}"),
                note=(candidate.note + "; " if candidate.note else "")
                + "label disambiguated from another transport/run namespace",
            )
        used_labels.add(candidate.label)
        ordered.append(candidate)
    return ordered


def entrypoint_for(source: str) -> str:
    if source == "stable":
        return "trainer.eval_entrypoint_model"
    if source == "live":
        return "trainer.eval_entrypoint"
    if source == "bucket":
        return "trainer.eval_entrypoint_bucket"
    raise ValueError(f"unsupported model source: {source}")


def command_for(
    *,
    python: str,
    suite: str,
    model: ModelSpec,
    mode: DecodeMode,
    output_path: Path,
    eval_dir: Path | None,
    device: str,
    precision: str,
    batch_size: int,
    bootstrap_samples: int,
    model_config_json: Path | None = None,
) -> list[str]:
    if mode.name.endswith("32"):
        raise RuntimeError("greedy-32 mode is intentionally excluded")
    cmd = [
        python,
        "-m",
        entrypoint_for(model.source),
        suite,
        "--repo-id",
        model.repo_id,
        "--run-id",
        model.run_id,
        "--pointer",
        model.pointer,
        "--temperature",
        format_float(mode.temperature),
        "--top-p",
        format_float(mode.top_p),
        "--top-k",
        str(mode.top_k),
        "--seed",
        str(mode.seed),
        "--samples-per-prompt",
        str(mode.samples_per_prompt),
        "--device",
        device,
        "--precision",
        precision,
        "--batch-size",
        str(batch_size),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--output-json",
        str(output_path),
    ]
    if model_config_json is not None:
        cmd.extend(("--model-config-json", str(model_config_json)))
    if eval_dir is not None:
        cmd.extend(("--eval-dir", str(eval_dir)))
    return cmd


def format_float(value: float) -> str:
    if value == int(value):
        return f"{value:.1f}" if value != 0 else "0"
    return str(value)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as error:  # noqa: BLE001
        return {"_json_error": str(error)}
    return payload if isinstance(payload, dict) else {"_json_error": "top-level JSON is not an object"}


def valid_result_for_mode(payload: Mapping[str, Any] | None, mode: DecodeMode) -> bool:
    if payload is None or "_json_error" in payload:
        return False
    sampling = payload.get("sampling")
    if not isinstance(sampling, Mapping):
        return False
    checks = {
        "temperature": mode.temperature,
        "top_p": mode.top_p,
        "top_k": mode.top_k,
        "seed": mode.seed,
        "samples_per_prompt": mode.samples_per_prompt,
    }
    for key, expected in checks.items():
        observed = sampling.get(key)
        if isinstance(expected, float):
            try:
                if abs(float(observed) - expected) > 1e-12:
                    return False
            except Exception:
                return False
        elif observed != expected:
            return False
    return isinstance(payload.get("result_sha256"), str)


def write_status(path: Path, status: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(status), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_streamed(cmd: Sequence[str], *, cwd: Path, env: Mapping[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 96 + "\n")
        log.write(_dt.datetime.now().isoformat(timespec="seconds") + "\n")
        log.write(" ".join(shlex.quote(part) for part in cmd) + "\n")
        log.flush()
        process = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def maybe_fetch(root: Path, *, strict: bool) -> None:
    if not (root / ".git").is_dir():
        print("[git] no .git directory found; skipping fetch", flush=True)
        return
    cmd = ["git", "fetch", "--all", "--prune"]
    print("[git] " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)
    if completed.returncode != 0:
        message = f"git fetch failed with code {completed.returncode}"
        if strict:
            raise RuntimeError(message)
        print(f"[git] warning: {message}; continuing with current checkout", flush=True)


def prompt_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("prompts", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    return []


def md_cell(value: object, *, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\", "\\\\")
    text = text.replace("\n", "\\n")
    text = text.replace("|", "\\|")
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def num_cell(value: object, digits: int = 6) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}g}"
    return md_cell(value)


def result_record(
    *,
    model: ModelSpec,
    mode: DecodeMode,
    work_dir: Path,
) -> RunRecord:
    base = work_dir / slugify(model.label) / mode.name
    output_path = base.with_suffix(".json")
    log_path = base.with_suffix(".log")
    status_path = base.with_suffix(".status.json")
    payload = read_json(output_path)
    status = read_json(status_path)
    state = "done" if valid_result_for_mode(payload, mode) else "pending"
    if state == "pending" and isinstance(status, Mapping):
        recorded = status.get("state")
        if recorded in {"running", "failed", "dry_run", "skipped"}:
            state = str(recorded)
    return RunRecord(
        model=model,
        mode=mode,
        state=state,
        output_path=output_path,
        log_path=log_path,
        status_path=status_path,
        payload=payload if isinstance(payload, dict) else None,
        status=status if isinstance(status, dict) else None,
    )


def all_records(models: Sequence[ModelSpec], work_dir: Path) -> list[RunRecord]:
    return [result_record(model=model, mode=mode, work_dir=work_dir) for model in models for mode in MODES]


def render_document(
    *,
    models: Sequence[ModelSpec],
    work_dir: Path,
    generated_by: str,
) -> str:
    records = all_records(models, work_dir)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    lines: list[str] = [
        "---",
        "status: evidence",
        f"date: {_dt.date.today().isoformat()}",
        "protocol: ADR-0136-sampled-native-and-greedy-native",
        "---",
        "",
        "# ADR 0136 all-model native-budget evaluation",
        "",
        f"This evidence file is generated by `{generated_by}` from JSON bundles under `{work_dir}`.",
        "",
        "## Scope and exclusions",
        "",
        "- Included: pretrained and ordinary SFT checkpoints selected from the Small-LLM Hugging Face model repositories and the 100M/10B checkpoint bucket.",
        "- Greedy-native mode: `temperature=0`, `top_p=1`, `top_k=0`, seed `17`, one sample, native prompt budgets.",
        "- Sampled-native mode: `temperature=1`, `top_p=1`, `top_k=0`, seed `17`, one sample, native prompt budgets.",
        "- Excluded: the separate ADR-0025 greedy-32 qualitative cap (`max_new_tokens=32`), R-SFT reasoning diagnostics, and non-canonical probe/smoke branches.",
        f"- Last rendered: `{now}`.",
        "",
        "## Summary",
        "",
        "| model | kind | source | mode | state | checkpoint | loss | ppl | bpb | prompts | result_sha256 |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for record in records:
        payload = record.payload or {}
        checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), Mapping) else {}
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
        prompts = prompt_rows(payload)
        sha = payload.get("result_sha256") if isinstance(payload.get("result_sha256"), str) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{md_cell(record.model.label)}`",
                    md_cell(record.model.kind),
                    md_cell(record.model.source),
                    f"`{record.mode.name}`",
                    md_cell(record.state),
                    f"`{md_cell(checkpoint.get('checkpoint_id'))}`",
                    num_cell(metrics.get("loss")),
                    num_cell(metrics.get("perplexity")),
                    num_cell(metrics.get("bits_per_byte")),
                    str(len(prompts)) if prompts else "",
                    f"`{md_cell(sha)}`" if sha else "``",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Shorthand prompts and continuations", ""])
    for model in models:
        model_records = [record for record in records if record.model == model]
        if not any(record.payload and valid_result_for_mode(record.payload, record.mode) for record in model_records):
            continue
        lines.extend(
            [
                f"### {model.label}",
                "",
                f"`run_id={md_cell(model.run_id)}`; `repo={md_cell(model.repo_id)}`; `kind={md_cell(model.kind)}`; `source={md_cell(model.source)}`.",
                "",
            ]
        )
        if model.note:
            lines.extend([f"Note: {md_cell(model.note)}", ""])
        for record in model_records:
            payload = record.payload or {}
            if not valid_result_for_mode(payload, record.mode):
                continue
            sampling = payload.get("sampling") if isinstance(payload.get("sampling"), Mapping) else {}
            lines.extend(
                [
                    f"#### {record.mode.name}",
                    "",
                    f"`temperature={num_cell(sampling.get('temperature'))}`; `top_p={num_cell(sampling.get('top_p'))}`; `top_k={md_cell(sampling.get('top_k'))}`; `seed={md_cell(sampling.get('seed'))}`; `samples_per_prompt={md_cell(sampling.get('samples_per_prompt'))}`; `max_new_tokens=native`.",
                    "",
                    "| prompt | shorthand prompt | shorthand continuation |",
                    "| --- | --- | --- |",
                ]
            )
            for row in prompt_rows(payload):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{md_cell(row.get('name'))}`",
                            md_cell(row.get("prompt"), limit=220),
                            md_cell(row.get("continuation"), limit=240),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    lines.extend(
        [
            "## Reproduction command",
            "",
            "```bash",
            f"tmux new -s small-llm-adr0136 '{sys.executable} {shlex.quote(generated_by)} --suite full'",
            "```",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_document(*, path: Path, models: Sequence[ModelSpec], work_dir: Path, generated_by: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_document(models=models, work_dir=work_dir, generated_by=generated_by),
        encoding="utf-8",
    )
    print(f"[doc] wrote {path}", flush=True)


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def selected_models(models: Sequence[ModelSpec], labels: Sequence[str] | None) -> list[ModelSpec]:
    if not labels:
        return list(models)
    wanted = {slugify(label) for label in labels}
    selected = [model for model in models if slugify(model.label) in wanted or slugify(model.run_id) in wanted]
    missing = sorted(wanted - {slugify(model.label) for model in selected} - {slugify(model.run_id) for model in selected})
    if missing:
        raise SystemExit(f"unknown --model label(s): {missing}")
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("fast", "full"), default="full")
    parser.add_argument("--work-dir", type=Path, default=default_work_dir(root))
    parser.add_argument("--doc", type=Path, default=default_doc_path(root))
    parser.add_argument("--eval-dir", type=Path, help="override eval_core_v1 directory; otherwise use trainer.eval_entrypoint discovery")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--model", action="append", help="run only this label or run_id; may be repeated")
    parser.set_defaults(discover_hf=True)
    parser.add_argument("--discover-hf", dest="discover_hf", action="store_true")
    parser.add_argument("--no-discover-hf", dest="discover_hf", action="store_false")
    parser.add_argument("--repo", action="append", dest="repos", help="additional HF model repo to discover; may be repeated")
    parser.add_argument("--skip-git-fetch", action="store_true")
    parser.add_argument("--strict-git-fetch", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun even when a valid output JSON already exists")
    parser.add_argument("--force-failed", action="store_true", help="rerun records whose status is failed")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print planned commands without running them")
    parser.add_argument("--skip-doc", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    env = environment()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples cannot be negative")

    if not args.skip_git_fetch:
        maybe_fetch(root, strict=bool(args.strict_git_fetch))

    repo_ids = list(DEFAULT_MODEL_REPOS)
    if args.repos:
        repo_ids.extend(args.repos)
    discovered = discover_hf_models(repo_ids, token=env.get("HF_TOKEN")) if args.discover_hf else []
    models = selected_models(merge_models(DEFAULT_MODELS, discovered), args.model)

    print("[matrix] models:", flush=True)
    for model in models:
        print(
            f"  - {model.label}: kind={model.kind} source={model.source} repo={model.repo_id} run={model.run_id}",
            flush=True,
        )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    model_config_paths = ensure_model_config_jsons(root, args.work_dir)
    if not args.skip_doc:
        write_document(path=args.doc, models=models, work_dir=args.work_dir, generated_by=Path(__file__).name)

    failures = 0
    for model in models:
        print(f"\n[model] {model.label}", flush=True)
        for mode in MODES:
            record = result_record(model=model, mode=mode, work_dir=args.work_dir)
            if record.state == "done" and not args.force:
                print(f"[skip] {model.label}/{mode.name}: existing valid JSON {record.output_path}", flush=True)
                continue
            if record.state == "failed" and not (args.force or args.force_failed):
                print(f"[skip] {model.label}/{mode.name}: previous failure recorded at {record.status_path}", flush=True)
                continue

            cmd = command_for(
                python=args.python,
                suite=args.suite,
                model=model,
                mode=mode,
                output_path=record.output_path,
                eval_dir=args.eval_dir,
                device=args.device,
                precision=args.precision,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap_samples,
                model_config_json=model_config_paths.get(config_key_for_model(model)),
            )
            status_base: dict[str, Any] = {
                "state": "dry_run" if args.dry_run else "running",
                "model": dataclasses.asdict(model),
                "mode": dataclasses.asdict(mode),
                "command": cmd,
                "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "output_path": str(record.output_path),
                "log_path": str(record.log_path),
                "excludes_greedy_32": True,
            }
            write_status(record.status_path, status_base)
            if args.dry_run:
                print("[dry-run] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
                continue

            started = time.monotonic()
            returncode = run_streamed(cmd, cwd=root, env=env, log_path=record.log_path)
            elapsed = time.monotonic() - started
            payload = read_json(record.output_path)
            success = returncode == 0 and valid_result_for_mode(payload, mode)
            final_status: dict[str, Any] = {
                **status_base,
                "state": "done" if success else "failed",
                "returncode": returncode,
                "elapsed_seconds": round(elapsed, 3),
                "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "output_json_valid": bool(valid_result_for_mode(payload, mode)),
            }
            if isinstance(payload, Mapping) and isinstance(payload.get("_json_error"), str):
                final_status["json_error"] = payload["_json_error"]
            write_status(record.status_path, final_status)
            if not success:
                failures += 1
                print(f"[failed] {model.label}/{mode.name}; see {record.log_path}", flush=True)
                if args.stop_on_error:
                    if not args.skip_doc:
                        write_document(path=args.doc, models=models, work_dir=args.work_dir, generated_by=Path(__file__).name)
                    return 1
            else:
                print(f"[done] {model.label}/{mode.name} -> {record.output_path}", flush=True)
                if not args.skip_doc:
                    write_document(path=args.doc, models=models, work_dir=args.work_dir, generated_by=Path(__file__).name)

        if not args.skip_doc:
            write_document(path=args.doc, models=models, work_dir=args.work_dir, generated_by=Path(__file__).name)

    if args.dry_run:
        print("[dry-run] no evaluation commands were executed", flush=True)
    elif failures:
        print(f"[complete] finished with {failures} failed run(s)", flush=True)
        return 1
    else:
        print("[complete] all requested evaluations are done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
