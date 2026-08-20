#!/usr/bin/env python3
"""Resume only manually-kept Superior Variant-D adaptations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEEP_RESUME_SCHEMA = "small-llm-superior-keep-resume-v1"
KEEP_COMPLETE_SCHEMA = "small-llm-superior-reasoning-curated-complete-v1"
KEEP_SUBDIR = "keep-resume"


def _load_base() -> ModuleType:
    name = "small_llm_rsft_superior_adaptation_base"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / "adapt_superior_reasoning.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load adaptation base")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


base = _load_base()
superior = base.superior


def _keep_root(work_dir: Path | str) -> Path:
    return Path(work_dir).expanduser().resolve() / KEEP_SUBDIR


def _keep_paths(work_dir: Path | str) -> tuple[Path, Path]:
    r = _keep_root(work_dir)
    return r / "candidates.jsonl", r / "candidates.manifest.json"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        for row in rows:
            h.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    tmp.replace(path)


def _validate_curation(rows, curation):
    ids = [str(r["id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate frozen candidate IDs")
    d = base.read_manual_curation(curation)
    missing = [x for x in ids if x not in d]
    extra = sorted(set(d) - set(ids))
    if missing or extra:
        raise RuntimeError(
            f"manual curation coverage drifted: missing={len(missing)} extra={len(extra)}"
        )
    return d, Counter(x["decision"] for x in d.values())


def _accepted_original_records(
    work_dir, candidate_rows, batch_size=base.DEFAULT_BATCH_SIZE
):
    root = Path(work_dir).expanduser().resolve()
    records = {}
    identities = []
    for i, batch in base._candidate_batches(
        root / "candidates.jsonl", batch_size=batch_size
    ):
        p = base._batch_path(root, i)
        if not p.is_file():
            continue
        payload = base._validate_batch_file(p, candidates=batch, index=i)
        identities.append(f"{i}:{base._sha256_path(p)}")
        for raw in payload["records"]:
            rid = str(raw["id"])
            if rid in records:
                raise RuntimeError(f"duplicate original accepted id {rid}")
            records[rid] = dict(raw)
    return records, identities


def prepare_keep_resume(
    work_dir: Path | str,
    *,
    manual_curation_jsonl: Path | str,
    baseline_manifest: Path | str | None = None,
    batch_size: int = base.DEFAULT_BATCH_SIZE,
):
    if not 1 <= batch_size <= superior.SIMPLIFICATION_MAX_BATCH_SIZE:
        raise ValueError("invalid batch_size")
    root = Path(work_dir).expanduser().resolve()
    cm = base._validate_candidate_manifest(
        root,
        baseline_manifest=Path(baseline_manifest).expanduser().resolve()
        if baseline_manifest
        else None,
    )
    rows = list(base._read_jsonl(root / "candidates.jsonl"))
    decisions, counts = _validate_curation(rows, manual_curation_jsonl)
    old, identities = _accepted_original_records(root, rows, batch_size)
    old_keep = {rid for rid in old if decisions[rid]["decision"] == "keep"}
    pending = [
        dict(r)
        for r in rows
        if decisions[str(r["id"])]["decision"] == "keep"
        and str(r["id"]) not in old_keep
    ]
    expected = int(counts.get("keep", 0))
    if len(old_keep) + len(pending) != expected:
        raise RuntimeError("keep partition drifted")
    cp, mp = _keep_paths(root)
    cur = Path(manual_curation_jsonl).expanduser().resolve()
    identity_sha = hashlib.sha256(
        ("\n".join(identities) + ("\n" if identities else "")).encode()
    ).hexdigest()
    frozen = {
        "schema": KEEP_RESUME_SCHEMA,
        "prompt_sha256": base._sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT),
        "prompt_policy": "ADR-0103-variant-D-fidelity-first",
        "batch_size": batch_size,
        "candidate_manifest_sha256": base._sha256_path(
            root / "candidates.manifest.json"
        ),
        "candidates_jsonl_sha256": cm["candidates_jsonl"]["sha256"],
        "manual_curation_sha256": base._sha256_path(cur),
        "manual_curation_counts": dict(sorted(counts.items())),
        "original_accepted_batch_count": len(identities),
        "original_accepted_batch_identity_sha256": identity_sha,
        "original_accepted_records": len(old),
        "original_accepted_kept_records": len(old_keep),
        "pending_kept_records": len(pending),
    }
    if cp.is_file() or mp.is_file():
        if not (cp.is_file() and mp.is_file()):
            raise RuntimeError("incomplete keep-resume cache")
        m = base._read_json(mp, label="keep-resume manifest")
        actual = list(base._read_jsonl(cp))
        if m.get("candidates_jsonl", {}).get("sha256") != base._sha256_path(cp):
            raise RuntimeError("keep-resume hash drifted")
        if [str(r["id"]) for r in actual] != [str(r["id"]) for r in pending]:
            raise RuntimeError("keep-resume identity drifted")
        for k, v in frozen.items():
            if m.get(k) != v:
                raise RuntimeError(f"keep-resume manifest drifted at {k}")
        return {
            "resumed_complete": True,
            "pending_kept_records": len(actual),
            "original_accepted_kept_records": len(old_keep),
            "candidates_jsonl": str(cp),
            "manifest": str(mp),
        }
    _write_jsonl(cp, pending)
    m = {
        **frozen,
        "candidates_jsonl": {
            "path": cp.name,
            "sha256": base._sha256_path(cp),
            "byte_size": cp.stat().st_size,
            "records": len(pending),
        },
    }
    base._atomic_json(mp, m)
    return {
        "resumed_complete": False,
        "pending_kept_records": len(pending),
        "original_accepted_kept_records": len(old_keep),
        "candidates_jsonl": str(cp),
        "manifest": str(mp),
    }


def _validate_keep_manifest(work_dir):
    cp, mp = _keep_paths(work_dir)
    m = base._read_json(mp, label="keep-resume manifest")
    if m.get("schema") != KEEP_RESUME_SCHEMA:
        raise RuntimeError("wrong keep-resume schema")
    if m.get("prompt_sha256") != base._sha256_text(
        superior.SIMPLIFICATION_SYSTEM_PROMPT
    ):
        raise RuntimeError("keep-resume prompt drifted")
    if m.get("candidates_jsonl", {}).get("sha256") != base._sha256_path(cp):
        raise RuntimeError("keep-resume candidates drifted")
    return m


def adapt_keep_wave(
    work_dir: Path | str,
    *,
    first_batch: int,
    batch_count: int,
    workers: int = 4,
    max_attempts: int = 2,
    retry_delay_seconds: float = 3.0,
):
    if min(first_batch, batch_count, workers) <= 0:
        raise ValueError("wave arguments must be positive")
    kr = _keep_root(work_dir)
    m = _validate_keep_manifest(work_dir)
    bs = int(m["batch_size"])
    cp = kr / "candidates.jsonl"
    n = int(m["candidates_jsonl"]["records"])
    total = (n + bs - 1) // bs
    last = min(total, first_batch + batch_count - 1)
    if first_batch > total:
        raise ValueError("first_batch exceeds total")
    selected = [
        (i, b)
        for i, b in base._candidate_batches(cp, batch_size=bs)
        if first_batch <= i <= last
    ]
    usage = Counter()
    done = calls = resumed = 0
    fail = []
    with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as ex:
        fm = {
            ex.submit(
                base._process_adaptation_batch,
                kr,
                batch_index=i,
                batch=b,
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay_seconds,
            ): (i, b)
            for i, b in selected
        }
        for f in as_completed(fm):
            i, b = fm[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - provider/batch failures are persisted and retried
                try:
                    r = _process_split_batch(
                        kr,
                        batch_index=i,
                        batch=b,
                        max_attempts=max_attempts,
                        retry_delay_seconds=retry_delay_seconds,
                    )
                except Exception as split_error:  # noqa: BLE001 - split recovery must persist any provider/validation failure
                    fail.append(
                        {
                            "batch_index": i,
                            "error_type": type(split_error).__name__,
                            "error": str(split_error),
                            "initial_error": str(e),
                        }
                    )
                    continue
            done += int(r["records"])
            calls += int(r["api_calls"])
            resumed += int(bool(r["resumed"]))
            u = r.get("usage")
            base._usage_add(usage, u if isinstance(u, Mapping) else None)
            print(
                f"[superior-keep:wave] batch={i}/{total} records_done={done}/{sum(len(b) for _, b in selected)}",
                flush=True,
            )
    return {
        "first_batch": first_batch,
        "last_batch": last,
        "requested_batches": len(selected),
        "completed_batches": len(selected) - len(fail),
        "completed_records": done,
        "api_calls_this_wave": calls,
        "resumed_batches": resumed,
        "usage": dict(sorted(usage.items())),
        "failures": sorted(fail, key=lambda r: int(r["batch_index"])),
        "total_batches": total,
    }


def keep_status(work_dir: Path | str):
    m = _validate_keep_manifest(work_dir)
    kr = _keep_root(work_dir)
    bs = int(m["batch_size"])
    accepted_batches = accepted_records = 0
    for i, b in base._candidate_batches(kr / "candidates.jsonl", batch_size=bs):
        p = base._batch_path(kr, i)
        if not p.is_file():
            continue
        payload = base._validate_batch_file(p, candidates=b, index=i)
        accepted_batches += 1
        accepted_records += len(payload["records"])
    return {
        "original_accepted_kept_records": int(m["original_accepted_kept_records"]),
        "resume_candidate_records": int(m["pending_kept_records"]),
        "resume_accepted_batches": accepted_batches,
        "resume_accepted_records": accepted_records,
        "resume_pending_records": int(m["pending_kept_records"]) - accepted_records,
        "total_accepted_kept_records": int(m["original_accepted_kept_records"])
        + accepted_records,
        "expected_total_kept_records": int(m["manual_curation_counts"]["keep"]),
    }


def _iter_resume_records(work_dir) -> Iterator[dict[str, Any]]:
    m = _validate_keep_manifest(work_dir)
    kr = _keep_root(work_dir)
    bs = int(m["batch_size"])
    for i, b in base._candidate_batches(kr / "candidates.jsonl", batch_size=bs):
        p = base._batch_path(kr, i)
        if not p.is_file():
            continue
        for raw in base._validate_batch_file(p, candidates=b, index=i)["records"]:
            yield dict(raw)


SPLIT_PART_SCHEMA = "small-llm-superior-keep-split-part-v1"


def _split_part_path(root: Path, batch_index: int, label: str) -> Path:
    return root / "split-batches" / f"batch-{batch_index:05d}-{label}.json"


def _split_attempt_path(root: Path, batch_index: int, label: str, attempt: int) -> Path:
    return (
        root
        / "split-attempts"
        / f"batch-{batch_index:05d}-{label}-attempt-{attempt:02d}.json"
    )


def _process_split_part(
    root: Path,
    *,
    batch_index: int,
    label: str,
    batch,
    max_attempts: int,
    retry_delay_seconds: float,
):
    accepted = _split_part_path(root, batch_index, label)
    ids = [str(r["id"]) for r in batch]
    if accepted.is_file():
        p = base._read_json(accepted, label="split accepted part")
        if p.get("schema") != SPLIT_PART_SCHEMA or p.get("ids") != ids:
            raise RuntimeError("split part identity drifted")
        return p
    existing = sorted(
        (root / "split-attempts").glob(
            f"batch-{batch_index:05d}-{label}-attempt-*.json"
        )
    )
    highest = max([int(p.stem.rsplit("-", 1)[1]) for p in existing] or [0])
    last = None
    for run_attempt in range(1, max_attempts + 1):
        attempt = highest + run_attempt
        ap = _split_attempt_path(root, batch_index, label, attempt)
        response_text = None
        system, user = superior.build_simplification_messages(
            [
                {
                    "id": r["id"],
                    "skill": superior.SOURCE_SKILLS[superior.PRODUCTION_DOMAIN],
                    "problem": r["problem"],
                    "reasoning": r["reasoning"],
                    "answer": r["answer"],
                }
                for r in batch
            ]
        )
        if run_attempt > 1:
            system = {
                "role": system["role"],
                "content": system["content"] + base.STRICT_RECOVERY_SUFFIX,
            }
        try:
            client = base.transport.GeminiDistillationClient(
                timeout_seconds=base.ADAPTATION_REQUEST_TIMEOUT_SECONDS
            )
            resp = client.complete((system, user))
            response_text = resp.content
            parsed = superior.parse_simplification_response(
                response_text, expected_ids=ids
            )
            records = [base._validate_rewrite(c, r) for c, r in zip(batch, parsed)]
            usage = dict(resp.usage) if isinstance(resp.usage, Mapping) else None
            payload = {
                "schema": SPLIT_PART_SCHEMA,
                "batch_index": batch_index,
                "label": label,
                "ids": ids,
                "records": records,
                "prompt_sha256": base._sha256_text(
                    superior.SIMPLIFICATION_SYSTEM_PROMPT
                ),
                "model": resp.model,
                "finish_reason": resp.finish_reason,
                "usage": usage,
                "attempt": attempt,
            }
            base._atomic_json(
                ap,
                {
                    "schema": base.ATTEMPT_SCHEMA,
                    "batch_index": batch_index,
                    "attempt": attempt,
                    "ids": ids,
                    "status": "accepted",
                    "split_label": label,
                    "response_text": response_text,
                    "accepted_part": payload,
                },
            )
            base._atomic_json(accepted, payload)
            return payload
        except Exception as e:  # noqa: BLE001 - provider/parse/validation failures are audit records
            last = e
            rej = {
                "schema": base.ATTEMPT_SCHEMA,
                "batch_index": batch_index,
                "attempt": attempt,
                "ids": ids,
                "status": "rejected",
                "split_label": label,
                "error_type": type(e).__name__,
                "error": str(e),
            }
            if response_text is not None:
                rej["response_text"] = response_text
            base._atomic_json(ap, rej)
            if run_attempt < max_attempts and retry_delay_seconds:
                time.sleep(retry_delay_seconds)
    raise RuntimeError(
        f"split part {batch_index}/{label} failed after {max_attempts} attempts"
    ) from last


def _process_split_batch(
    root: Path,
    *,
    batch_index: int,
    batch,
    max_attempts: int,
    retry_delay_seconds: float,
):
    main = base._batch_path(root, batch_index)
    if main.is_file():
        p = base._validate_batch_file(main, candidates=batch, index=batch_index)
        return {
            "batch_index": batch_index,
            "records": len(batch),
            "api_calls": 0,
            "resumed": True,
            "usage": p.get("usage"),
        }
    pieces = (
        [("p1", batch[:2]), ("p2", batch[2:])] if len(batch) > 2 else [("p1", batch)]
    )
    payloads = []
    for label, part in pieces:
        try:
            payloads.append(
                _process_split_part(
                    root,
                    batch_index=batch_index,
                    label=label,
                    batch=part,
                    max_attempts=max_attempts,
                    retry_delay_seconds=retry_delay_seconds,
                )
            )
        except Exception:
            if len(part) == 1:
                raise
            for j, row in enumerate(part, 1):
                payloads.append(
                    _process_split_part(
                        root,
                        batch_index=batch_index,
                        label=f"{label}s{j}",
                        batch=[row],
                        max_attempts=max_attempts,
                        retry_delay_seconds=retry_delay_seconds,
                    )
                )
    by_id = {str(r["id"]): r for p in payloads for r in p["records"]}
    ids = [str(r["id"]) for r in batch]
    if set(by_id) != set(ids):
        raise RuntimeError("split recovery did not cover original batch")
    records = [by_id[rid] for rid in ids]
    usage = Counter()
    for p in payloads:
        base._usage_add(
            usage, p.get("usage") if isinstance(p.get("usage"), Mapping) else None
        )
    combined = {
        "schema": base.BATCH_SCHEMA,
        "batch_index": batch_index,
        "ids": ids,
        "records": records,
        "prompt_sha256": base._sha256_text(superior.SIMPLIFICATION_SYSTEM_PROMPT),
        "strict_recovery": True,
        "teacher_prompt_sha256": "split-recovery",
        "model": "gemini-variant-d-split-recovery",
        "finish_reason": "split-recovery",
        "usage": dict(sorted(usage.items())),
        "attempt": 0,
        "split_recovery": True,
    }
    base._atomic_json(main, combined)
    base._validate_batch_file(main, candidates=batch, index=batch_index)
    return {
        "batch_index": batch_index,
        "records": len(batch),
        "api_calls": 0,
        "resumed": False,
        "usage": dict(sorted(usage.items())),
        "split_recovery": True,
    }


def finalize_curated_complete_dataset(
    work_dir: Path | str,
    *,
    baseline_jsonl: Path | str,
    baseline_manifest: Path | str,
    manual_curation_jsonl: Path | str,
    output_jsonl: Path | str,
    seed: int = superior.DEFAULT_SEED,
):
    root = Path(work_dir).expanduser().resolve()
    cm = base._validate_candidate_manifest(
        root, baseline_manifest=Path(baseline_manifest).expanduser().resolve()
    )
    rows = list(base._read_jsonl(root / "candidates.jsonl"))
    decisions, counts = _validate_curation(rows, manual_curation_jsonl)
    resume_manifest = _validate_keep_manifest(root)
    old, identities = _accepted_original_records(
        root, rows, batch_size=int(resume_manifest["batch_size"])
    )
    new = list(_iter_resume_records(root))
    new_by = {str(r["id"]): r for r in new}
    expected_new = [
        str(r["id"]) for r in base._read_jsonl(_keep_root(root) / "candidates.jsonl")
    ]
    if len(new_by) != len(new) or set(new_by) != set(expected_new):
        raise RuntimeError(
            f"curated completion missing keep-resume rows: expected={len(expected_new)} accepted={len(new_by)}"
        )
    accepted = {
        rid: r for rid, r in old.items() if decisions[rid]["decision"] == "keep"
    }
    for rid, r in new_by.items():
        if rid in accepted:
            raise RuntimeError(f"keeper adapted twice: {rid}")
        if decisions[rid]["decision"] != "keep":
            raise RuntimeError(f"non-keep in resume: {rid}")
        accepted[rid] = r
    expected_keep = int(counts.get("keep", 0))
    if len(accepted) != expected_keep:
        raise RuntimeError(f"accepted keep rewrites {len(accepted)} != {expected_keep}")
    bp = Path(baseline_jsonl).expanduser().resolve()
    bm = base._validate_baseline_manifest(
        Path(baseline_manifest).expanduser().resolve()
    )
    if bm.get("output_sha256") != base._sha256_path(bp):
        raise RuntimeError("baseline hash drifted")
    fields = ("skill", "difficulty", "problem", "reasoning", "answer")
    combined = []
    norm = set()
    unchanged = gemini = 0
    for line, raw in enumerate(base._read_jsonl(bp), 1):
        if set(raw) != set(fields):
            raise RuntimeError(f"baseline row {line} wrong schema")
        rec = {f: str(raw[f]).strip() for f in fields}
        unchanged += int(
            rec["skill"] == superior.SOURCE_SKILLS[superior.PRODUCTION_DOMAIN]
        )
        gemini += int(
            rec["skill"] != superior.SOURCE_SKILLS[superior.PRODUCTION_DOMAIN]
        )
        h = superior._normalized_input_hash(rec["problem"])
        if h in norm:
            raise RuntimeError("duplicate baseline prompt")
        norm.add(h)
        combined.append(rec)
    dup = []
    adapted = 0
    for c in rows:
        rid = str(c["id"])
        raw = accepted.get(rid)
        if raw is None:
            continue
        rec = superior.to_rsft_mapping(raw)
        h = superior._normalized_input_hash(rec["problem"])
        if h in norm:
            dup.append(rid)
            continue
        norm.add(h)
        combined.append(rec)
        adapted += 1
    min_t = None
    max_t = 0
    for rec in combined:
        t = superior.atomic_rsft_serialized_tokens(
            problem=rec["problem"], reasoning=rec["reasoning"], answer=rec["answer"]
        )
        if t > superior.PRODUCTION_CONTEXT_LENGTH:
            raise RuntimeError(f"over-context final row: {t}")
        min_t = t if min_t is None else min(min_t, t)
        max_t = max(max_t, t)
    random.Random(seed).shuffle(combined)
    dest = Path(output_jsonl).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        for rec in combined:
            h.write(
                json.dumps(
                    {f: rec[f] for f in fields},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    tmp.replace(dest)
    identity_sha = hashlib.sha256(
        ("\n".join(identities) + ("\n" if identities else "")).encode()
    ).hexdigest()
    manifest = {
        "schema": KEEP_COMPLETE_SCHEMA,
        "policy": superior.PRODUCTION_FILTER_VERSION,
        "production_domain": superior.PRODUCTION_DOMAIN,
        "adaptation_policy": "ADR-0103-variant-D-fidelity-first",
        "context_length": superior.PRODUCTION_CONTEXT_LENGTH,
        "seed": seed,
        "candidate_rows": len(rows),
        "candidate_manifest_sha256": base._sha256_path(
            root / "candidates.manifest.json"
        ),
        "candidates_jsonl_sha256": cm["candidates_jsonl"]["sha256"],
        "manual_curation_counts": dict(sorted(counts.items())),
        "manual_curation_sha256": base._sha256_path(
            Path(manual_curation_jsonl).expanduser().resolve()
        ),
        "accepted_keep_rewrites": len(accepted),
        "original_accepted_batch_identity_sha256": identity_sha,
        "keep_resume_manifest_sha256": base._sha256_path(
            _keep_root(root) / "candidates.manifest.json"
        ),
        "duplicate_rewrite_exclusions": len(dup),
        "duplicate_rewrite_excluded_ids": dup,
        "unchanged_superior_rows": unchanged,
        "adapted_superior_rows": adapted,
        "clean_superior_instruction_rows": unchanged + adapted,
        "gemini_rows": gemini,
        "combined_rows": len(combined),
        "serialized_token_range": {"min": min_t, "max": max_t},
        "baseline_jsonl_sha256": base._sha256_path(bp),
        "output_jsonl": str(dest.relative_to(REPO))
        if dest.is_relative_to(REPO)
        else str(dest),
        "output_sha256": base._sha256_path(dest),
        "output_byte_size": dest.stat().st_size,
    }
    mp = dest.with_suffix(dest.suffix + ".manifest.json")
    base._atomic_json(mp, manifest)
    return {**manifest, "manifest": str(mp)}


def _positive_int(v: str) -> int:
    n = int(v)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def _nonnegative_float(v: str) -> float:
    n = float(v)
    if n < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    x = s.add_parser("prepare")
    x.add_argument("--work-dir", type=Path, required=True)
    x.add_argument("--manual-curation-jsonl", type=Path, required=True)
    x.add_argument("--baseline-manifest", type=Path)
    x.add_argument("--batch-size", type=_positive_int, default=base.DEFAULT_BATCH_SIZE)
    x = s.add_parser("adapt-wave")
    x.add_argument("--work-dir", type=Path, required=True)
    x.add_argument("--first-batch", type=_positive_int, required=True)
    x.add_argument("--batch-count", type=_positive_int, required=True)
    x.add_argument("--workers", type=_positive_int, default=4)
    x.add_argument("--max-attempts", type=_positive_int, default=2)
    x.add_argument("--retry-delay-seconds", type=_nonnegative_float, default=3.0)
    x = s.add_parser("status")
    x.add_argument("--work-dir", type=Path, required=True)
    x = s.add_parser("finalize")
    x.add_argument("--work-dir", type=Path, required=True)
    x.add_argument("--baseline-jsonl", type=Path, required=True)
    x.add_argument("--baseline-manifest", type=Path, required=True)
    x.add_argument("--manual-curation-jsonl", type=Path, required=True)
    x.add_argument("--output-jsonl", type=Path, required=True)
    x.add_argument("--seed", type=int, default=superior.DEFAULT_SEED)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.command == "prepare":
        r = prepare_keep_resume(
            a.work_dir,
            manual_curation_jsonl=a.manual_curation_jsonl,
            baseline_manifest=a.baseline_manifest,
            batch_size=a.batch_size,
        )
    elif a.command == "adapt-wave":
        r = adapt_keep_wave(
            a.work_dir,
            first_batch=a.first_batch,
            batch_count=a.batch_count,
            workers=a.workers,
            max_attempts=a.max_attempts,
            retry_delay_seconds=a.retry_delay_seconds,
        )
    elif a.command == "status":
        r = keep_status(a.work_dir)
    else:
        r = finalize_curated_complete_dataset(
            a.work_dir,
            baseline_jsonl=a.baseline_jsonl,
            baseline_manifest=a.baseline_manifest,
            manual_curation_jsonl=a.manual_curation_jsonl,
            output_jsonl=a.output_jsonl,
            seed=a.seed,
        )
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
