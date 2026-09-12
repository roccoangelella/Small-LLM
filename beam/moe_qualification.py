"""Production qualification of the accepted MoE identity on Beam RTX 4090 and RTX 5090.

Same protocol as the Modal qualification of 2026-09-12 (evidence record
`llm_docs/evidence/moe_production_qualification_modal_2026-09-12.md`): the real production
CLI on the staged SuperBPE-8000 corpus with checkpoints, a resume compared byte for byte,
an fp16 arm, then engine-level throughput arms with and without the compile lane.
Run from the repository root: `python beam/moe_qualification.py --gpus RTX4090,RTX5090`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def _load_base():
    source = Path(__file__).resolve().with_name("launch.py")
    spec = importlib.util.spec_from_file_location("small_llm_beam_base_launch", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_base = _load_base()
DATASET = _base.DATA_ROOT / "qualification-20260912"
EVIDENCE = _base.RUN_ROOT / "qualification-20260912"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_diff(first: Path, second: Path) -> dict[str, object]:
    import pickle
    import torch

    def flat(o, prefix=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from flat(v, f"{prefix}/{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                yield from flat(v, f"{prefix}[{i}]")
        elif isinstance(o, torch.Tensor):
            yield prefix, o

    ta = dict(flat(pickle.load(first.open("rb"))))
    tb = dict(flat(pickle.load(second.open("rb"))))
    diffs = {}
    for k in ta:
        if k in tb and ta[k].shape == tb[k].shape and ta[k].is_floating_point():
            d = (ta[k].float() - tb[k].float()).abs().max().item()
            if d > 0:
                diffs[k] = d
    return {"tensors_compared": len(ta), "tensors_differing": len(diffs),
            "max_abs_diff": max(diffs.values()) if diffs else 0.0,
            "worst": sorted(diffs.items(), key=lambda kv: -kv[1])[:6]}


def _run(gpu_label: str, budget_seconds: int, microbatches: list[int], cli_microbatch: int,
         compile_variants: list[str], source_commit: str) -> dict[str, object]:
    repo = _base._install_beam_imports()
    started = time.perf_counter()
    root = Path("/tmp/qualification-evidence")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    state = {"gpu_label": gpu_label, "status": "running", "phases": {}, "source_commit": source_commit}
    import torch

    state["gpu"] = torch.cuda.get_device_name(0)
    state["torch"] = str(torch.__version__)
    env = {**os.environ, **_base.RUNTIME_ENV, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1",
           "PYTHONPATH": str(repo)}
    shard = DATASET / "train" / "train-000000.bin"
    assert shard.is_file(), "staged corpus missing"
    probe = repo / "beam" / "moe_qualification_probe.py"

    def remaining() -> float:
        return budget_seconds - (time.perf_counter() - started)

    def sh(name: str, argv: list[str], cap: int):
        if remaining() < cap + 45:
            state["phases"][name] = {"skipped": "budget", "remaining": remaining()}
            print("skip", name, round(remaining()), flush=True)
            return None
        t = time.perf_counter()
        try:
            with (root / (name + ".log")).open("w") as f:
                r = subprocess.run(argv, env=env, cwd=str(repo), stdout=f,
                                   stderr=subprocess.STDOUT, timeout=cap)
            code: object = r.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        state["phases"][name] = {"seconds": time.perf_counter() - t, "returncode": code}
        print(name, code, round(time.perf_counter() - t, 1), flush=True)
        return code

    def cli(run_dir: Path, steps: int, precision: str, extra: list[str]) -> list[str]:
        return [sys.executable, "-m", "MOE_model", "--dataset-dir", str(DATASET),
                "--checkpoint-dir", str(run_dir / "checkpoints"),
                "--experiment-dir", str(run_dir / "artifacts"), "--steps", str(steps),
                "--model-size", "accepted", "--architecture", "gdn2_hybrid",
                "--gdn-chunk-size", "32", "--load-balancing", "quantile",
                "--balancing-step-size", "0", "--initialization", "normal",
                "--optimizer", "hybrid_muon_adamw", "--precision", precision,
                "--microbatch-size", str(cli_microbatch), "--source-commit", source_commit,
                "--validation-blocks", "1", "--seed", "17", *extra]

    try:
        a = Path("/tmp/q/a")
        shutil.rmtree("/tmp/q", ignore_errors=True)
        a.mkdir(parents=True)
        code = sh("cli_bf16_continuous", cli(a, 6, "bf16", ["--checkpoint-every-steps", "3"]), 480)
        ck_a3, ck_a6 = a / "checkpoints/step-00000003", a / "checkpoints/step-00000006"
        if code == 0 and ck_a3.is_dir() and ck_a6.is_dir():
            b = Path("/tmp/q/b")
            (b / "checkpoints").mkdir(parents=True)
            shutil.copytree(ck_a3, b / "checkpoints/step-00000003")
            code = sh("cli_bf16_resume", cli(b, 3, "bf16", ["--resume", "step-00000003",
                                                            "--checkpoint-every-steps", "3"]), 300)
            ck_b6 = b / "checkpoints/step-00000006"
            if code == 0 and ck_b6.is_dir():
                ha, hb = _sha(ck_a6 / "trainer_state.pkl"), _sha(ck_b6 / "trainer_state.pkl")
                cmp: dict[str, object] = {"continuous_sha256": ha, "resumed_sha256": hb, "identical": ha == hb}
                if ha != hb:
                    cmp.update(_tensor_diff(ck_a6 / "trainer_state.pkl", ck_b6 / "trainer_state.pkl"))
                state["resume_comparison"] = cmp
                print("resume comparison", json.dumps({k: v for k, v in cmp.items() if k != "worst"}), flush=True)
            state["checkpoint_bytes"] = sum(f.stat().st_size for f in ck_a6.iterdir())
            for sub in ("a", "b"):
                for p in (Path("/tmp/q") / sub / "artifacts").rglob("*"):
                    if p.is_file() and p.stat().st_size < 4_000_000 and p.suffix in {".json", ".jsonl", ".txt"}:
                        (root / f"{sub}_{p.name}").write_bytes(p.read_bytes())
        c = Path("/tmp/q/c")
        c.mkdir(parents=True)
        sh("cli_fp16_short", cli(c, 3, "fp16", ["--checkpoint-every-steps", "0"]), 300)
        shutil.rmtree("/tmp/q", ignore_errors=True)

        for mb in microbatches:
            sh(f"tp_bf16_mb{mb}", [sys.executable, str(probe), str(shard),
                                   str(root / f"tp_bf16_mb{mb}.json"), "8", str(mb), "bf16", "off"], 300)
        for variant in compile_variants:
            tag = variant.replace("+", "_")
            for mb in microbatches:
                sh(f"tp_{tag}_mb{mb}", [sys.executable, str(probe), str(shard),
                                        str(root / f"tp_{tag}_mb{mb}.json"), "12", str(mb), "bf16", variant], 480)
        state["status"] = "pass"
    except BaseException as error:
        state.update(status="fail", error=repr(error))
        raise
    finally:
        state["elapsed"] = time.perf_counter() - started
        (root / "run.json").write_text(json.dumps(state, indent=2, default=str))
        try:
            out = EVIDENCE / gpu_label
            out.mkdir(parents=True, exist_ok=True)
            for p in root.iterdir():
                if p.is_file():
                    shutil.copy2(p, out / p.name)
        except OSError as error:
            state["evidence_copy_error"] = repr(error)
    return {"state": state,
            "files": {p.name: p.read_text() for p in root.iterdir()
                      if p.suffix in {".json", ".log", ".jsonl", ".txt"} and p.stat().st_size < 4_000_000}}


_KW = {
    "cpu": 4,
    "memory": "32Gi",
    "timeout": 2400,
    "retries": 0,
    "volumes": [_base.DATA_VOLUME, _base.RUN_VOLUME, _base.CACHE_VOLUME],
    "env": _base.RUNTIME_ENV,
}


@_base.function(name="test-qualifica-rtx4090", gpu="RTX4090", image=_base.LEGACY_SERVERLESS_IMAGE, **_KW)
def qualify_rtx4090(budget: int, source_commit: str) -> dict[str, object]:
    return _run("RTX4090", budget, [16], 16, ["blocks", "blocks+reduce-overhead"], source_commit)


@_base.function(name="test-qualifica-rtx5090", gpu="RTX5090", image=_base.BLACKWELL_IMAGE, **_KW)
def qualify_rtx5090(budget: int, source_commit: str) -> dict[str, object]:
    return _run("RTX5090", budget, [16, 32], 16, ["blocks"], source_commit)


FUNCTIONS = {"RTX4090": qualify_rtx4090, "RTX5090": qualify_rtx5090}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="RTX4090,RTX5090")
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    source_commit = _base._local_source_commit()
    out = Path(args.out) if args.out else Path.cwd() / "beam-qualification"
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    def worker(gpu: str) -> None:
        try:
            results[gpu] = FUNCTIONS[gpu].remote(args.budget, source_commit)
        except Exception as error:  # noqa: BLE001
            results[gpu] = {"error": repr(error)}

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary = {}
    for gpu, result in results.items():
        if not isinstance(result, dict) or "state" not in result:
            summary[gpu] = result
            continue
        folder = out / gpu
        folder.mkdir(exist_ok=True)
        for name, content in result["files"].items():
            (folder / name).write_text(content)
        (folder / "returned-state.json").write_text(json.dumps(result["state"], indent=2, default=str))
        summary[gpu] = {k: v.get("returncode", v.get("skipped")) for k, v in result["state"]["phases"].items()}
        summary[gpu]["resume"] = result["state"].get("resume_comparison", {}).get("identical")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
