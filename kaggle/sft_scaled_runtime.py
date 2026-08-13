#!/usr/bin/env python3
"""Scaled SFT runtime extensions."""
from __future__ import annotations

import json
from pathlib import Path

import sft_runtime as base


def prepare(
    profile: base.SFTProfileSpec,
    *,
    replay_root: str,
    prepared_dir: str | None,
    output_dir: str | None,
    parent_consumed_tokens: int | None,
    revision: str | None,
) -> int:
    replay = base._resolve_replay_root(replay_root)
    worktree = base._prepare_worktree(profile)
    prepared = Path(prepared_dir).expanduser().resolve() if prepared_dir else profile.default_prepared
    output = Path(output_dir).expanduser().resolve() if output_dir else profile.default_bundle
    prepared_manifest_path = prepared / "prepared-manifest.json"
    revision_args = ["--revision", revision] if revision else []
    if prepared_manifest_path.is_file():
        if revision is not None:
            try:
                payload = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as error:
                raise base.RuntimeFailure("existing prepared SFT source manifest is invalid") from error
            if not isinstance(payload, dict) or payload.get("revision") != revision:
                raise base.RuntimeFailure("existing prepared SFT source uses a different pinned revision")
    else:
        base._run(
            base._uv_prefix(datasets=True)
            + ["python", "-m", "post_training.sft.bundle", "prepare", "--output-dir", str(prepared), *revision_args],
            cwd=worktree,
        )

    exact_parent_tokens = base._exact_parent_tokens(profile, parent_consumed_tokens)
    expected_targets = base._expected_sft_targets(profile, exact_parent_tokens)
    if not base._verify_existing_bundle_budget(output, expected_targets=expected_targets):
        if output.exists():
            raise base.RuntimeFailure(
                f"refusing to replace incomplete/non-bundle SFT output directory: {output}"
            )
        base._run(
            base._uv_prefix()
            + [
                "python", "-m", "post_training.sft.scaled_bundle",
                "--prepared-dir", str(prepared),
                "--replay-root", str(replay),
                "--output-dir", str(output),
                "--parent-consumed-tokens", str(exact_parent_tokens),
                "--fraction-numerator", str(profile.sft_fraction_numerator),
                "--fraction-denominator", str(profile.sft_fraction_denominator),
                "--optimizer-target-tokens", "32768",
                "--instruction-share", "0.85",
                "--replay-share", "0.15",
                "--seed", "17",
            ],
            cwd=worktree,
        )
    return base._run(
        base._uv_prefix() + ["python", "-m", "post_training.sft.bundle", "verify", "--dataset-dir", str(output)],
        cwd=worktree,
    )


def publish(profile: base.SFTProfileSpec, **kwargs) -> int:
    prepare(
        profile,
        replay_root=kwargs["replay_root"],
        prepared_dir=kwargs.get("prepared_dir"),
        output_dir=kwargs.get("output_dir"),
        parent_consumed_tokens=kwargs.get("parent_consumed_tokens"),
        revision=kwargs.get("revision"),
    )
    # base.publish re-verifies the now-existing 10% bundle and then publishes it.
    return base.publish(profile, **kwargs)


def evaluate(profile: base.SFTProfileSpec, **kwargs) -> int:
    return base.evaluate(profile, **kwargs)


__all__ = ["evaluate", "prepare", "publish"]
