"""Fail-closed profile for wiring SFT to the completed 100M/10B parent."""
from __future__ import annotations

from pathlib import Path

import sft_runtime


class Profile(sft_runtime.SFTProfileSpec):
    """100M/10B parent identity with no scientific SFT recipe selected yet."""

    @property
    def recipe_ready(self) -> bool:
        return False

    @property
    def parent_pointer(self) -> str:
        return "latest"

    @property
    def parent_transport(self) -> str:
        return "hf_storage_bucket"

    @property
    def requested_sft_targets(self) -> None:
        # ADR 0138 deliberately leaves the SFT horizon undecided.
        return None

    @property
    def run_root(self) -> Path:
        return sft_runtime.WORK / "small-llm-100m-10b-sft"

    @property
    def default_bundle(self) -> Path:
        return sft_runtime.WORK / "small-llm-100m-10b-sft-bundle"


PROFILE = Profile(
    model_parameters=100_000_000,
    parent_training_tokens=10_000_000_000,
    model_label="100M",
    token_label="10B",
    token_key="10b",
    parent_run_id="100m-10b-deep-decay-from-step15500",
    # These identities are placeholders for operator discovery only. No action
    # may launch until a later ADR selects the actual 100M/10B SFT recipe.
    sft_run_id="100m-10b-sft-pending",
    wandb_run_id="100m-10b-sft-pending",
    wandb_run_name="100M / 10B parent / SFT recipe pending",
    dataset_slug="small-llm-100m-10b-sft-pending",
    known_parent_consumed_tokens=10_000_007_168,
    # Deliberately invalid as a git pin so any accidental bypass still fails
    # closed in _prepare_worktree.
    launch_commit="pending-recipe",
    # Zero is not a recipe. dry-run/profiles expose this profile as pending and
    # requested_sft_targets stays None.
    sft_fraction_numerator=0,
    sft_fraction_denominator=1,
    microbatch_size=2,
    cadence_steps=250,
    learning_rate=0.0,
)


__all__ = ["PROFILE", "Profile"]
