"""Fixed profile for the 100M/2B SFT experiment."""
from __future__ import annotations

from pathlib import Path

import sft_runtime

IMPLEMENTATION_COMMIT = "3e409b09dd8af24c004ad42c22edcce4e8c9d077"


class Profile(sft_runtime.SFTProfileSpec):
    @property
    def run_root(self) -> Path:
        return sft_runtime.WORK / "small-llm-100m-2b-sft"

    @property
    def default_bundle(self) -> Path:
        return sft_runtime.WORK / "small-llm-100m-2b-sft-bundle"


PROFILE = Profile(
    model_parameters=100_000_000,
    parent_training_tokens=2_000_000_000,
    model_label="100M",
    token_label="2B",
    token_key="2b",
    parent_run_id="100m-2b-data-001",
    sft_run_id="100m-2b-sft-s0-001",
    wandb_run_id="100m-2b-sft-s0-001",
    wandb_run_name="100M / 2B parent / SFT S0 / 4%",
    dataset_slug="small-llm-100m-2b-sft-s0-001",
    known_parent_consumed_tokens=2_001_000_448,
    launch_commit=IMPLEMENTATION_COMMIT,
    sft_fraction_numerator=4,
    sft_fraction_denominator=100,
    microbatch_size=2,
    cadence_steps=250,
    learning_rate=3e-5,
)


__all__ = ["IMPLEMENTATION_COMMIT", "PROFILE", "Profile"]