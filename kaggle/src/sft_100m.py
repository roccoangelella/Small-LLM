"""Fixed profile for the 100M/2B SFT experiment."""
from __future__ import annotations

from pathlib import Path

import sft_runtime

IMPLEMENTATION_COMMIT = "ca16b22905ebedc5925ab0abb9c40125254f1e1c"


class Profile(sft_runtime.SFTProfileSpec):
    @property
    def _canonical_four_percent(self) -> bool:
        return self.sft_fraction_numerator * 100 == 4 * self.sft_fraction_denominator

    @property
    def run_root(self) -> Path:
        if self._canonical_four_percent:
            return sft_runtime.WORK / "small-llm-100m-2b-sft"
        return sft_runtime.WORK / self.sft_run_id

    @property
    def default_bundle(self) -> Path:
        if self._canonical_four_percent:
            return sft_runtime.WORK / "small-llm-100m-2b-sft-bundle"
        return sft_runtime.WORK / f"{self.dataset_slug}-bundle"


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
