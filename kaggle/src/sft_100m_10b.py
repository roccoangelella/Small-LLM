"""Fixed same-SFT-data profile for the completed 100M/10B parent."""
from __future__ import annotations

from pathlib import Path

import sft_runtime

IMPLEMENTATION_COMMIT = "704d819309588a1dc7cf24b75981cb22dd99b3fe"
PARENT_TARGETS = 10_000_007_168
SFT_TRAIN_TARGETS = 200_100_044
SHARED_DATASET_SLUG = "small-llm-100m-2b-sft-s0-10pct-001"


class Profile(sft_runtime.SFTProfileSpec):
    """100M/10B SFT profile reusing the 100M/2B 10% S0 corpus exactly."""

    @property
    def recipe_ready(self) -> bool:
        return True

    @property
    def allow_sft_fraction_override(self) -> bool:
        return False

    @property
    def parent_pointer(self) -> str:
        return "latest"

    @property
    def parent_transport(self) -> str:
        return "hf_storage_bucket"

    @property
    def requested_sft_targets(self) -> int:
        return SFT_TRAIN_TARGETS

    @property
    def run_root(self) -> Path:
        return sft_runtime.WORK / "small-llm-100m-10b-sft-same-2b10pct-data"

    @property
    def default_bundle(self) -> Path:
        return sft_runtime.WORK / f"{self.dataset_slug}-bundle"


PROFILE = Profile(
    model_parameters=100_000_000,
    parent_training_tokens=10_000_000_000,
    model_label="100M",
    token_label="10B",
    token_key="10b",
    parent_run_id="100m-10b-deep-decay-from-step15500",
    sft_run_id="100m-10b-sft-s0-2b10pct-data-001",
    wandb_run_id="100m-10b-sft-s0-2b10pct-data-001",
    wandb_run_name="100M / 10B parent / SFT S0 / 2B-10pct data / peak-through-3000",
    dataset_slug=SHARED_DATASET_SLUG,
    known_parent_consumed_tokens=PARENT_TARGETS,
    launch_commit=IMPLEMENTATION_COMMIT,
    # Express the absolute 100M/2B 10% SFT horizon as a fraction of the exact
    # 100M/10B final parent counter. This keeps legacy trainer budget plumbing
    # exact while making the scientific comparison an equal-SFT-token run.
    sft_fraction_numerator=SFT_TRAIN_TARGETS,
    sft_fraction_denominator=PARENT_TARGETS,
    microbatch_size=2,
    cadence_steps=250,
    learning_rate=3e-5,
)


__all__ = [
    "IMPLEMENTATION_COMMIT",
    "PARENT_TARGETS",
    "PROFILE",
    "Profile",
    "SFT_TRAIN_TARGETS",
    "SHARED_DATASET_SLUG",
]
