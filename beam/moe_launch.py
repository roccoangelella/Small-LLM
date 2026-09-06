#!/usr/bin/env python3
"""Dedicated Beam-side launcher for the frozen 8-expert Top-1 MoE pretraining experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from MOE_model.config import MoEModelConfig
from MOE_model.setup import setup as moe_setup
from MOE_model.setup import validation_reader as moe_validation_reader


def _contract() -> dict[str, object]:
    config = MoEModelConfig.substantive(architecture="gdn2_hybrid")
    return {
        "experiment": "small-llm-moe-8e-top1-v1",
        "model": config.as_dict(),
        "training_origin": "scratch",
        "notes": {
            "experts_per_layer": 8,
            "moe_layers": config.n_layers,
            "independent_expert_ffns": config.n_layers * config.num_experts,
            "load_balancing": "none; observe natural token-driven routing",
            "dropped_tokens": 0,
            "router_optimizer": "AdamW at base LR",
            "expert_optimizer": "Muon",
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--contract"]:
        print(json.dumps(_contract(), indent=2, sort_keys=True))
        return 0

    import trainer.cli as trainer_cli

    trainer_cli.setup = moe_setup
    trainer_cli.validation_reader = moe_validation_reader
    return trainer_cli.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
