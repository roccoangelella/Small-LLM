from __future__ import annotations

from pathlib import Path

import torch

from trainer import post_pretraining_prompt_suite as suite


def test_load_model_accepts_streamed_torch_checkpoint(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "trainer_state.pkl"
    expected_model_state = {"weight": torch.tensor([1.0])}
    torch.save(
        {
            "version": 1,
            "model": expected_model_state,
            "model_config": {"sentinel": "config"},
            "global_step": 46_250,
            "consumed_tokens": 6_062_080_000,
        },
        checkpoint,
    )

    class FakeConfig:
        pass

    class FakeModel:
        def __init__(self, config) -> None:
            self.config = config
            self.loaded = None
            self.device = None
            self.is_eval = False

        def load_state_dict(self, state, *, strict: bool) -> None:
            assert strict is True
            self.loaded = state

        def to(self, device) -> "FakeModel":
            self.device = device
            return self

        def eval(self) -> "FakeModel":
            self.is_eval = True
            return self

    fake_config = FakeConfig()
    monkeypatch.setattr(suite, "_normalize_model_config", lambda raw: fake_config)
    monkeypatch.setattr(suite, "SmallLLM", FakeModel)

    model, config, state = suite._load_model(
        tmp_path,
        device=torch.device("cpu"),
        model_config_json=None,
    )

    assert config is fake_config
    assert model.loaded is not None
    assert torch.equal(model.loaded["weight"], expected_model_state["weight"])
    assert model.device == torch.device("cpu")
    assert model.is_eval is True
    assert state["global_step"] == 46_250
