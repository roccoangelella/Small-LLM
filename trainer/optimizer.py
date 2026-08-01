"""Optimizer construction from the model's decay contract."""
from __future__ import annotations
from torch import nn
from torch.optim import AdamW
from model.accounting import optimizer_no_weight_decay_parameter_names
from .config import TrainerConfig

def build_adamw(model: nn.Module, config: TrainerConfig) -> AdamW:
    exclusions = optimizer_no_weight_decay_parameter_names(model)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = {name for name, _ in named}
    unknown = exclusions - names
    if unknown:
        raise ValueError(f"weight-decay exclusions name missing parameters: {sorted(unknown)}")
    decay = [parameter for name, parameter in named if name not in exclusions]
    no_decay = [parameter for name, parameter in named if name in exclusions]
    if not decay:
        raise ValueError("AdamW decay parameter group is empty")
    groups: list[dict[str, object]] = [{"params": decay, "weight_decay": config.weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return AdamW(groups, lr=config.learning_rate, betas=(config.beta1, config.beta2),
                 eps=config.adam_epsilon)
