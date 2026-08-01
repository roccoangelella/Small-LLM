from __future__ import annotations
from dataclasses import dataclass
import pickle
import torch
from torch import nn
from trainer import TokenBatch

def payload(sequences: list[list[int]]) -> bytes:
    result = bytearray()
    for sequence in sequences:
        for token in sequence:
            result.extend(int(token).to_bytes(2, "little"))
    return bytes(result)

@dataclass
class PreparedBlock:
    block_id: int
    split: str
    sequence_count: int
    token_count: int
    payload: bytes
    cumulative_source_tokens: int = 0
    schema_version: int = 2

class TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, width: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab, bias=False)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(self.embedding(input_ids)))

class Coordinator:
    def __init__(self):
        self.saved = {}
    def save(self, *, checkpoint_id, trainer, pipeline_state,
             optimizer_step_complete, validation_metrics=None):
        if not optimizer_step_complete:
            raise AssertionError
        self.saved[checkpoint_id] = pickle.dumps((trainer.state_dict(), dict(pipeline_state)))
        return checkpoint_id
    def load(self, checkpoint_id, trainer):
        state, pipeline = pickle.loads(self.saved[checkpoint_id])
        trainer.load_state_dict(state)
        return pipeline

def batch(block_id: int, offset: int = 0, split: str = "train") -> TokenBatch:
    inputs = torch.tensor([[1 + offset, 2 + offset, 3 + offset],
                           [2 + offset, 3 + offset, 4 + offset]]) % 16
    labels = torch.tensor([[2 + offset, 3 + offset, 4 + offset],
                           [3 + offset, 4 + offset, 5 + offset]]) % 16
    return TokenBatch(block_id, split, inputs.long(), labels.long(), 2, 6)
