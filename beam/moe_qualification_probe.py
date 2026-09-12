"""Throughput arm of the production qualification: the accepted geometry exactly as
`MoEModelConfig.accepted()` defines it (64 experts, Top-2, Quantile Balancing on), on real
SuperBPE-8000 schema-v2 blocks, engine-level, warm synchronized update seconds.

argv: shard_bin out steps microbatch precision compile_mode
"""
import json, math, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
from trainer import TokenBatch, TrainerConfig
from MOE_model.config import MoEModelConfig
from MOE_model.model import MoESmallLLM
from MOE_model.engine import MoETrainerEngine
from MOE_model.initialization import initialize_moe_model

shard, output = Path(sys.argv[1]), Path(sys.argv[2])
steps, microbatch, precision = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
compile_mode = sys.argv[6] if len(sys.argv) > 6 else "off"

need = steps + 1
raw = shard.read_bytes()[: need * 64 * 2049 * 2]
rows = np.frombuffer(raw, dtype='<u2').reshape(need, 64, 2049).astype(np.int64)
config = MoEModelConfig.accepted()
assert int(rows.max()) < config.semantic_vocab_size, 'token id outside the semantic vocabulary'

torch.set_num_threads(2); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
model = initialize_moe_model(MoESmallLLM(config), 'normal')
result = {
    'arm': output.stem, 'microbatch': microbatch, 'precision': precision, 'compile': compile_mode,
    'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
    'num_experts': config.num_experts, 'top_k': config.top_k, 'load_balancing': config.load_balancing,
    'parameters': sum(p.numel() for p in model.parameters()),
    'data_sha256': hashlib.sha256(raw).hexdigest(), 'steps': [], 'status': 'running',
}
output.write_text(json.dumps(result, indent=2))

compile_seconds = None
if compile_mode != 'off':
    # Filled in once the opt-in compile lane exists in the tree; see qual_modal.py.
    from MOE_model.compile_lane import apply_compile_lane  # noqa: E402
    lane, _, variant = compile_mode.partition('+')
    t = time.perf_counter()
    apply_compile_lane(model, lane, torch_mode=(variant or None))
    compile_seconds = time.perf_counter() - t

engine = MoETrainerEngine(
    model,
    TrainerConfig(optimizer='hybrid_muon_adamw', precision=precision, microbatch_size=microbatch, seed=17),
    device='cuda',
)


def block(index):
    r = rows[index]
    return TokenBatch(index, 'train', torch.from_numpy(r[:, :-1].copy()),
                      torch.from_numpy(r[:, 1:].copy()), 64, 131072)


try:
    for i in range(steps):
        torch.cuda.synchronize(); start = time.perf_counter()
        metric = engine.train_batch(block(i)); torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert math.isfinite(metric.loss) and math.isfinite(metric.gradient_norm)
        router = model.blocks[0].ffn.router
        row = {'step': i + 1, 'synchronized_seconds': elapsed, 'loss': metric.loss,
               'gradient_norm': metric.gradient_norm,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'layer0_bias_absmax': float(router.selection_bias.abs().max().item())}
        result['steps'].append(row)
        output.write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(row, default=str), flush=True)
    result['compile_seconds'] = compile_seconds
    exposure = torch.stack([b.ffn.router.token_exposure.float() for b in model.blocks])
    fractions = exposure / exposure.sum(dim=1, keepdim=True).clamp_min(1)
    result['routing'] = {'max_load_fraction_per_layer': fractions.max(dim=1).values.tolist(),
                         'min_load_fraction_per_layer': fractions.min(dim=1).values.tolist(),
                         'uniform_fraction': 1.0 / config.num_experts,
                         'dead_slots': int((exposure == 0).sum().item())}
    result['status'] = 'pass'
except Exception as e:
    result.update(status='fail', error=repr(e)); raise
finally:
    output.write_text(json.dumps(result, indent=2, default=str))
