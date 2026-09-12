"""CPU contracts for the opt-in ``torch.compile`` execution lane.

Numerical comparisons use ``backend="aot_eager"`` on CPU: FP32 agreement,
unchanged state-dict keys, bounded graphs and capacity-driven recompilation.
Selection bias is bit-identical only when the router inputs are identical, as
in this CPU test. On GPU with Inductor, H100 first-update layer0 bias absmax
was 0.0332623720 eager versus 0.0332735777 compiled, a rounding-level difference.
CPU coverage does not establish GPU backward equivalence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch._dynamo as dynamo

from MOE_model.compile_lane import (
    COMPILE_MODES,
    DEFAULT_COMPILE_MODE,
    apply_compile_lane,
    compiled_block_count,
    router_bookkeeping_is_excluded,
)
from MOE_model.config import MoEModelConfig
from MOE_model.initialization import initialize_moe_model
from MOE_model.model import MoESmallLLM
from MOE_model.__main__ import parse_args as parse_moe_args
from MOE_model.setup import setup
from model.config import ModelConfig
from tests.trainer_fixtures import payload

BACKEND = "aot_eager"


def _tiny_accepted(n_layers: int = 4) -> MoEModelConfig:
    """The accepted geometry with the layer count, width and expert count reduced.

    Every axis the compile lane touches is preserved: the frozen
    ``(gdn, gdn, gdn, mha)`` pattern, Top-2 routing, the padded batched expert
    GEMM, sqrt-softplus affinities and Quantile Balancing.
    """

    accepted = MoEModelConfig.accepted()
    dense = ModelConfig(
        semantic_vocab_size=64, padded_vocab_size=64, max_seq_len=32,
        d_model=16, n_layers=n_layers, d_ff=10, n_heads=2, head_dim=8,
        gdn_num_key_heads=2, gdn_num_value_heads=2, gdn_key_dim=8, gdn_value_dim=8,
        gdn_conv_kernel_size=4, gdn_chunk_size=4,
    )
    return MoEModelConfig(
        dense=dense, num_experts=6, top_k=accepted.top_k, expert_d_ff=10,
        expert_type=accepted.expert_type, router_combine=accepted.router_combine,
        dispatch=accepted.dispatch, moe_layer_indices=tuple(range(n_layers)),
        router_scoring=accepted.router_scoring,
        router_init_std=accepted.router_init_std,
        router_z_loss_coefficient=accepted.router_z_loss_coefficient,
        load_balancing=accepted.load_balancing, version=accepted.version,
    )


def _model(seed: int = 11) -> MoESmallLLM:
    torch.manual_seed(seed)
    return initialize_moe_model(MoESmallLLM(_tiny_accepted()), "normal")


def _tokens(model: MoESmallLLM, *, sequences: int, length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, model.config.semantic_vocab_size, (sequences, length), generator=generator
    )


def _loss_and_grads(model: MoESmallLLM, inputs: torch.Tensor):
    model.zero_grad(set_to_none=True)
    logits, aux = model.forward_with_aux(inputs)
    loss = logits.float().square().mean() + aux.z_loss
    loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return float(loss.detach()), grads


class CompileLaneContract(unittest.TestCase):
    def test_modes_and_default(self) -> None:
        self.assertEqual(COMPILE_MODES, ("off", "blocks"))
        self.assertEqual(DEFAULT_COMPILE_MODE, "off")

    def test_off_is_a_strict_no_op(self) -> None:
        model = _model()
        apply_compile_lane(model, "off")
        self.assertEqual(compiled_block_count(model), 0)

    def test_armed_lane_matches_backward_outside_autocast(self) -> None:
        from torch._functorch import config as functorch_config

        if not hasattr(functorch_config, "backward_pass_autocast"):
            self.skipTest("this torch version has no backward autocast setting")
        with functorch_config.patch(backward_pass_autocast="same_as_forward"):
            model = _model()
            apply_compile_lane(model, "off")
            self.assertEqual(functorch_config.backward_pass_autocast, "same_as_forward")
            apply_compile_lane(model, "blocks", backend=BACKEND)
            self.assertEqual(functorch_config.backward_pass_autocast, "off")

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_compile_lane(_model(), "graphs")

    def test_blocks_mode_compiles_every_decoder_block(self) -> None:
        model = _model()
        apply_compile_lane(model, "blocks", backend=BACKEND)
        self.assertEqual(compiled_block_count(model), model.config.n_layers)
        self.assertTrue(router_bookkeeping_is_excluded())


class CompiledStateDictIsUnchanged(unittest.TestCase):
    def test_state_dict_keys_and_parameter_identity_survive_compilation(self) -> None:
        model = _model()
        keys_before = list(model.state_dict().keys())
        parameters_before = {name: id(p) for name, p in model.named_parameters()}
        captured = [id(p) for p in model.parameters()]

        apply_compile_lane(model, "blocks", backend=BACKEND)

        self.assertEqual(list(model.state_dict().keys()), keys_before)
        self.assertEqual(
            {name: id(p) for name, p in model.named_parameters()}, parameters_before
        )
        self.assertEqual([id(p) for p in model.parameters()], captured)
        # A checkpoint written eager must load into the compiled model and back.
        eager = _model()
        model.load_state_dict(eager.state_dict())
        eager.load_state_dict(model.state_dict())
        self.assertNotIn("_orig_mod", " ".join(keys_before))


class CompiledMatchesEager(unittest.TestCase):
    def test_loss_and_gradients_agree_on_identical_weights_and_inputs(self) -> None:
        eager = _model()
        compiled = _model()
        compiled.load_state_dict(eager.state_dict())
        inputs = _tokens(eager, sequences=2, length=16, seed=3)

        eager_loss, eager_grads = _loss_and_grads(eager, inputs)
        apply_compile_lane(compiled, "blocks", backend=BACKEND)
        compiled_loss, compiled_grads = _loss_and_grads(compiled, inputs)

        self.assertAlmostEqual(eager_loss, compiled_loss, delta=1e-4 * max(1.0, abs(eager_loss)))
        self.assertEqual(set(eager_grads), set(compiled_grads))
        for name, expected in eager_grads.items():
            torch.testing.assert_close(
                compiled_grads[name], expected, rtol=1e-5, atol=1e-6, msg=name
            )


    def test_backward_matches_eager_across_two_capacities(self) -> None:
        """CPU aot_eager backward agrees within FP32 tolerance at both capacities."""
        from torch._dynamo.backends.registry import lookup_backend

        eager = _model()
        compiled = _model()
        compiled.load_state_dict(eager.state_dict())
        index = eager.layer_kinds.index("mha")
        eager_block, compiled_block = eager.blocks[index], compiled.blocks[index]
        executed_frames = 0
        aot_eager = lookup_backend(BACKEND)

        def counting_backend(graph, example_inputs):
            compiled_graph = aot_eager(graph, example_inputs)

            def execute(*args):
                nonlocal executed_frames
                executed_frames += 1
                return compiled_graph(*args)

            return execute

        dynamo.reset()
        self.addCleanup(dynamo.reset)
        apply_compile_lane(compiled, "blocks", backend=counting_backend)
        capacities = []
        for sequences in (2, 4):
            hidden = eager.token_embedding(
                _tokens(eager, sequences=sequences, length=16, seed=sequences * 7)
            ).detach()
            eager_input = hidden.clone().requires_grad_()
            compiled_input = hidden.clone().requires_grad_()
            eager_block.zero_grad(set_to_none=True)
            compiled_block.zero_grad(set_to_none=True)
            expected, eager_z, eager_telemetry = eager_block(eager_input)
            before = executed_frames
            actual, compiled_z, telemetry = compiled_block(compiled_input)
            (expected.square().mean() + eager_z).backward()
            (actual.square().mean() + compiled_z).backward()
            self.assertGreater(executed_frames, before, "no compiled frame executed")
            capacity = int(telemetry.expert_counts.max())
            self.assertEqual(capacity, int(eager_telemetry.expert_counts.max()))
            capacities.append(capacity)
            torch.testing.assert_close(compiled_input.grad, eager_input.grad,
                                       rtol=1e-5, atol=1e-6)
            for (name, parameter), (actual_name, actual_parameter) in zip(
                eager_block.named_parameters(), compiled_block.named_parameters(), strict=True
            ):
                self.assertEqual(name, actual_name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertIsNotNone(actual_parameter.grad, name)
                torch.testing.assert_close(actual_parameter.grad, parameter.grad,
                                           rtol=1e-5, atol=1e-6, msg=name)
        self.assertEqual(len(set(capacities)), 2, "test did not vary the expert capacity")


class QuantileBiasIsBitIdentical(unittest.TestCase):
    """CPU aot_eager bias is bit-identical only with identical router inputs."""

    @staticmethod
    def _one_step(model: MoESmallLLM, microbatches: list[torch.Tensor]) -> list[torch.Tensor]:
        model.train()
        positions = sum(int(batch.numel()) for batch in microbatches)
        for block in model.blocks:
            block.ffn.router.begin_step(positions)
        counts = [
            torch.zeros(model.config.num_experts, dtype=torch.long)
            for _ in range(model.config.n_layers)
        ]
        for micro in microbatches:
            _, aux = model.forward_with_aux(micro)
            for accumulator, layer in zip(counts, aux.layers, strict=True):
                accumulator.add_(layer.expert_counts)
        for block, accumulated in zip(model.blocks, counts, strict=True):
            block.ffn.router.commit_load(accumulated)
        return [
            block.ffn.router.selection_bias.detach().clone() for block in model.blocks
        ]

    def test_selection_bias_after_one_step_is_bit_identical(self) -> None:
        eager = _model()
        compiled = _model()
        compiled.load_state_dict(eager.state_dict())
        microbatches = [
            _tokens(eager, sequences=2, length=16, seed=5),
            _tokens(eager, sequences=1, length=16, seed=6),
        ]

        eager_bias = self._one_step(eager, [m.clone() for m in microbatches])
        apply_compile_lane(compiled, "blocks", backend=BACKEND)
        compiled_bias = self._one_step(compiled, [m.clone() for m in microbatches])

        self.assertTrue(any(bool(bias.abs().sum()) for bias in eager_bias))
        for layer, (expected, actual) in enumerate(zip(eager_bias, compiled_bias, strict=True)):
            self.assertTrue(
                torch.equal(expected, actual),
                f"layer {layer} selection bias drifted: {expected} vs {actual}",
            )


class GraphBreaksAreBounded(unittest.TestCase):
    """Record what Dynamo actually produces for one block, and hold it bounded.

    The mixer dominates the count. On CPU the GDN-2 block runs the adaptive
    chunkwise fallback, whose data-dependent bisection loop breaks repeatedly;
    on CUDA that whole path is replaced by FLA's ``ChunkGDN2Function``, so the
    GDN number here is an upper bound for a different kernel, not a prediction.
    The MHA block is the honest reading of the MoE part itself.
    """

    LIMITS = {"mha": 10, "gdn": 24}

    def test_one_block_stays_within_a_bounded_number_of_graphs(self) -> None:
        model = _model()
        dynamo.reset()
        hidden = model.token_embedding(_tokens(model, sequences=2, length=16, seed=7))
        observed: dict[str, tuple[int, int]] = {}
        for index, kind in enumerate(model.layer_kinds):
            if kind in observed:
                continue
            explanation = dynamo.explain(model.blocks[index])(hidden)
            observed[kind] = (explanation.graph_count, explanation.graph_break_count)
            reasons = sorted(
                {str(getattr(r, "reason", r)).splitlines()[0] for r in explanation.break_reasons}
            )
            print(
                f"[compile-lane] block {index} ({kind}): "
                f"graphs={explanation.graph_count} breaks={explanation.graph_break_count}"
            )
            for reason in reasons:
                print(f"[compile-lane]     break reason: {reason}")
        dynamo.reset()

        self.assertIn("mha", observed)
        for kind, (graphs, breaks) in observed.items():
            self.assertGreaterEqual(graphs, 1)
            self.assertLessEqual(
                graphs, self.LIMITS[kind],
                f"{kind} block produced {graphs} graphs ({breaks} breaks)",
            )


class CapacityChangesDoNotRecompile(unittest.TestCase):
    """``capacity = int(counts.max())`` varies every microbatch and must not recompile."""

    @staticmethod
    def _frames() -> int:
        return int(dynamo.utils.counters["frames"]["total"])

    def test_consecutive_microbatches_with_different_loads_reuse_one_compilation(self) -> None:
        model = _model()
        apply_compile_lane(model, "blocks", backend=BACKEND)
        dynamo.reset()
        dynamo.utils.counters.clear()
        model.train()

        capacities: list[int] = []
        frame_totals: list[int] = []
        for seed in (21, 22, 23):
            _, aux = model.forward_with_aux(_tokens(model, sequences=2, length=16, seed=seed))
            capacities.append(max(int(layer.expert_counts.max()) for layer in aux.layers))
            frame_totals.append(self._frames())

        print(f"[compile-lane] whole model capacities={capacities} frames={frame_totals}")
        self.assertGreater(len(set(capacities)), 1, "test did not vary the expert load")
        self.assertEqual(
            frame_totals[0], frame_totals[-1],
            f"a changed capacity recompiled: frame totals {frame_totals}",
        )
        self.assertFalse(dict(dynamo.utils.counters.get("recompiles", {})))

    def test_expert_dispatch_also_absorbs_a_changed_microbatch_size(self) -> None:
        """Isolated on the MHA block: its mixer has no data-dependent Python branch,
        so every frame counted here belongs to the router and the expert dispatch.
        The CPU-only adaptive GDN fallback does re-specialise on a changed sequence
        count; FLA's fixed 64-token CUDA kernel replaces that path in production."""

        model = _model()
        block = model.blocks[model.layer_kinds.index("mha")]
        apply_compile_lane(model, "blocks", backend=BACKEND)
        dynamo.reset()
        dynamo.utils.counters.clear()
        model.train()

        capacities: list[int] = []
        frame_totals: list[int] = []
        for sequences in (2, 3, 2, 4):
            hidden = model.token_embedding(
                _tokens(model, sequences=sequences, length=16, seed=sequences * 7)
            )
            _, _, telemetry = block(hidden)
            capacities.append(int(telemetry.expert_counts.max()))
            frame_totals.append(self._frames())

        print(f"[compile-lane] mha block capacities={capacities} frames={frame_totals}")
        self.assertGreater(len(set(capacities)), 1, "test did not vary the expert load")
        self.assertEqual(
            frame_totals, [frame_totals[0]] * len(frame_totals),
            f"the expert dispatch recompiled: frame totals {frame_totals}",
        )


class CommandLineWiring(unittest.TestCase):
    """``--compile`` reaches ``MOE_model.setup`` and changes nothing else."""

    @staticmethod
    def _dataset(root: Path) -> Path:
        data = root / "data"
        data.mkdir()
        shards = []
        for split, rows in (("train", 4), ("validation", 2)):
            (data / split).mkdir()
            raw = payload([[1 + i, 2 + i, 3 + i, 4 + i] for i in range(rows)])
            name = f"{split}/000.bin"
            (data / name).write_bytes(raw)
            shards.append(
                dict(
                    filename=name, split=split, byte_size=len(raw),
                    checksum=hashlib.sha256(raw).hexdigest(), sequence_count=rows,
                    first_block_id=0, last_block_id=rows // 2 - 1,
                )
            )
        (data / "manifest.json").write_text(
            json.dumps(
                dict(
                    schema_version=2, sequence_format="context_plus_one", context_length=3,
                    stored_tokens_per_sequence=4, sequences_per_block=2, shards=shards,
                )
            )
        )
        return data

    def _setup_with(self, *extra: str):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            args = parse_moe_args(
                [
                    "--dataset-dir", str(data),
                    "--checkpoint-dir", str(root / "checkpoints"),
                    "--steps", "1", "--model-size", "smoke", "--device", "cpu",
                    "--precision", "fp32", "--microbatch-size", "1", "--seed", "13",
                    *extra,
                ]
            )
            model_config, _, engine, _, _ = setup(args)
            return args, model_config, engine.model

    def test_default_is_off_and_nothing_is_compiled(self) -> None:
        args, config, model = self._setup_with()
        self.assertEqual(args.compile_mode, "off")
        self.assertEqual(compiled_block_count(model), 0)
        self.assertEqual(config, MoEModelConfig.smoke())

    def test_blocks_mode_compiles_and_leaves_the_checkpoint_identity_alone(self) -> None:
        args, config, model = self._setup_with("--compile", "blocks")
        self.assertEqual(args.compile_mode, "blocks")
        self.assertEqual(compiled_block_count(model), config.n_layers)
        # The compile mode is execution, never checkpoint-visible architecture.
        self.assertEqual(config, MoEModelConfig.smoke())
        self.assertNotIn("compile", config.as_dict())

    def test_an_unknown_mode_is_refused_by_the_parser(self) -> None:
        with self.assertRaises(SystemExit):
            parse_moe_args(["--dataset-dir", ".", "--compile", "graphs"])


if __name__ == "__main__":
    unittest.main()
