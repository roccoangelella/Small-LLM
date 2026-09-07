# ADR 0156 — Expose training-free looping through an opt-in chat flag

Date: 2026-09-07
Status: Accepted interface; loop semantics still gated by ADR 0155

## Decision

The training-free looping experiment will be implemented on branch `looping_transformer` as an opt-in inference path rather than by mutating the production `SmallLLM` checkpoint format or retraining the model.

The experimental implementation will:

- live in a dedicated new inference module/file rather than embedding the recurrence logic directly into the checkpoint loader;
- expose the experiment through `chat.py --loop`;
- leave normal `chat.py` behavior unchanged when `--loop` is absent;
- reuse the already-loaded frozen `SmallLLM` modules and weights rather than copying or replacing checkpoint parameters;
- preserve `model.eval()` / inference-only execution and the current tokenizer, chat template, decoding configuration, checkpoint verification, and artifact cache behavior;
- initially target the completed 100M/10B pretrained model; broader stage/profile support is not implied by this decision;
- print the active loop configuration so an interactive chat session cannot be mistaken for the normal forward path.

## Safety/implementation observation

The current GDN-2 forward path is suitable for repeated inference calls without persistent hidden-state contamination when no cache is passed: it constructs local recurrent state/history for the call and returns only the output unless cache return is explicitly requested. The chat path already runs the model in evaluation mode under `torch.inference_mode()` during generation.

## Still not authorized

ADR 0155's implementation gate remains active. This ADR does not yet authorize a specific layer window, block-vs-layer recurrence mode, recurrence count `K`, damping rule, or anchor coefficient. Those semantics must be understood and explicitly selected before the looping forward path is wired.
