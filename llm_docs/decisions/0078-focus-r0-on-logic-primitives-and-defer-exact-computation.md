---
status: accepted
last_reviewed: 2026-08-14
---

# ADR 0078 — Focus R0 on logic primitives and defer exact computation

## Decision

The first reasoning-oriented SFT stage (`R-SFT R0`) will prioritize transferable logical reasoning primitives rather than arithmetic calculation or calculus.

The accepted R0 skill taxonomy is:

- `INF` — immediate inference: negation, quantifiers, implication, conjunction/disjunction, and simple categorical transformations;
- `DED` — deduction: derive necessary conclusions from explicit premises or rules;
- `REL` — relational reasoning: ordering, transitivity, containment, temporal/spatial relationships, and comparisons;
- `CSP` — constraint reasoning: eliminate alternatives and satisfy several simultaneous restrictions;
- `IND` — induction: infer a compact rule or pattern from controlled observations;
- `ABD` — abduction: select the explanation most consistent with supplied evidence from a controlled hypothesis set;
- `MAG` — numerical magnitude awareness: less/greater/equal, sign, ranges, min/max, approximate magnitude, and ordering of quantities.

Exact arithmetic, long-form numerical calculation, algebraic manipulation, and calculus are not core R0 skills. Numerical values may appear as symbols participating in logical/relational tasks, but the target capability is understanding relations between quantities rather than becoming a standalone calculator.

Exact computation will be explored later through tool-use cold-start supervision and tool-aided RL/RLVR, where the model learns to identify what needs to be computed, invoke an appropriate calculator/symbolic tool, and interpret the returned result.

## Difficulty contract

The previously accepted three shuffled difficulty bands are now defined by dependent reasoning depth rather than prompt length or arithmetic size:

- `L1`: one atomic inference or relation;
- `L2`: roughly 2–3 dependent inferences;
- `L3`: roughly 3–5 dependent inferences, optionally including a small amount of branching or elimination.

R0 traces remain concise and capacity-aligned. Difficulty and skill labels remain dataset metadata only and are not exposed in the model-visible input/target. Training remains globally shuffled across the three bands.

## Measurement

Validation/qualification should report loss and behavioral accuracy across the `skill × difficulty` grid, including reasoning-span and answer-span metrics where the serialization permits it. The goal is to observe which logical capabilities emerge first and which remain bottlenecks without using the labels to steer the initial training order.

## Rationale

The project deliberately separates reasoning from computation. Recent 2026 work treats arithmetic computation as distinct from semantic understanding and mathematical/logical reasoning, shows that numerical magnitude/ranking representations are useful even when explicit arithmetic is weak, and demonstrates transfer from explicitly trained logical primitives such as deduction, induction, abduction, quantifier handling, and relational inference.

For an approximately-100M general-purpose student, spending scarce capacity on transferable inference structure is preferred over attempting to internalize reliable exact calculation. Tools can later provide exact computation while the model remains responsible for deciding what operation is needed and how the result affects the answer.
