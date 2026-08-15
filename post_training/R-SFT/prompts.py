"""Teacher prompt contracts for the reasoning-SFT synthetic data generator.

The project keeps internal skill codes and difficulty labels out of teacher-facing
text. Callers select an internal skill here and may provide plain-language
structural requirements derived from the project's difficulty metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

DEFAULT_BATCH_SIZE = 10
R0_SKILLS = ("INF", "DED", "REL", "CSP", "IND", "ABD", "MAG")


@dataclass(frozen=True, slots=True)
class SkillPromptSpec:
    name: str
    definition: str
    good_structures: tuple[str, ...]
    avoid: str
    example_problem: str
    example_reasoning: str
    example_answer: str
    diversity_note: str


SKILL_PROMPT_SPECS: dict[str, SkillPromptSpec] = {
    "INF": SkillPromptSpec(
        name="immediate-inference",
        definition=(
            "The main challenge in every problem must be immediate logical inference: "
            "the answer should follow from a small local set of explicit statements by "
            "correctly interpreting their logical form, rather than by performing a long "
            "multi-stage deduction."
        ),
        good_structures=(
            "applying a universal or categorical statement to a stated case",
            "interpreting negation, conjunction, disjunction, implication, or an explicit quantifier",
            "applying a necessary or sufficient condition in the valid direction",
            "recognizing whether a local conclusion follows or does not follow",
        ),
        avoid=(
            "Do not turn these into long implication chains, multi-variable constraint puzzles, "
            "arithmetic exercises, or tasks whose main difficulty is search."
        ),
        example_problem=(
            "Every ceramic token in the archive is heat-resistant. Tile K is a ceramic token "
            "in the archive. What follows about Tile K?"
        ),
        example_reasoning=(
            "Tile K is one of the ceramic tokens covered by the rule, and every such token is "
            "heat-resistant. Therefore Tile K is heat-resistant."
        ),
        example_answer="Tile K is heat-resistant.",
        diversity_note=(
            "Vary the local logical form across the batch; do not let simple universal-rule "
            "application dominate every example."
        ),
    ),
    "DED": SkillPromptSpec(
        name="deductive",
        definition=(
            "The main challenge in every problem must be deduction: the answer should follow "
            "by combining explicit facts, rules, conditions, or logical statements supplied "
            "in the problem."
        ),
        good_structures=(
            "chaining implications or rules",
            "applying necessary or sufficient conditions",
            "combining multiple premises",
            "using negation or contraposition when valid",
            "reasoning from exclusive or exhaustive alternatives",
            "determining what must follow, what cannot follow, or what remains possible",
        ),
        avoid=(
            "Do not make arithmetic, calculation, scheduling, ordering puzzles, spatial puzzles, "
            "or constraint-search the primary challenge. Small amounts of these may appear in "
            "the setting, but the solution should fundamentally depend on deduction from the "
            "stated premises."
        ),
        example_problem=(
            "An archive stores every restricted document in an encrypted partition. Every "
            "document in the encrypted partition requires supervisor approval before it can be "
            "opened. Report K is a restricted document. What follows about opening Report K?"
        ),
        example_reasoning=(
            "Report K is restricted, so it is stored in the encrypted partition. Documents in "
            "that partition require supervisor approval before they can be opened. Therefore, "
            "Report K requires supervisor approval before opening."
        ),
        example_answer="Report K requires supervisor approval before it can be opened.",
        diversity_note=(
            "Vary the underlying deductive pattern across the batch. Do not let one pattern such "
            "as contraposition, simple implication chaining, biconditionals, or elimination "
            "dominate the examples."
        ),
    ),
    "REL": SkillPromptSpec(
        name="relational",
        definition=(
            "The main challenge in every problem must be relational reasoning: the answer should "
            "follow by composing or comparing explicit relations between entities while keeping "
            "the direction and meaning of each relation correct."
        ),
        good_structures=(
            "transitive comparisons such as taller, earlier, heavier, or higher-priority",
            "containment and nested membership relations",
            "before/after and other temporal relations",
            "left/right, north/south, inside/outside, or other explicitly defined spatial relations",
            "determining an entity's relation to another after combining stated relations",
        ),
        avoid=(
            "Do not make the main challenge a large scheduling or assignment search, exact "
            "calculation, or a general deductive rule chain unrelated to relations."
        ),
        example_problem=(
            "Box R is stored inside Cabinet M. Cabinet M is inside Storage Room Q. Where is Box R "
            "relative to Storage Room Q?"
        ),
        example_reasoning=(
            "Box R is inside Cabinet M, and Cabinet M is inside Storage Room Q. Containment carries "
            "through the nesting, so Box R is inside Storage Room Q."
        ),
        example_answer="Box R is inside Storage Room Q.",
        diversity_note=(
            "Vary relation types across the batch rather than producing ten versions of the same "
            "linear ordering chain."
        ),
    ),
    "CSP": SkillPromptSpec(
        name="constraint-reasoning",
        definition=(
            "The main challenge in every problem must be satisfying several explicit constraints "
            "at once. The solver should have to eliminate incompatible possibilities or combine "
            "restrictions until a unique requested conclusion follows."
        ),
        good_structures=(
            "small assignment problems with mutually exclusive options",
            "compatibility or incompatibility constraints",
            "small ordering or placement problems that require multiple restrictions simultaneously",
            "eliminating alternatives until one valid possibility remains",
            "determining a forced choice from a compact set of explicit constraints",
        ),
        avoid=(
            "Keep the search space compact and transparent. Do not create giant logic-grid puzzles, "
            "long brute-force searches, heavy arithmetic, or problems with multiple equally valid "
            "answers."
        ),
        example_problem=(
            "Nora, Luca, and Sara each occupy one of seats 1, 2, and 3, with no shared seats. Luca "
            "is in seat 2. Nora cannot sit in seat 1. Which seat must Nora occupy?"
        ),
        example_reasoning=(
            "Luca already occupies seat 2. Nora cannot use seat 1, so the only remaining seat "
            "available to Nora is seat 3."
        ),
        example_answer="Nora must occupy seat 3.",
        diversity_note=(
            "Vary the kinds of constraints and solution structures across the batch; do not merely "
            "change the names in one seat-assignment template."
        ),
    ),
    "IND": SkillPromptSpec(
        name="inductive",
        definition=(
            "The main challenge in every problem must be induction from controlled observations: "
            "the solver should infer a compact rule or pattern that explains the supplied examples "
            "and then apply that rule to a new case."
        ),
        good_structures=(
            "inferring which explicitly available attribute controls a label or outcome",
            "discovering a compact mapping from a small table of controlled observations",
            "identifying a simple rule from examples when the allowed hypothesis space is stated",
            "applying the inferred rule to a new observation",
        ),
        avoid=(
            "Do not use arbitrary number sequences, trivia, real-world generalization, or examples "
            "where many unrelated rules fit equally well. Define enough of the hypothesis space and "
            "provide enough observations that the intended rule is identifiable."
        ),
        example_problem=(
            "A sorter labels tokens using exactly one property: shape or color. Red circles are "
            "labeled X, blue circles are labeled X, red squares are labeled Y, and blue squares are "
            "labeled Y. What label should a green square receive?"
        ),
        example_reasoning=(
            "Changing color does not change the label for circles or squares, while changing shape "
            "does. The labeling rule therefore depends on shape: circles are X and squares are Y. "
            "A green square should be labeled Y."
        ),
        example_answer="The green square should be labeled Y.",
        diversity_note=(
            "Vary the latent rule and representation across the batch while keeping each rule "
            "uniquely recoverable from the supplied observations."
        ),
    ),
    "ABD": SkillPromptSpec(
        name="abductive",
        definition=(
            "The main challenge in every problem must be controlled abduction: given a finite set "
            "of explicitly described candidate explanations and observed evidence, the solver should "
            "identify the explanation that is compatible with the evidence or best accounts for it "
            "under the stated rules."
        ),
        good_structures=(
            "selecting among explicitly enumerated hypotheses from their predicted observations",
            "eliminating candidate explanations that conflict with supplied evidence",
            "matching a distinctive evidence signature to one hypothesis",
            "choosing the only explanation consistent with all stated observations",
        ),
        avoid=(
            "Do not rely on common-sense causation, medical or scientific background knowledge, or "
            "unstated likelihoods. Keep the hypothesis set closed and make the evidence-to-hypothesis "
            "relationship explicit enough to yield a well-defined answer."
        ),
        example_problem=(
            "Exactly one of three faults caused a machine alert. Fault A produces a red light only. "
            "Fault B produces a buzzing sound only. Fault C produces both a red light and a buzzing "
            "sound. The machine shows a red light and is buzzing. Which fault caused the alert?"
        ),
        example_reasoning=(
            "The observed alert has both a red light and a buzzing sound. Fault A explains only the "
            "light, and Fault B explains only the buzzing. Fault C predicts both observations, so it "
            "is the compatible explanation."
        ),
        example_answer="Fault C caused the alert.",
        diversity_note=(
            "Vary the hypothesis structures and evidence signatures across the batch; do not make "
            "every problem a renamed machine-fault diagnosis."
        ),
    ),
    "MAG": SkillPromptSpec(
        name="numerical-magnitude",
        definition=(
            "The main challenge in every problem must be reasoning about numerical magnitude rather "
            "than carrying out long exact calculations. The answer should follow from signs, ranges, "
            "inequalities, ordering, bounds, minima/maxima, or approximate relative size."
        ),
        good_structures=(
            "comparing positive and negative quantities",
            "combining inequality or range information",
            "determining which quantity must be larger or smaller",
            "reasoning from upper or lower bounds",
            "identifying a forced minimum, maximum, sign, or magnitude relationship",
        ),
        avoid=(
            "Do not make long arithmetic, multi-step exact calculation, algebraic manipulation, "
            "calculus, or memorized numerical facts the primary challenge."
        ),
        example_problem=(
            "Quantity A is greater than Quantity B, and Quantity B is greater than 12. What must be "
            "true about Quantity A relative to 12?"
        ),
        example_reasoning=(
            "Quantity B is greater than 12, and Quantity A is greater than B. Therefore Quantity A "
            "must also be greater than 12."
        ),
        example_answer="Quantity A must be greater than 12.",
        diversity_note=(
            "Vary signs, bounds, comparisons, ranges, and ordering structures across the batch rather "
            "than repeating one inequality chain."
        ),
    ),
}


def _bullet_lines(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item};" for item in items)


def build_generation_prompt(
    skill: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    structural_requirements: str | None = None,
) -> str:
    """Build one teacher-facing batch prompt for an internal R0 skill code.

    ``structural_requirements`` is deliberately plain-language text. Callers may
    derive it from internal difficulty metadata, but labels such as L1/L2/L3
    should not be passed through to the teacher.
    """

    normalized_skill = skill.strip().upper() if isinstance(skill, str) else ""
    if normalized_skill not in SKILL_PROMPT_SPECS:
        raise ValueError(f"unknown R0 reasoning skill: {skill!r}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if structural_requirements is not None:
        if not isinstance(structural_requirements, str) or not structural_requirements.strip():
            raise ValueError("structural_requirements must be a non-empty string when provided")
        structural_block = (
            "\nFor this batch, also satisfy these structural requirements:\n"
            f"{structural_requirements.strip()}\n"
        )
    else:
        structural_block = ""

    spec = SKILL_PROMPT_SPECS[normalized_skill]
    return dedent(
        f"""
        Generate {batch_size} self-contained {spec.name} reasoning problems.

        {spec.definition}

        Good problem structures include:
        {_bullet_lines(spec.good_structures)}

        Keep every problem fully self-contained. Every fact, definition, relationship, or rule needed
        to solve it must be explicitly stated. Do not rely on outside knowledge, unstated assumptions,
        common-sense facts, or hidden definitions.

        All stated premises must be mutually consistent unless the problem explicitly asks the solver
        to identify a contradiction. Never resolve an inconsistency by silently ignoring one of the
        premises.

        Use precise wording. Be unambiguous about quantities, thresholds, exclusivity, relations, and
        necessary or sufficient conditions whenever they matter.

        {spec.avoid}

        {structural_block}
        Prefer open-ended questions when natural. Ask for the conclusion, state, entity, classification,
        consequence, relation, supported claim, contradiction, or other result that follows from the
        supplied information. Yes/no questions are allowed when genuinely natural, but do not make them
        the dominant question type.

        Here is one example of the style and reasoning structure we want:

        Problem:
        {spec.example_problem}

        Reasoning:
        {spec.example_reasoning}

        Answer:
        {spec.example_answer}

        Generate genuinely different problems. Do not simply reproduce the same underlying instance
        while changing names, nouns, verbs, or settings.

        Vary both the reasoning structure and the subject matter across the batch. Use a broad mix of
        ordinary situations, organizations, objects, natural phenomena, classifications, and technical
        settings. {spec.diversity_note}

        For each problem, provide a concise but complete reasoning path and a natural final answer that
        states the actual conclusion. The final answer may be a word, phrase, sentence, or short
        explanation depending on what best answers the question. Do not force answers into a yes/no
        format.

        Use whatever reasoning depth is naturally required. Do not add filler and do not omit necessary
        inferences.

        Return ONLY a valid JSON array containing exactly {batch_size} objects.

        Each object must contain exactly these three string fields:

        {{
          "problem": "...",
          "reasoning": "...",
          "answer": "..."
        }}

        Do not include markdown fences, headings, explanations, or any text outside the JSON array.
        """
    ).strip()


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "R0_SKILLS",
    "SKILL_PROMPT_SPECS",
    "SkillPromptSpec",
    "build_generation_prompt",
]
