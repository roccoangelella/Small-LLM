"""All tunable settings for the Nemotron-ClimbMix curation pipeline.

The pipeline deliberately keeps policy in this file and its implementation in
focused modules under ``dataset/src/``. Change the settings here after inspecting
the sample and review artifacts; do not edit a generated selection plan by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Input and reproducibility
# ---------------------------------------------------------------------------

DATASET_REPOSITORY = "nvidia/Nemotron-ClimbMix"
# Pin this to an immutable Hugging Face commit before a production run.  The
# current short revision is intentionally explicit instead of silently using
# a future version of ``main``.
DATASET_REVISION = "5eaa64b"
DATASET_SPLIT = "train"
# The repository also has ``climbmix_small``.  It is a duplicate small subset,
# so only the root tokenized JSONL files are part of the source stream.
DATASET_DATA_FILES_GLOB = "part_*.tokenized.jsonl"
TOKENIZER_ENCODING = "gpt2"
RANDOM_SEED = "small-llm-climbmix-v1"


# ---------------------------------------------------------------------------
# Paths and file layout
# ---------------------------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = DATASET_DIR / "artifacts"
SAMPLES_PATH = ARTIFACTS_DIR / "cluster_samples.jsonl"
INVENTORY_PATH = ARTIFACTS_DIR / "cluster_inventory.json"
SELECTION_PLAN_PATH = ARTIFACTS_DIR / "selection_plan.json"
MANUAL_REVIEW_PATH = ARTIFACTS_DIR / "manual_review.md"
LLM_REVIEW_DIR = ARTIFACTS_DIR / "llm_reviews"
REVIEW_SUMMARY_PATH = ARTIFACTS_DIR / "cluster_review_summary.json"

OUTPUT_DIR = DATASET_DIR / "output"
SELECTION_STATE_PATH = OUTPUT_DIR / "selection_state.json"
SELECTION_MANIFEST_PATH = OUTPUT_DIR / "selection_manifest.json"
AUDIT_DIR = ARTIFACTS_DIR / "audit"


# ---------------------------------------------------------------------------
# Cluster policy and quota.  These are the production defaults to validate
# against the sample review.  Percentages must total 100 across accepted IDs.
# ---------------------------------------------------------------------------

KEEP = "keep"
KEEP_WITHOUT_CODE = "keep_without_code"
EXCLUDE = "exclude_or_downweight"


@dataclass(frozen=True)
class ClusterPolicy:
    """A reviewable per-cluster decision and the target share of selected tokens."""

    expected_topic: str
    decision: str
    quota_percent: int
    rationale: str


CLUSTER_POLICIES: dict[int, ClusterPolicy] = {
    1: ClusterPolicy("Environment, public health, policy development, medical innovation", KEEP, 5, "Useful scientific and policy prose."),
    2: ClusterPolicy("Technology, neurophysiology, health and safety, innovative research, rehabilitation", KEEP, 6, "Useful technical and health prose."),
    3: ClusterPolicy("Restoration efforts, climate, ecosystems, community engagement", KEEP, 5, "Environment and civic knowledge."),
    4: ClusterPolicy("Diagnostics, diseases, prevention and control", KEEP, 5, "Medical and public-health knowledge."),
    5: ClusterPolicy("Vehicles, ecology, communities, conservation", KEEP, 4, "General and environmental knowledge."),
    6: ClusterPolicy("Energy, science, materials, nanostructures, quantum computing", KEEP, 6, "Scientific and technical prose; code filter still applies."),
    7: ClusterPolicy("Physics, particle accelerators, materials, architecture, systems", KEEP, 6, "Scientific and systems prose; code filter still applies."),
    8: ClusterPolicy("Biology, genetics, astronomy, climate science", KEEP, 6, "Core scientific knowledge."),
    9: ClusterPolicy("Earth science, space science, scientific collaboration", KEEP, 6, "Core scientific knowledge."),
    10: ClusterPolicy("Health, symptoms, treatments, therapy, disorders and medical conditions", KEEP, 5, "Medical prose."),
    11: ClusterPolicy("Communication, biographies, history, society and policy", KEEP, 5, "General knowledge and humanities."),
    12: ClusterPolicy("Culture, education, sustainability, community, public health, crime and economics", KEEP, 5, "General knowledge and social science."),
    13: ClusterPolicy("Arts, literature, education and history", KEEP, 4, "Humanities balance."),
    14: ClusterPolicy("Geography, government, organizations, religion, agriculture, economics and civilizations", KEEP, 4, "General knowledge balance."),
    15: ClusterPolicy("Science, technology, education, engineering and collaboration", KEEP_WITHOUT_CODE, 7, "Retain explanatory prose while strictly excluding code-heavy material."),
    16: ClusterPolicy("Science, health, minerals, population, agriculture, vaccination, welfare and management", KEEP, 6, "Applied science and general knowledge."),
    17: ClusterPolicy("Role-playing, problem solving, mathematics and algorithms", KEEP_WITHOUT_CODE, 5, "Keep mathematics and reasoning prose, but no implementation-heavy documents."),
    18: ClusterPolicy("Revolutions, parliament, efficiency, communication and animal behaviour", KEEP, 5, "History, society, and general knowledge."),
    19: ClusterPolicy("History, culture, economics, energy, markets and policy", KEEP, 5, "Humanities, economics, and policy balance."),
    20: ClusterPolicy("Python and programming code", EXCLUDE, 0, "Explicit programming cluster; excluded at cluster level."),
}

ACCEPTED_DECISIONS = frozenset({KEEP, KEEP_WITHOUT_CODE})


# ---------------------------------------------------------------------------
# Corpus size, planning and deterministic sampling
# ---------------------------------------------------------------------------

# 90B GPT-2 tokens is deliberately in the requested 80--100B range.  Text
# bytes are measured after UTF-8 decoding, i.e. before JSON framing.
TARGET_TOKENS = 90_000_000_000
MINIMUM_TOKENS = 80_000_000_000
MAXIMUM_TOKENS = 100_000_000_000
TARGET_TEXT_BYTES = 400_000_000_000
MAXIMUM_TEXT_BYTES = 425_000_000_000

# A first stream creates the eligible-token inventory.  Planning then turns the
# configured quotas into stable hash acceptance rates.  The 5% headroom lets
# token-sized documents fill their quotas without materially changing balance.
PLANNING_OVERSUBSCRIPTION = 1.05
REQUIRE_SELECTION_PLAN = True
MINIMUM_CLUSTER_AVAILABILITY_RATIO = 1.00
# Source documents are indivisible, so allow a negligible final-document
# overshoot instead of requiring each token quota to be an exact integer sum.
MAX_CLUSTER_QUOTA_OVERSHOOT_TOKENS = 500_000

# Selection output is ordinary UTF-8 JSONL, intentionally uncompressed so its
# size is transparent and every document is easy to inspect.
OUTPUT_SHARD_MAX_BYTES = 2_000_000_000
CHECKPOINT_EVERY_DOCUMENTS = 1_000
PROGRESS_EVERY_DOCUMENTS = 100_000


# ---------------------------------------------------------------------------
# Sampling, LLM review, and human spot-check artifacts
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS_PER_CLUSTER = 50
SAMPLE_TEXT_CHARACTERS = 6_000
MANUAL_EXCERPT_CHARACTERS = 1_500
MANUAL_EXCERPTS_PER_CLUSTER = 3

LLM_REVIEW_BATCH_SIZE = 10
LLM_REVIEW_TEXT_CHARACTERS = 2_000
# Some routed models spend output tokens before emitting their JSON. Keep enough
# room for the fixed review object instead of treating a truncated response as a
# valid review.
LLM_REVIEW_MAX_TOKENS = 2_000
LLM_REVIEW_TEMPERATURE = 0.0
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_DELAY_SECONDS = 3.0
LLM_REVIEW_SCHEMA_VERSION = 1

# The pipeline uses the same local GemRouter configuration as Pi.  A supplied
# environment variable wins; otherwise the existing Pi auth file is read
# locally.  No credential is written to artifacts, logs, or this repository.
GEMROUTER_BASE_URL = "https://gemr.84-8-255-231.nip.io/v1"
GEMROUTER_MODEL = "gemini-3.6-flash"
GEMROUTER_API_KEY_ENV = "GEMROUTER_API_KEY"
GEMROUTER_PI_AUTH_PATH = Path.home() / ".pi" / "agent" / "auth.json"
GEMROUTER_AUTH_HEADER = "Authorization"
GEMROUTER_TIMEOUT_SECONDS = 120


# ---------------------------------------------------------------------------
# Deterministic quality filters.  They are intentionally conservative: cluster
# IDs carry most of the curation signal, while these rules remove documents
# which are plainly source/code/API dumps or non-English fragments.
# ---------------------------------------------------------------------------

MIN_DOCUMENT_CHARACTERS = 200
REQUIRE_LIKELY_ENGLISH = True
MIN_ASCII_LETTER_RATIO = 0.55
MIN_ENGLISH_MARKER_RATIO = 0.008
MIN_ENGLISH_MARKER_HITS = 2

MAX_CODE_LINE_FRACTION = 0.35
MAX_FENCED_CODE_FRACTION = 0.25
MAX_CODE_SYMBOL_FRACTION = 0.13
MIN_CODE_LINES_FOR_REJECTION = 4
MIN_LINES_FOR_CODE_FRACTION = 6
MIN_API_MARKERS_FOR_REJECTION = 5
MIN_REPOSITORY_PATHS_FOR_REJECTION = 5

CODE_FILE_EXTENSIONS = (
    "py", "pyi", "ipynb", "js", "jsx", "ts", "tsx", "java", "c", "h", "cc",
    "cpp", "cxx", "cs", "go", "rs", "rb", "php", "swift", "kt", "kts", "scala",
    "sh", "bash", "zsh", "ps1", "sql", "r", "lua", "pl", "dart", "vue", "html",
    "css", "scss", "xml", "yaml", "yml", "toml", "ini", "gradle",
)


# ---------------------------------------------------------------------------
# Final audit.  This is a fresh deterministic sample of selected output, not a
# reuse of the pre-selection sample.
# ---------------------------------------------------------------------------

AUDIT_DOCUMENTS_PER_CLUSTER = 100
AUDIT_TEXT_CHARACTERS = 2_000
AUDIT_LLM_BATCH_SIZE = 20
AUDIT_LLM_MAX_TOKENS = 2_000
AUDIT_USE_LLM = True
MAX_AUDIT_CODE_DOMINATED_FRACTION = 0.002
MIN_AUDIT_ENGLISH_FRACTION = 0.985


def validate_config() -> None:
    """Fail early when a policy or corpus-size setting is internally invalid."""

    expected_ids = set(range(1, 21))
    if set(CLUSTER_POLICIES) != expected_ids:
        raise ValueError("CLUSTER_POLICIES must define exactly clusters 1 through 20")
    allowed = {KEEP, KEEP_WITHOUT_CODE, EXCLUDE}
    if any(policy.decision not in allowed for policy in CLUSTER_POLICIES.values()):
        raise ValueError("Cluster policy has an unknown decision")
    quota_total = sum(
        policy.quota_percent
        for policy in CLUSTER_POLICIES.values()
        if policy.decision in ACCEPTED_DECISIONS
    )
    if quota_total != 100:
        raise ValueError(f"Accepted cluster quotas must total 100, got {quota_total}")
    if any(
        policy.quota_percent != 0
        for policy in CLUSTER_POLICIES.values()
        if policy.decision == EXCLUDE
    ):
        raise ValueError("Excluded clusters must have zero quota")
    if not (MINIMUM_TOKENS <= TARGET_TOKENS <= MAXIMUM_TOKENS):
        raise ValueError("TARGET_TOKENS must lie within the requested token range")
    if TARGET_TEXT_BYTES > MAXIMUM_TEXT_BYTES:
        raise ValueError("TARGET_TEXT_BYTES cannot exceed MAXIMUM_TEXT_BYTES")
