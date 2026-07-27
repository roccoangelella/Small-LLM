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
# NVIDIA's numeric ClimbLab topic table, used for the matching ClimbMix IDs.
CLUSTER_TOPIC_SOURCE_URL = "https://research.nvidia.com/labs/lpr/climb/"


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
# Cluster policy and quota. Topics use NVIDIA's published CLIMB table for these
# numeric IDs, verified against bounded live samples in
# ``cluster_map_validation.json``. Individual documents can still sit near a
# cluster boundary, so the 50-document review is the final policy check.
# Percentages must total 100 across accepted IDs.
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
    1: ClusterPolicy("Mathematics, Algorithms, Programming, Software Development, Data Analysis", KEEP_WITHOUT_CODE, 3, "Retain mathematics and explanatory material, not implementations or repositories."),
    2: ClusterPolicy("Books, Education, Writing, Literature, AI Ethics, History, Philosophy", KEEP, 6, "Humanities and education balance."),
    3: ClusterPolicy("Environmental Education, History, Architecture, Engineering, Classical Music", KEEP, 6, "Broad factual and cultural prose."),
    4: ClusterPolicy("Education, Teaching, Science, Engineering, Psychology, Special Education", KEEP, 7, "Useful educational and scientific prose."),
    5: ClusterPolicy("International Trade, Business, Economics, AI Consulting, Ethical Decision Making", KEEP, 5, "Economics, institutions, and applied reasoning."),
    6: ClusterPolicy("Genetics, Biotechnology, AI, Robotics, Aging, Healthcare, Industrial Automation", KEEP_WITHOUT_CODE, 6, "Keep scientific explanation while removing implementation-heavy material."),
    7: ClusterPolicy("Chemistry, Insects, Taxonomy, Agriculture, Gardening, Veterinary Science", KEEP, 5, "Natural science and practical knowledge."),
    8: ClusterPolicy("Gaming, Role-Playing, Board Games, Video Games, Strategy, Fantasy, Virtual Reality", KEEP, 4, "Keep useful narrative, strategy, and cultural prose; audit quality."),
    9: ClusterPolicy("Astronomy, Cosmology, Astrophysics, Space Exploration, Urban Planning", KEEP, 4, "Scientific and factual prose."),
    10: ClusterPolicy("Health, Sleep, Clinical Technology, Healthcare, Fitness, Addiction, Early Childhood Education", KEEP, 7, "Health and public-interest knowledge."),
    11: ClusterPolicy("Software Development, Programming, Web Development, JavaScript, Databases", KEEP_WITHOUT_CODE, 2, "Retain natural-language technical explanation only; source and API material must go."),
    12: ClusterPolicy("Technology, Mathematics, Legal Content, Human Rights, Energy Efficiency, Industrial Equipment", KEEP_WITHOUT_CODE, 7, "Keep explanatory technical and civic prose, not code-heavy pages."),
    13: ClusterPolicy("Sports, Cricket, Soccer, Tennis, Basketball, Cultural Heritage, Competition", KEEP, 3, "Cultural and general-knowledge balance."),
    14: ClusterPolicy("Music, Instrumental Practice, Guitar, Jazz, Singing, Composition, Music Theory", KEEP, 3, "Arts and music education."),
    15: ClusterPolicy("Film, Cinema, Horror, Sci-Fi, Comics, Literature, Criticism, Philosophy", KEEP, 4, "Arts, criticism, and narrative balance."),
    16: ClusterPolicy("Sustainability, Climate Change, Renewable Energy, Environmental Conservation", KEEP, 8, "Core environmental and scientific knowledge."),
    17: ClusterPolicy("Cardiovascular Health, Medical Research, Immunology, Cancer Prevention, Drug Therapy", KEEP, 8, "Medical and biomedical knowledge."),
    18: ClusterPolicy("Technology, Cybersecurity, Social Media, Privacy, Artificial Intelligence, Cloud Computing", KEEP_WITHOUT_CODE, 4, "Keep technical prose while excluding implementation, logs, and dumps."),
    19: ClusterPolicy("Social Media, Digital Communication, Internet Culture, Misinformation, Psychology", KEEP, 4, "Modern society and communication knowledge."),
    20: ClusterPolicy("Public Safety, Law Enforcement, Political History, Social Justice, Government", KEEP, 4, "Civic and historical prose; this is not the programming cluster."),
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
