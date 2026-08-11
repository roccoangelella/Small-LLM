"""Frozen production policy for the Nemotron-ClimbMix token-only corpus.

Everything that defines *what* the production corpus is lives here as plain
constants. The implementations in :mod:`dataset.src`, :mod:`dataset.production`,
and the frozen profile registry in :mod:`dataset.qualification` read this policy
rather than inventing source-selection semantics of their own.

The production path never decodes accepted documents. It reads existing GPT-2
token IDs by deterministic HTTP byte-range access to the pinned Hugging Face
source, keeps records by numeric ``cluster_id`` (the only semantic signal), and
writes little-endian ``uint16`` context-plus-one sequences into immutable
schema-v2 train/validation shards with crash-safe resume and remote durability.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Source identity (frozen)
# ---------------------------------------------------------------------------

DATASET_REPOSITORY = "nvidia/Nemotron-ClimbMix"
# Full immutable commit SHA corresponding to the verified short revision
# ``5eaa64b``.  Resolved from the Hugging Face Hub API on 2026-07-27 and pinned
# so a future change to the repository's ``main`` branch can never silently
# alter the production source.  Do not shorten this.
DATASET_REVISION = "5eaa64b9c0c85b7f56af01d7dffdb0795816b12b"
SHORT_REVISION = "5eaa64b"
# Only the root tokenized JSONL shards are part of the source stream.  The
# repository also contains ``climbmix_small`` (a duplicate small subset),
# ``assets``, and ``nanoGPT``; all subdirectories are excluded by this glob.
SOURCE_DATA_GLOB = "part_*.tokenized.jsonl"

# Tokenizer: we reuse the source's existing GPT-2 byte-level BPE token IDs
# directly and never re-tokenize.  The vocabulary plus the end-of-document
# marker fits in an unsigned 16-bit integer.
TOKENIZER_ID = "gpt2"
TOKENIZER_DESCRIPTION = "climbmix GPT-2 token IDs reused verbatim"
VOCAB_SIZE = 50257

# End-of-document marker appended between accepted documents.  50256 is GPT-2's
# <|endoftext|> id and is already the last vocabulary entry.
EOD_TOKEN_ID = 50256
TOKEN_MIN = 0
TOKEN_MAX = 50256


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# Versioned, fixed seed for all deterministic choices (work-plan shuffle,
# train/validation split).  Changing this changes every downstream artefact.
SELECTION_SEED = "small-llm-climbmix-production-v1"

# Approximately 0.1 % of accepted documents are reserved for validation using a
# stable deterministic hash of the source identity.
VALIDATION_PROBABILITY = 0.001
SPLIT_HASH_VERSION = "small-llm-train-validation-v1"


# ---------------------------------------------------------------------------
# Cluster policy: the only semantic selection mechanism
# ---------------------------------------------------------------------------

# NVIDIA's numeric CLIMB topic map (verified against bounded live samples in
# ``cluster_map_validation.json`` at the repository root).  These strings are
# documentation/manifest material only; selection is the numeric IDs below.
CLUSTER_TOPICS: dict[int, str] = {
    1: "Mathematics, Algorithms, Data Analysis",
    2: "Books, Education, Writing, Literature, Philosophy",
    3: "Environmental Education, History, Architecture, Engineering",
    4: "Education, Teaching, Science, Psychology",
    5: "International Trade, Business, Economics",
    6: "Genetics, Biotechnology, AI, Robotics, Healthcare",
    7: "Chemistry, Taxonomy, Agriculture, Veterinary Science",
    8: "Gaming, Strategy, Fantasy, Virtual Reality",
    9: "Astronomy, Cosmology, Space Exploration, Urban Planning",
    10: "Health, Sleep, Clinical Technology, Fitness",
    11: "Software Development, Programming, Web Development, Databases",
    12: "Technology, Mathematics, Legal, Energy, Industrial Equipment",
    13: "Sports, Cultural Heritage, Competition",
    14: "Music, Instrumental Practice, Theory, Composition",
    15: "Film, Cinema, Horror, Sci-Fi, Comics, Criticism",
    16: "Sustainability, Climate Change, Renewable Energy",
    17: "Cardiovascular Health, Medical Research, Immunology, Cancer",
    18: "Technology, Cybersecurity, Social Media, Cloud Computing",
    19: "Digital Communication, Internet Culture, Psychology",
    20: "Public Safety, Law Enforcement, Political History, Government",
}

# Accept clusters 1-10 and 12-20.  Exclude cluster 11 (software/programming).
# No per-cluster quotas: the source mixture is preserved approximately by
# sampling source byte regions uniformly (see :mod:`dataset.src.workplan`).
ACCEPTED_CLUSTER_IDS: frozenset[int] = frozenset(range(1, 11)) | frozenset(range(12, 21))
EXCLUDED_CLUSTER_IDS: frozenset[int] = frozenset({11})
ALL_CLUSTER_IDS: frozenset[int] = frozenset(range(1, 21))


# ---------------------------------------------------------------------------
# Corpus size targets
# ---------------------------------------------------------------------------

# Selection stops on accepted *source* tokens (tokens originally present in
# accepted documents), NOT on written tokens (source + inserted EOD markers).
TARGET_ACCEPTED_SOURCE_TOKENS = 90_000_000_000
MINIMUM_ACCEPTED_SOURCE_TOKENS = 80_000_000_000
MAXIMUM_ACCEPTED_SOURCE_TOKENS = 100_000_000_000


# ---------------------------------------------------------------------------
# Binary output format
# ---------------------------------------------------------------------------

# Raw GPT-2 token IDs as unsigned 16-bit integers, explicit little-endian, no
# header, no JSON framing, no compression.  Documents are separated by EOD.
INT_TYPE = "uint16"
BYTE_ORDER = "little"


# ---------------------------------------------------------------------------
# Work plan and throughput defaults
# ---------------------------------------------------------------------------

# Each source file is divided into fixed-size logical byte regions; the full set
# of regions is deterministically shuffled (see SELECTION_SEED) and processed in
# that saved order.  256 MiB is small enough to bound single-record memory and
# large enough to keep HTTP request overhead negligible.
REGION_BYTES = 256 * 1024 * 1024

# In-memory write buffers per active writer before a soft flush to the OS.
WRITER_BUFFER_BYTES = 256 * 1024 * 1024

# A durable checkpoint is taken at most every CHECKPOINT_BYTES_THRESHOLD written
# bytes so a crash never loses more than roughly this much work.
CHECKPOINT_BYTES_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1 GiB

# Boundary recovery reads in fixed chunks so a single multi-gigabyte record can
# never be held whole while locating its edges.
BOUNDARY_SCAN_CHUNK_BYTES = 4 * 1024 * 1024
FORWARD_FETCH_CHUNK_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Source HTTP access (Hugging Face resolve endpoint)
# ---------------------------------------------------------------------------

HF_HUB_BASE = "https://huggingface.co"
# Base URL template for resolving a single source file's raw bytes.
RESOLVE_URL_TEMPLATE = (
    HF_HUB_BASE + "/datasets/{repository}/resolve/{revision}/{path}"
)
# Tree/listing API used once to resolve the immutable file list and sizes.
TREE_URL_TEMPLATE = HF_HUB_BASE + "/api/datasets/{repository}/tree/{revision}"

HTTP_TIMEOUT_SECONDS = 60.0
HTTP_MAX_RETRIES = 6
HTTP_BACKOFF_BASE_SECONDS = 1.5
HTTP_BACKOFF_MAX_SECONDS = 30.0
HTTP_USER_AGENT = "small-llm-token-only-corpus/1.0"


# ---------------------------------------------------------------------------
# Disk-space preflight
# ---------------------------------------------------------------------------

# Required free space = target_tokens * 2 bytes * (1 + EOD overhead) * safety
# multiplier.  The multiplier makes the requirement scale with the target: the
# 90 B default requires ~222 GiB (~239 GB), inside the conservative 230-250 GB
# window, while a bounded smoke run needs only a few MB.
DISK_EOD_OVERHEAD_FRACTION = 0.02
DISK_SAFETY_MULTIPLIER = 1.30


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

DATASET_DIR: Path = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR: Path = DATASET_DIR / "output"

PROGRESS_FILENAME = "progress.json"
WORK_PLAN_FILENAME = "work_plan.json"
MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# Artefact schema versions
# ---------------------------------------------------------------------------

PROGRESS_SCHEMA_VERSION = 3
WORK_PLAN_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Dataset attribution
# ---------------------------------------------------------------------------

DATASET_LICENSE = "cc-by-nc-4.0"
ATTRIBUTION = (
    "NVIDIA Nemotron-ClimbMix (https://huggingface.co/datasets/nvidia/"
    "Nemotron-ClimbMix). Built from the official GPT-2-tokenized JSONL shards."
)
# The corpus is cluster-filtered (cluster 11 excluded), not guaranteed code-free.
CORPUS_DESCRIPTION = "cluster-filtered Nemotron-ClimbMix subset, programming cluster excluded"


@dataclass(frozen=True)
class EffectiveConfig:
    """Frozen resolved settings for a single build or verify invocation.

    Production defaults come from module constants; only safe smoke/test
    overrides populate the optional fields.  The frozen policy (source
    revision, seed, cluster policy, format, split) is never overridden here.
    """

    output_dir: Path
    target_accepted_source_tokens: int
    minimum_accepted_source_tokens: int
    maximum_accepted_source_tokens: int
    region_bytes: int
    writer_buffer_bytes: int
    checkpoint_bytes_threshold: int
    max_work_items: int | None
    resume: bool
    strict: bool
    allow_unsafe_low_disk: bool
    reset: bool
    full_scan: bool
    crash_after_written_bytes: int | None