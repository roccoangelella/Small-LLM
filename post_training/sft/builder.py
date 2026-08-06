"""End-to-end deterministic S0 dataset construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping

from .config import SFTDataConfig
from .filters import S0RecordFilter
from .mixture import BufferedShuffle, TargetTokenMixer, build_atomic_blocks
from .schema import ConversationRecord, TokenizedSFTRecord
from .storage import SFTDatasetWriter
from .template import GPT2ChatTemplate, TokenEncoder


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(slots=True)
class SourceAudit:
    seen: int = 0
    accepted: int = 0
    rejected: Counter[str] = None  # type: ignore[assignment]
    serialized_tokens: int = 0
    target_tokens: int = 0

    def __post_init__(self) -> None:
        if self.rejected is None:
            self.rejected = Counter()

    def as_dict(self) -> dict[str, object]:
        return {
            "seen": self.seen,
            "accepted": self.accepted,
            "rejected": dict(sorted(self.rejected.items())),
            "serialized_tokens": self.serialized_tokens,
            "loss_bearing_target_tokens": self.target_tokens,
        }


class SFTDatasetBuilder:
    """Build a finite immutable SFT stream from independently replaceable sources."""

    def __init__(
        self,
        config: SFTDataConfig,
        *,
        encoder: TokenEncoder,
        record_filter: S0RecordFilter | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.record_filter = record_filter or S0RecordFilter(
            allowed_sources=frozenset(config.instruction_source_shares)
        )
        self.template = GPT2ChatTemplate(
            eos_token_id=config.eos_token_id,
            maximum_context_tokens=config.context_length,
            maximum_assistant_tokens=config.maximum_assistant_tokens,
        )
        self.audit = {
            source: SourceAudit()
            for source in config.instruction_source_shares
        }

    def _fit_and_tokenize(
        self,
        record: ConversationRecord,
    ) -> TokenizedSFTRecord | None:
        """Drop oldest complete dialogue pairs until the target fits."""

        messages = list(record.messages)
        while True:
            candidate = ConversationRecord(
                conversation_id=record.conversation_id,
                source=record.source,
                messages=tuple(messages),
                split=record.split,
                metadata=record.metadata,
            )
            try:
                return self.template.encode_conversation(candidate, self.encoder)
            except ValueError as error:
                message = str(error)
                if "assistant response exceeds" in message:
                    self.audit[record.source].rejected["assistant_too_long"] += 1
                    return None
                if "serialized conversation exceeds" not in message:
                    raise
                start = 1 if messages[0].role == "system" else 0
                dialogue = messages[start:]
                if len(dialogue) <= 2:
                    self.audit[record.source].rejected["conversation_too_long"] += 1
                    return None
                del dialogue[:2]
                messages = messages[:start] + dialogue

    def _instruction_records(
        self,
        source: str,
        records: Iterable[ConversationRecord],
    ) -> Iterable[TokenizedSFTRecord]:
        audit = self.audit[source]
        for record in records:
            audit.seen += 1
            if record.source != source:
                raise ValueError(
                    f"instruction stream {source!r} yielded {record.source!r}"
                )
            decision = self.record_filter.evaluate(record)
            if not decision.accepted:
                audit.rejected[decision.reason] += 1
                continue
            tokenized = self._fit_and_tokenize(record)
            if tokenized is None:
                continue
            audit.accepted += 1
            audit.serialized_tokens += tokenized.serialized_token_count
            audit.target_tokens += tokenized.target_token_count
            yield tokenized

    def build(
        self,
        *,
        instruction_sources: Mapping[str, Iterable[ConversationRecord]],
        replay_source: Iterable[TokenizedSFTRecord],
        output_dir: Path | str,
    ) -> dict[str, object]:
        configured = set(self.config.instruction_source_shares)
        if set(instruction_sources) != configured:
            raise ValueError(
                "instruction_sources must exactly match configured source shares"
            )

        mixed_sources: dict[str, Iterable[TokenizedSFTRecord]] = {}
        for source, records in instruction_sources.items():
            mixed_sources[source] = BufferedShuffle(
                self._instruction_records(source, records),
                seed=_stable_seed(self.config.seed, source),
                buffer_size=self.config.shuffle_buffer_records,
            )
        mixed_sources["climbmix-replay"] = BufferedShuffle(
            replay_source,
            seed=_stable_seed(self.config.seed, "climbmix-replay"),
            buffer_size=self.config.shuffle_buffer_records,
        )

        mixed = TargetTokenMixer(
            mixed_sources,
            self.config.complete_source_shares,
            seed=_stable_seed(self.config.seed, "source-mixer"),
            target_loss_tokens=self.config.target_loss_tokens,
        )
        blocks = build_atomic_blocks(
            mixed,
            target_tokens_per_block=self.config.optimizer_target_tokens,
        )
        writer = SFTDatasetWriter(output_dir, self.config)
        manifest = writer.write(blocks)
        total = int(manifest["totals"]["loss_bearing_target_tokens"])  # type: ignore[index]
        maximum_expected_shortfall = self.config.context_length
        if total < self.config.target_loss_tokens - maximum_expected_shortfall:
            shutil.rmtree(Path(output_dir), ignore_errors=True)
            raise RuntimeError(
                "SFT sources were exhausted before the configured target-token horizon"
            )

        report_without_hash = {
            "schema": "small-llm-sft-build-report",
            "manifest_identity": manifest["manifest_sha256"],
            "source_audit": {
                source: audit.as_dict()
                for source, audit in sorted(self.audit.items())
            },
            "actual_source_target_tokens": manifest["totals"]["source_target_tokens"],  # type: ignore[index]
        }
        encoded = json.dumps(
            report_without_hash,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        report = {
            **report_without_hash,
            "report_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        report_path = Path(output_dir) / "build-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"manifest": manifest, "report": report}


__all__ = ["SFTDatasetBuilder", "SourceAudit"]
