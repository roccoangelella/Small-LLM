from __future__ import annotations

from dataset.production import builder as production_builder
from dataset.src.remote import InMemoryDriveStore
from dataset.src.streaming import SourceDocument, StreamCacheConfig, synthetic_test_weights
from dataset.src.workplan import WorkItem, WorkPlan


def stream_config() -> StreamCacheConfig:
    return StreamCacheConfig(
        context_length=8,
        sequences_per_block=2,
        target_shard_bytes=1024,
        reader_workers=2,
        max_in_flight_work_items=2,
        per_cluster_queue_limit=4,
        prepared_block_queue_limit=10,
        prefetch_head_start=0,
        weights=synthetic_test_weights(),
        scheduler_tie_break_seed="test",
    )


def work_plan() -> WorkPlan:
    return WorkPlan(
        2,
        "repo",
        "rev",
        "*",
        "seed",
        100,
        (),
        (WorkItem(0, "part", 0, 100),),
        "plan-hash",
    )


def documents(lengths=(3, 3, 4, 5)):
    return [
        (False, SourceDocument(f"d{i}", 1 + i % 2, tuple(range(n)), 0, i * 10))
        for i, n in enumerate(lengths)
    ]


class ReaderPatchMixin:
    def setUp(self) -> None:
        super().setUp()
        self._original_reader = production_builder.parallel_read_documents

    def tearDown(self) -> None:
        production_builder.parallel_read_documents = self._original_reader
        super().tearDown()

    def use_documents(self, values) -> None:
        production_builder.parallel_read_documents = lambda *args, **kwargs: iter(values)


class CountingDriveStore(InMemoryDriveStore):
    def __init__(self) -> None:
        super().__init__()
        self.uploads = 0
        self.verifies = 0

    def upload_finalized_shard(self, **kwargs):
        self.uploads += 1
        return super().upload_finalized_shard(**kwargs)

    def verify_remote_shard(self, **kwargs):
        self.verifies += 1
        return super().verify_remote_shard(**kwargs)


class FlakyDriveStore(CountingDriveStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def upload_finalized_shard(self, **kwargs):
        if not self.failed_once:
            self.failed_once = True
            raise OSError("transient upload failure")
        return super().upload_finalized_shard(**kwargs)
