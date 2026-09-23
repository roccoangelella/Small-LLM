"""One immutable checkpoint in flight; completion is observed by the trainer thread."""
from concurrent.futures import ThreadPoolExecutor


class AsyncPublication:
    def __init__(self, consume):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-upload")
        self._pending = None
        self._consume = consume

    def poll(self, *, wait=False):
        if self._pending is None or (not wait and not self._pending.done()):
            return
        pending, self._pending = self._pending, None
        self._consume(pending.result())

    def submit(self, function, *args, **kwargs):
        # Bounded memory/disk and strict latest/best publication order.
        self.poll(wait=True)
        self._pending = self._executor.submit(function, *args, **kwargs)

    def close(self):
        try:
            self.poll(wait=True)
        finally:
            self._executor.shutdown(wait=True)
