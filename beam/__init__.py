"""Project-local namespace bridge for Beam remote handler imports.

Beam derives remote handler module names from source paths. The canonical
launcher therefore becomes ``beam.launch`` when invoked from the repository
root. Remote workers prepend the synced checkout to ``sys.path``, so this
package intentionally owns that namespace and forwards the SDK objects used by
the launcher to Beam's underlying ``beta9`` package.
"""
from __future__ import annotations

from beta9 import Image, Volume, function

__all__ = ["Image", "Volume", "function"]
