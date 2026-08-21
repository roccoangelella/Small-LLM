"""R-SFT reasoning source adapters.

Each source module should expose a stable SOURCE_NAME and a prepare(...) function
that emits fit.jsonl, candidates.jsonl, and manifest.json using dataset/common.py.
Source-specific parsing/filtering belongs here; context repair and final assembly
do not.
"""
