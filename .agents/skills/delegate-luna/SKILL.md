---
name: delegate-luna
description: >-
  Delegate document reading, large log exploration, multi-file searches, and shallow extraction tasks to GPT-5.6 Luna with max thinking.
---

# Luna Task & Context Delegation

Use this skill when handling bulky documentation, long reference contracts, incident logs, or shallow extraction tasks.

## Protocol

1. **Trigger Condition**:
   - Ingesting files > 50 lines (runbooks, status, benchmarks).
   - Multi-file regex or codebase searches across directories.
   - Shallow extraction (table parsing, manifest verification, JSON formatting).

2. **Worker Invocation (Zero Pre-Flight Probing)**:
   - Model target: `GPT-5.6 Luna` with maximum thinking / reasoning budget.
   - Do NOT run verification commands (`command -v pi`, `pi --help`, `pi --list-models`) to check availability; dispatch directly.
   - Instruction: Perform deep reading and return synthesized high-signal findings or structured tables. Do NOT append arbitrary word-count limits (e.g. "under 220 words").
   - Output constraint: Never dump raw file contents to the parent session.

3. **Parent Synthesis**:
   - The primary agent consumes the synthesized output to execute engineering decisions, keeping the primary context window lean and fast.
