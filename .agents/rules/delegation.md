# Subagent Delegation Rules

## Context Discipline & Worker Delegation

1. **Lead Role**: The main agent acts as the Lead System Architect and Core Engineer.
2. **Worker Role**: Delegate all document reading (> 50 lines), deep log exploration, and shallower mechanical tasks (extracting tables, formatting JSON, scanning multiple files) directly to a `GPT-5.6 Luna` worker agent configured with max thinking.
3. **Zero Pre-Flight Probing**: Do NOT run exploratory verification commands (e.g. `command -v pi`, `pi --help`, `pi --list-models`) to check if models or `pi` exist. They are guaranteed available.
4. **Context Protection**: The worker must return synthesized, high-signal results. Never dump raw document files into the parent conversation.
