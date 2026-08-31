# Subagent Delegation Rules

## Context Discipline & Worker Delegation

1. **Lead Role**: The main agent acts as the Lead System Architect and Core Engineer.
2. **Worker Role**: Delegate all document reading (> 50 lines), deep log exploration, and shallower mechanical tasks (extracting tables, formatting JSON, scanning multiple files) to a `GPT-5.6 Luna` worker agent configured with max thinking.
3. **Context Protection**: The worker must return only synthesized, high-signal results (< 300 words). Never dump raw document files into the parent conversation.
