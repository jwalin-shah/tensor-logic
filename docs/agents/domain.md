# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repo root, if it exists.
- `CONTEXT-MAP.md` at the repo root, if it exists.
- `docs/adr/`, if it exists.

If any of these files do not exist, proceed silently. Producer skills create them lazily when terms or decisions actually get resolved.

## File structure

This repo uses a single-context layout:

```text
/
├── CONTEXT.md
├── docs/adr/
└── tensor_logic/
```

## Use the glossary's vocabulary

When output names a domain concept, use the term as defined in `CONTEXT.md`. If the concept is not in the glossary yet, note that it may need to be added during a design conversation.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly rather than silently overriding it.
