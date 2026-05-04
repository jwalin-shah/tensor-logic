# Symphony Run Protocol

This document defines the repo-local run protocol for Symphony/Jules-style workers executing Tensor Linear issues. Future runnable issues should link to this protocol.

## Handoff Contract

When executing an issue in this repository, follow these rules strictly:

1. **Task Contract**: Treat the Linear issue as the complete and final task contract.
2. **Branch Naming**: Create or use a git branch named exactly after the issue key (e.g., `SYM-25`).
3. **Execution Notes**: Write local execution notes or use a workpad if the repository has an established place for that (e.g., in `notes/` directory or a designated task file).
4. **Scope Constraint**: Keep one issue to one branch and one Pull Request. Do not bundle multiple issues.
5. **Validation**: Validate your changes with the smallest, most direct command that proves the acceptance criteria have been met.
6. **Documentation of Results**: Record the PR URL, commit SHA, validation command used, and any blockers encountered during execution.
7. **Status Management**: Do not mark the Linear issue as Done until the PR is fully merged or explicitly accepted by a maintainer.

## Notes for Agents

- Keep your actions practical and executable.
- Avoid broad implementation changes unless explicitly requested by the issue.
- If you are making only documentation changes, state why no code test was needed in your validation step.
