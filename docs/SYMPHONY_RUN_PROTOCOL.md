# Symphony Run Protocol

This document defines the repo-local run protocol for Symphony/Jules-style workers executing Tensor Linear issues. Future runnable issues should link to this protocol.

## Handoff Contract

When executing an issue in this repository, follow these rules strictly:

1. **Task Contract**: Treat the Linear issue as the complete and final task contract.
2. **Branch Naming**: Create or use a git branch named from the issue key, e.g. `codex/SYM-25-short-title`.
3. **Execution Notes**: Write local execution notes or use a workpad if the repository has an established place for that (e.g., in `notes/` directory or a designated task file).
4. **Scope Constraint**: Keep one issue to one branch and one Pull Request. Do not bundle multiple issues.
5. **Validation**: Validate your changes with `python3 tools/local_validation.py` before PR handoff, or with the smallest, most direct command that proves the acceptance criteria when a narrower command is explicitly justified.
6. **Documentation of Results**: Record the PR URL, commit SHA, validation command used, and any blockers encountered during execution.
7. **Status Management**: Do not mark the Linear issue as Done until the PR is fully merged or explicitly accepted by a maintainer.

## PR Publishing Contract

Implementation work is not complete until it is visible as a real GitHub PR.

- Commit scoped changes.
- Push the issue branch to `origin`.
- Open a GitHub PR against `main`.
- Comment on the Linear issue with the PR URL, branch name, commit SHA, validation command, and result.
- Do not stop at local commits, task summaries, `make_pr` metadata, or a local-only branch. The PR must be visible to `gh pr view`.
- If pushing or PR creation is blocked, comment on Linear with the exact blocker, branch name, commit SHA, and failed command.

## Support/Stability Fast Path

The current Tensor research lane is support/stability. After a clean checkout install:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v
```

Neural baseline smoke validation stays out of CI, but should be run by workers touching `exp86`:

```bash
python experiments/exp86_support_baselines.py --quick
```

Full pre-handoff validation is:

```bash
python3 tools/local_validation.py
```

## Notes for Agents

- Keep your actions practical and executable.
- Avoid broad implementation changes unless explicitly requested by the issue.
- If you are making only documentation changes, state why no code test was needed in your validation step.
