# Contributing to Φ

Φ is a learning repo first. Contributions that make it easier for the *next* student are the most valuable kind.

## Ground rules
1. **Reproducible or it didn't happen.** Every result must trace to a committed config + a logged command + a fixed seed. Put numbers in [`experiments/`](experiments/) with the command that produced them.
2. **Weights & datasets go on the Hugging Face Hub, never in git.** Commit the *card* (a small markdown pointer + metadata), not the bytes. See [`datasets/`](datasets/) and [`models/`](models/).
3. **Docs are part of the change.** If you add a capability, add/adjust the doc page for it.
4. **Small, focused PRs.** One change per PR; fill in the PR template.

## Workflow
```bash
git checkout -b feat/<short-name>
make lint            # ruff + mypy
make test            # unit + smoke
# ... commit, push, open a PR against main ...
```

## Good first issues
Look for the `good-first-issue` label. Typical ones: document a gotcha you hit, add a task card, add a training config, improve a doc page, write a small dataset-QA check.

## Coding standards
- Python ≥3.12, formatted with **ruff**, typed where practical (**mypy**).
- Keep `src/phi/` a *thin* layer over LeRobot — prefer calling upstream over copying it.
- Name experiments `YYYY-MM-DD_<slug>`.

## Reporting
- **Bugs / hardware issues / experiments** → open an issue with the matching template.
- **Security or safety concern with the arm** → flag it in the issue title with `[SAFETY]`.
