# Pinned third-party dependencies

## tau2-bench

- Upstream: https://github.com/sierra-research/tau2-bench
- Pinned commit: `a2c024725189473d2d7cea3a5cfdbcc67478e41f`
- Commit date: 2026-08-18
- **Must equal policygate's pin** (`third_party/PIN.md` in the PolicyGate repo at
  the tag this project depends on). Bumping it affects both projects.

The working copy under `third_party/tau2-bench/` is a detached-HEAD checkout of
that commit and is **git-ignored**. Reproduce it with:

```bash
git clone https://github.com/sierra-research/tau2-bench.git third_party/tau2-bench
git -C third_party/tau2-bench checkout a2c024725189473d2d7cea3a5cfdbcc67478e41f
```
