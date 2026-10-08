# Cube-pKa public-release checklist

The local `0.1.0rc1` bundle is prepared for GitHub review. Public release should
proceed only after every remaining unchecked item below is resolved.

- [ ] PI/institution approves the repository owner and public release.
- [x] Cube-pKa code and model weights are licensed under MIT.
- [x] The MolGpKa SMARTS source and upstream MIT notice are included.
- [x] Public package name and intended repository are `Cube-pKa` and
      `https://github.com/min0210/Cube-pKa`.
- [ ] Installation and CPU inference pass in a fresh Python 3.11 environment.
- [ ] GitHub CI runs package installation and `unittest` on every pull request.
- [ ] Model card documents intended use, limitations and validation roles.
- [ ] Versioned release and archival DOI are created.
- [ ] No credentials, absolute home paths, restricted data or cluster logs are
      tracked.

Before the first push, inspect the exact staged content:

```bash
git status --short
git diff --cached --stat
git diff --cached --check
git grep -nE '/home/|api[_-]?key|secret|token|password' -- ':!RELEASE_CHECKLIST.md'
```
