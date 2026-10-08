# Release-candidate verification

Verification date: 2026-10-08

## Passed locally

- Python source and tests compile under Python 3.11.15.
- Five self-contained unit tests pass.
- Acid and base checkpoint SHA-256 values match `model_manifest.json`.
- The SMARTS snapshot SHA-256 matches `model_manifest.json`.
- Both checkpoints load strictly into the packaged model definitions.
- Aspirin, ibuprofen, lidocaine and diphenhydramine predictions match the
  canonical V4 inference results to six decimal places.
- Candidate indices matched the original V4 candidate generator for 20 probes
  spanning 10 molecules and both acid/base tasks.
- A wheel was built and installed into an isolated target directory without
  project-source paths; CPU CLI inference succeeded.
- The wheel contains the Cube-pKa MIT license, third-party notice and the exact
  upstream MolGpKa MIT notice.
- CSV batch prediction with a per-row task column produced four successful
  JSONL records.
- No checkpoint or bundled data file exceeds GitHub's 100 MB file limit.
- No credentials or local absolute home paths were found in tracked source
  candidates.

## Not yet verified

- GitHub Actions has not run because no remote repository is configured.
- A completely fresh environment dependency installation has not been tested.
- CUDA inference has not been tested for this release candidate.
- Public redistribution and license conditions remain unresolved; see
  `RELEASE_CHECKLIST.md`.
