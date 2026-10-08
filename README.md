# Cube-pKa

Cube-pKa V4 predicts candidate-site pKa values and reports the strongest acidic
or basic site for a standardized neutral parent structure. The release contains
the frozen task-specific D-MPNN checkpoints selected for the JCIM Application
Note workflow.

> **Release status:** `0.1.0rc1` is a review candidate. The Cube-pKa code and
> model weights are released under the MIT License. The bundled SMARTS snapshot
> retains the upstream MolGpKa MIT notice in `LICENSES/MolGpKa-MIT.txt`.

## Model

| Task | Architecture | Fine-tune best epoch | Checkpoint SHA-256 |
|---|---|---:|---|
| Acid | legacy D-MPNN, h300/d3/f300x1, K=3 | 102 | `3b8c7226057bc67ee9199f0cad62dc7bf8e79a5c5b06870428f70724a260a9d0` |
| Base | legacy D-MPNN, h300/d4/f300x1, K=3 | 98 | `4bc85bf8e06796453ed6c108e3f4e7eb2854a6dcb13baf707f8eadfd08c8a150` |

Both models use 72-dimensional atom features, 14-dimensional bond features and
300-dimensional contextual atom embeddings. The candidate readout concatenates
the candidate-center embedding, a radius-3 distance-weighted local embedding
mean and a task scalar. Acid predictions use the minimum candidate score; base
predictions use the maximum.

The output layer is linear and is not restricted to a predefined pKa range.

## Installation

Python 3.11 is the tested environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The frozen inference environment uses Python 3.11, Chemprop 2.2.3, PyTorch
2.7.1, RDKit 2025.09.6 and NumPy 2.4.1. These versions are pinned because
checkpoint-level numerical reproduction has not yet been established for other
dependency versions.

## Single-molecule prediction

```bash
cube-pka "CC(=O)Oc1ccccc1C(=O)O" --task acid --device cpu
```

Python API:

```python
from cubepka import predict_smiles

result = predict_smiles(
    "CCN(CC)CC(=O)Nc1c(C)cccc1C",
    task="base",
    device="cpu",
)
print(result["predictions"][0]["strongest_pka"])
```

Each successful task result contains every eligible candidate atom, its
zero-based `atom_index`, one-based `atom_number`, candidate family, SMARTS
pattern identifiers, site pKa score and selected-site flag.

## Batch prediction

```bash
cube-pka \
  --input-csv examples/molecules.csv \
  --smiles-column smiles \
  --id-column id \
  --task-column task \
  --device cpu \
  --output predictions.jsonl
```

Invalid SMILES and other row-level failures are retained as JSONL error records
instead of terminating the whole batch.

## Frozen inference contract

1. Parse the input SMILES.
2. Apply RDKit cleanup, select the largest organic fragment, uncharge it and
   remove explicit hydrogens.
3. Detect acid or base candidate atoms with the versioned MolGpKa SMARTS
   snapshot.
4. Exclude a base nitrogen directly bonded to an `S(=O)2` center.
5. Predict each eligible candidate with the corresponding V4 checkpoint.
6. Report the minimum acid score or maximum base score as `strongest_pka`.

The measured pKa and training-site annotation are never inference inputs.

## Validation snapshot

| Evaluation | usable/total | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Development external | 1,414/1,419 | 0.463 | 0.735 | 0.935 |
| SAMPL6 | 31/31 | 0.754 | 1.076 | 0.840 |
| SAMPL7 | 20/20 | 0.493 | 0.601 | 0.935 |
| SAMPL8 | 24/24 | 0.923 | 1.167 | 0.730 |
| All external | 1,489/1,494 | 0.477 | 0.751 | 0.932 |

Organic and Novartis data form the development external validation set.
SAMPL6–8 are post-selection confirmation sets and are reported separately.

## Limitations

- Candidate-site scores are weakly supervised predictions, not experimentally
  measured microscopic pKa values.
- Candidate coverage depends on the frozen SMARTS policy.
- The selected-site atom has not been externally validated as atom-level ground
  truth by the molecule-level benchmark.
- The checkpoints are single-seed models without calibrated uncertainty or an
  applicability-domain threshold.
- SAMPL8 is currently the weakest reported challenge set.

## Integrity and tests

The package checks checkpoint, configuration and SMARTS hashes before
inference. Run the self-contained release tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The reference tests reproduce the frozen V4 predictions for aspirin, ibuprofen,
lidocaine and diphenhydramine to six decimal places.

## Repository scope

This release contains inference code, canonical checkpoints, candidate patterns,
tests and a small example. Training data, raw external benchmark data, Slurm
files, logs and internal filesystem paths are intentionally excluded.

## License and attribution

Cube-pKa code and model weights are available under the [MIT License](LICENSE).
The candidate SMARTS snapshot is reproduced from
[Xundrug/MolGpKa](https://github.com/Xundrug/MolGpKa/blob/master/src/utils/smarts_pattern.tsv),
whose repository is MIT-licensed. Its original notice is preserved in
[`LICENSES/MolGpKa-MIT.txt`](LICENSES/MolGpKa-MIT.txt).
