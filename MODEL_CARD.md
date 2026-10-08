# Cube-pKa V4 model card

## Summary

Cube-pKa V4 is a pair of task-specific directed message-passing neural networks
for candidate-site acidic and basic pKa prediction. It returns every eligible
candidate score and identifies the minimum-scoring acid site or maximum-scoring
base site.

## Intended use

- Prioritizing plausible acidic or basic sites in small organic molecules.
- Producing a strongest-site pKa estimate with atom-level traceability.
- Research workflows in which the input is a molecular SMILES and the requested
  task is acid, base or both.

The model is not intended to replace experimental measurement, predict a full
microstate network, or make high-stakes clinical, environmental or regulatory
decisions without independent validation.

## Architecture

- Chemprop directed-bond message passing.
- Atom feature dimension 72; bond feature dimension 14.
- Hidden dimension 300 and dropout 0.1.
- Acid encoder depth 3; base encoder depth 4.
- Candidate readout: center embedding, radius-3 distance-weighted contextual
  embedding mean and a task scalar.
- Linear unrestricted pKa output.

## Training

The models were pretrained with ChEMBL35-derived finite macro-pKa targets and
MolGpKa-selected pseudo-sites, then fine-tuned on 8,895 pKaChu and IUPAC records
with mapped protonation-transition atoms. Acid and base models were trained
separately. Parent-grouped splitting used seed `19980210`; external Organic,
Novartis and SAMPL6–8 identities were excluded from pretraining.

Pseudo-sites and mapped-transition atoms are training annotations. They are not
independently measured atom-resolved pKa labels.

## Candidate policy

Candidate atoms are detected by the frozen SMARTS snapshot included in the
package. Base nitrogen directly bonded to an `S(=O)2` center is excluded. The
candidate atom remains in the molecular graph and can affect neighboring atom
embeddings.

## Evaluation

| Evaluation | usable/total | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Development external | 1,414/1,419 | 0.463 | 0.735 | 0.935 |
| SAMPL6 | 31/31 | 0.754 | 1.076 | 0.840 |
| SAMPL7 | 20/20 | 0.493 | 0.601 | 0.935 |
| SAMPL8 | 24/24 | 0.923 | 1.167 | 0.730 |
| All external | 1,489/1,494 | 0.477 | 0.751 | 0.932 |

Organic and Novartis were inspected during development. SAMPL6–8 are
post-selection confirmation sets. Results come from one training seed.

## Known limitations

- Molecule-level pKa evaluation does not establish atom-level site accuracy.
- Candidate coverage and selected sites depend on hand-defined SMARTS.
- The neutral-parent standardization may not represent the experimentally
  relevant microstate for every molecule.
- No calibrated prediction interval or applicability-domain threshold is
  provided.
- SAMPL8 performance is weaker than SAMPL6 and SAMPL7.
- The model does not explicitly predict coupled or sequential protonation
  transitions.

## Reproducibility

The package verifies exact checkpoint and candidate-pattern SHA-256 hashes at
runtime. `src/cubepka/resources/model_manifest.json` records model
configurations, best epochs, validation site MAE values and training seed.
