from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_UNCHARGER = rdMolStandardize.Uncharger()
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()


@dataclass(frozen=True)
class StandardizedMolecule:
    model_smiles: str
    parent_key: str
    connectivity_key: str


def standardize_smiles(smiles: str) -> StandardizedMolecule:
    """Apply the frozen V4 neutral-parent standardization contract."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("invalid_smiles")
    cleaned = rdMolStandardize.Cleanup(Chem.Mol(mol))
    parent = _FRAGMENT_CHOOSER.choose(cleaned)
    parent = _UNCHARGER.uncharge(parent)
    parent = Chem.RemoveHs(parent)
    Chem.SanitizeMol(parent)
    model_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)

    identity = _TAUTOMER_ENUMERATOR.Canonicalize(Chem.Mol(parent))
    Chem.RemoveStereochemistry(identity)
    parent_key = Chem.MolToSmiles(identity, canonical=True, isomericSmiles=False)
    inchi_key = Chem.MolToInchiKey(identity)
    if not inchi_key:
        raise ValueError("inchikey_generation_failed")
    return StandardizedMolecule(
        model_smiles=model_smiles,
        parent_key=parent_key,
        connectivity_key=inchi_key.split("-", 1)[0],
    )
