from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from rdkit import Chem

Task = Literal["acid", "base"]


@dataclass(frozen=True)
class IonizationPattern:
    pattern_id: int
    query: Chem.Mol
    center_positions: tuple[int, ...]
    task: Task


@dataclass(frozen=True)
class CandidateSite:
    atom_index: int
    atom_symbol: str
    family: str
    pattern_ids: tuple[int, ...]


def load_patterns(path: str | Path) -> tuple[IonizationPattern, ...]:
    patterns: list[IonizationPattern] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle, delimiter="\t"):
            row = {key.strip(): value.strip() for key, value in raw_row.items()}
            query = Chem.MolFromSmarts(row["SMARTS"])
            if query is None:
                raise ValueError(f"invalid SMARTS pattern {row['Substructure']}")
            task: Task = "acid" if row["Acid_or_base"].upper() == "A" else "base"
            patterns.append(
                IonizationPattern(
                    pattern_id=int(row["Substructure"]),
                    query=query,
                    center_positions=tuple(int(value) for value in row["Index"].split(",")),
                    task=task,
                )
            )
    if not patterns:
        raise ValueError("candidate pattern file is empty")
    return tuple(patterns)


def is_sulfonamide_nitrogen(mol: Chem.Mol, atom_index: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_index)
    if atom.GetAtomicNum() != 7:
        return False
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() != 16:
            continue
        double_oxygen_count = sum(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(neighbor).GetAtomicNum() == 8
            for bond in neighbor.GetBonds()
        )
        if double_oxygen_count >= 2:
            return True
    return False


def _acid_hydrogen_to_heavy(mol_h: Chem.Mol, hydrogen_index: int) -> int:
    atom = mol_h.GetAtomWithIdx(hydrogen_index)
    if atom.GetAtomicNum() != 1:
        raise ValueError(f"acid candidate {hydrogen_index} is not hydrogen")
    heavy = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1]
    if len(heavy) != 1:
        raise ValueError(f"acid candidate {hydrogen_index} has {len(heavy)} heavy neighbors")
    return heavy[0]


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetAtomicNum() in {8, 16}
        for bond in atom.GetBonds()
    )


def classify_site(mol: Chem.Mol, atom_index: int, task: Task) -> str:
    atom = mol.GetAtomWithIdx(atom_index)
    neighbors = list(atom.GetNeighbors())
    if task == "acid":
        if atom.GetAtomicNum() == 8:
            if any(_is_carbonyl_carbon(neighbor) for neighbor in neighbors):
                return "carboxylic_acid"
            if any(neighbor.GetIsAromatic() for neighbor in neighbors):
                return "phenol"
            if any(neighbor.GetAtomicNum() == 16 for neighbor in neighbors):
                return "sulfonic_acid"
            if any(neighbor.GetAtomicNum() == 15 for neighbor in neighbors):
                return "phosphoric_acid"
            return "oxygen_acid"
        if atom.GetAtomicNum() == 7:
            return "nitrogen_acid"
        return "other_acid"
    if atom.GetAtomicNum() == 7:
        if atom.GetIsAromatic():
            return "aromatic_nitrogen"
        if any(_is_carbonyl_carbon(neighbor) for neighbor in neighbors):
            return "amide_like_nitrogen"
        if any(neighbor.GetIsAromatic() for neighbor in neighbors):
            return "aniline_like_nitrogen"
        hydrogens = atom.GetTotalNumHs()
        return {2: "primary_amine", 1: "secondary_amine"}.get(hydrogens, "tertiary_amine")
    if atom.GetAtomicNum() == 8:
        return "oxygen_base"
    if atom.GetAtomicNum() == 16:
        return "sulfur_base"
    return "other_base"


def find_candidate_sites(
    mol: Chem.Mol,
    task: Task,
    patterns: Iterable[IonizationPattern],
) -> tuple[CandidateSite, ...]:
    """Reproduce the frozen V4 MolGpKa SMARTS candidate policy."""
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise ValueError("candidate detection expects a heavy-atom molecule")
    mol_h = Chem.AddHs(mol)
    matched: dict[int, set[int]] = {}
    for pattern in patterns:
        if pattern.task != task:
            continue
        for match in mol_h.GetSubstructMatches(pattern.query, uniquify=True):
            for position in pattern.center_positions:
                if position >= len(match):
                    raise ValueError(
                        f"pattern {pattern.pattern_id} center {position} exceeds its match"
                    )
                atom_index = match[position]
                if task == "acid":
                    atom_index = _acid_hydrogen_to_heavy(mol_h, atom_index)
                elif is_sulfonamide_nitrogen(mol_h, atom_index):
                    continue
                if atom_index >= mol.GetNumAtoms():
                    raise ValueError(f"pattern {pattern.pattern_id} selected an explicit hydrogen")
                matched.setdefault(atom_index, set()).add(pattern.pattern_id)
    return tuple(
        CandidateSite(
            atom_index=atom_index,
            atom_symbol=mol.GetAtomWithIdx(atom_index).GetSymbol(),
            family=classify_site(mol, atom_index, task),
            pattern_ids=tuple(sorted(pattern_ids)),
        )
        for atom_index, pattern_ids in sorted(matched.items())
    )
