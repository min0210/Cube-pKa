from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem


@dataclass(frozen=True)
class SiteRecord:
    record_id: str
    smiles: str
    task: str
    candidates: tuple[int, ...]


@dataclass
class SiteBatch:
    bmg: BatchMolGraph
    candidate_atom_index: torch.Tensor
    candidate_task: torch.Tensor
    candidate_ptr: torch.Tensor
    subgraph_atom_index: torch.Tensor
    subgraph_candidate_index: torch.Tensor
    subgraph_distance: torch.Tensor
    record_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> "SiteBatch":
        self.bmg.to(device)
        for name in (
            "candidate_atom_index",
            "candidate_task",
            "candidate_ptr",
            "subgraph_atom_index",
            "subgraph_candidate_index",
            "subgraph_distance",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self


class SiteBatchCollator:
    def __init__(self, shell_count: int = 3) -> None:
        if shell_count < 1:
            raise ValueError("shell_count must be positive")
        self.shell_count = shell_count
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

    def __call__(self, records: Sequence[SiteRecord]) -> SiteBatch:
        if not records:
            raise ValueError("cannot collate an empty batch")
        mols: list[Chem.Mol] = []
        mol_graphs = []
        atom_offsets: list[int] = []
        atom_offset = 0
        for record in records:
            if record.task not in {"acid", "base"}:
                raise ValueError(f"invalid task: {record.task}")
            if not record.candidates or len(set(record.candidates)) != len(record.candidates):
                raise ValueError(f"invalid candidates: {record.record_id}")
            mol = Chem.MolFromSmiles(record.smiles)
            if mol is None:
                raise ValueError(f"invalid SMILES: {record.record_id}")
            if min(record.candidates) < 0 or max(record.candidates) >= mol.GetNumAtoms():
                raise ValueError(f"candidate outside molecule: {record.record_id}")
            mols.append(mol)
            mol_graphs.append(self.featurizer(mol))
            atom_offsets.append(atom_offset)
            atom_offset += mol.GetNumAtoms()

        candidate_atoms: list[int] = []
        candidate_tasks: list[int] = []
        candidate_ptr = [0]
        subgraph_atoms: list[int] = []
        subgraph_candidates: list[int] = []
        subgraph_distances: list[int] = []
        candidate_offset = 0
        for record, mol, atom_start in zip(records, mols, atom_offsets):
            distances = Chem.GetDistanceMatrix(mol)
            task_index = 0 if record.task == "acid" else 1
            for local_candidate, atom_index in enumerate(record.candidates):
                global_candidate = candidate_offset + local_candidate
                candidate_atoms.append(atom_start + atom_index)
                candidate_tasks.append(task_index)
                for neighbor in range(mol.GetNumAtoms()):
                    distance = int(round(distances[atom_index, neighbor]))
                    if distance <= self.shell_count:
                        subgraph_atoms.append(atom_start + neighbor)
                        subgraph_candidates.append(global_candidate)
                        subgraph_distances.append(distance)
            candidate_offset += len(record.candidates)
            candidate_ptr.append(candidate_offset)

        return SiteBatch(
            bmg=BatchMolGraph(mol_graphs),
            candidate_atom_index=torch.tensor(candidate_atoms, dtype=torch.long),
            candidate_task=torch.tensor(candidate_tasks, dtype=torch.long),
            candidate_ptr=torch.tensor(candidate_ptr, dtype=torch.long),
            subgraph_atom_index=torch.tensor(subgraph_atoms, dtype=torch.long),
            subgraph_candidate_index=torch.tensor(subgraph_candidates, dtype=torch.long),
            subgraph_distance=torch.tensor(subgraph_distances, dtype=torch.long),
            record_ids=tuple(record.record_id for record in records),
        )
