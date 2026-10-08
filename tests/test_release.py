from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from importlib import resources
from pathlib import Path

from rdkit import Chem

from cubepka.chemistry import find_candidate_sites, is_sulfonamide_nitrogen, load_patterns
from cubepka.predict import _predict_csv, predict_smiles
from cubepka.standardize import standardize_smiles


class ReleaseIntegrityTests(unittest.TestCase):
    def test_resource_hashes_match_manifest(self) -> None:
        root = resources.files("cubepka.resources")
        manifest = json.loads(root.joinpath("model_manifest.json").read_text())
        for task, entry in manifest["models"].items():
            observed = hashlib.sha256(root.joinpath(entry["checkpoint"]).read_bytes()).hexdigest()
            self.assertEqual(observed, entry["checkpoint_sha256"], task)
        observed = hashlib.sha256(root.joinpath("smarts_pattern.tsv").read_bytes()).hexdigest()
        self.assertEqual(observed, manifest["smarts_sha256"])

    def test_reference_predictions_match_canonical_v4(self) -> None:
        cases = (
            ("CC(=O)Oc1ccccc1C(=O)O", "acid", 3.456773281097412),
            ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "acid", 4.572304725646973),
            ("CCN(CC)CC(=O)Nc1c(C)cccc1C", "base", 7.9458327293396),
            ("CN(C)CCOC(c1ccccc1)c1ccccc1", "base", 8.785608291625977),
        )
        for smiles, task, expected in cases:
            result = predict_smiles(smiles, task=task, device="cpu")
            prediction = result["predictions"][0]
            self.assertEqual(prediction["status"], "ok")
            self.assertAlmostEqual(prediction["strongest_pka"], expected, places=6)

    def test_base_sulfonamide_nitrogen_is_not_a_candidate(self) -> None:
        identity = standardize_smiles("NS(=O)(=O)c1ccc(N)cc1")
        mol = Chem.MolFromSmiles(identity.model_smiles)
        root = resources.files("cubepka.resources")
        patterns = load_patterns(root.joinpath("smarts_pattern.tsv"))
        candidates = find_candidate_sites(mol, "base", patterns)
        self.assertTrue(candidates)
        mol_h = Chem.AddHs(mol)
        self.assertTrue(
            all(not is_sulfonamide_nitrogen(mol_h, site.atom_index) for site in candidates)
        )

    def test_invalid_smiles_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid_smiles"):
            predict_smiles("not-a-smiles", task="acid", device="cpu")

    def test_batch_task_column_and_row_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "molecules.csv"
            path.write_text(
                "id,smiles,task\n"
                "aspirin,CC(=O)Oc1ccccc1C(=O)O,acid\n"
                "bad,not-a-smiles,base\n",
                encoding="utf-8",
            )
            rows = _predict_csv(
                path,
                smiles_column="smiles",
                id_column="id",
                task_column="task",
                task="both",
                device="cpu",
            )
        self.assertEqual(rows[0]["predictions"][0]["task"], "acid")
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
