from __future__ import annotations

import argparse
import csv
import hashlib
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem

from .chemistry import CandidateSite, find_candidate_sites, load_patterns
from .data import SiteBatchCollator, SiteRecord
from .model import SiteDMPNN, SiteModelConfig, aggregate_site_pka
from .standardize import standardize_smiles

TaskChoice = Literal["acid", "base", "both"]
_RESOURCE_PACKAGE = "cubepka.resources"


@lru_cache(maxsize=1)
def _manifest() -> dict[str, object]:
    resource = resources.files(_RESOURCE_PACKAGE).joinpath("model_manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _patterns():
    resource = resources.files(_RESOURCE_PACKAGE).joinpath("smarts_pattern.tsv")
    with resources.as_file(resource) as path:
        observed = _sha256(path)
        expected = str(_manifest()["smarts_sha256"])
        if observed != expected:
            raise RuntimeError(
                f"SMARTS checksum mismatch: expected {expected}, observed {observed}"
            )
        return load_patterns(path)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@lru_cache(maxsize=8)
def _load_model(task: str, device_name: str) -> tuple[SiteDMPNN, str]:
    if task not in {"acid", "base"}:
        raise ValueError("task must be acid or base")
    entry = _manifest()["models"][task]
    checkpoint_resource = resources.files(_RESOURCE_PACKAGE).joinpath(entry["checkpoint"])
    with resources.as_file(checkpoint_resource) as checkpoint_path:
        observed_hash = _sha256(checkpoint_path)
        expected_hash = str(entry["checkpoint_sha256"])
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"{task} checkpoint checksum mismatch: expected {expected_hash}, "
                f"observed {observed_hash}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if checkpoint["model_config"] != entry["model_config"]:
        raise RuntimeError(f"{task} checkpoint config does not match the release manifest")
    config = SiteModelConfig(**checkpoint["model_config"])
    model = SiteDMPNN(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(torch.device(device_name))
    model.eval()
    return model, observed_hash


def _task_prediction(
    model_smiles: str,
    task: Literal["acid", "base"],
    device: torch.device,
) -> dict[str, object]:
    mol = Chem.MolFromSmiles(model_smiles)
    if mol is None:
        raise ValueError("standardized_smiles_not_parseable")
    sites = find_candidate_sites(mol, task, _patterns())
    if not sites:
        return {"task": task, "status": "no_candidate", "sites": []}

    candidates = tuple(site.atom_index for site in sites)
    record = SiteRecord("prediction", model_smiles, task, candidates)
    model, checkpoint_hash = _load_model(task, str(device))
    batch = SiteBatchCollator(shell_count=model.config.shell_count)((record,)).to(device)
    with torch.inference_mode():
        values = model(batch).detach().cpu()
        aggregate = aggregate_site_pka(values, task)
    selected_offset = int(values.argmin().item() if task == "acid" else values.argmax().item())

    site_rows = []
    for offset, (site, value) in enumerate(zip(sites, values.tolist())):
        site_rows.append(
            {
                "atom_index": site.atom_index,
                "atom_number": site.atom_index + 1,
                "atom_symbol": site.atom_symbol,
                "family": site.family,
                "pattern_ids": list(site.pattern_ids),
                "pka": float(value),
                "selected": offset == selected_offset,
            }
        )
    selected: CandidateSite = sites[selected_offset]
    return {
        "task": task,
        "status": "ok",
        "strongest_pka": float(aggregate),
        "selected_atom_index": selected.atom_index,
        "selected_atom_number": selected.atom_index + 1,
        "checkpoint_sha256": checkpoint_hash,
        "sites": site_rows,
    }


def predict_smiles(
    smiles: str,
    *,
    task: TaskChoice = "both",
    device: str = "auto",
) -> dict[str, object]:
    """Predict candidate-site and strongest-site pKa values for one molecule."""
    if task not in {"acid", "base", "both"}:
        raise ValueError("task must be acid, base, or both")
    identity = standardize_smiles(smiles)
    resolved_device = _resolve_device(device)
    tasks = ("acid", "base") if task == "both" else (task,)
    return {
        "model_version": _manifest()["model_version"],
        "input_smiles": smiles,
        "model_smiles": identity.model_smiles,
        "parent_key": identity.parent_key,
        "device": str(resolved_device),
        "predictions": [
            _task_prediction(identity.model_smiles, current_task, resolved_device)
            for current_task in tasks
        ],
    }


def _safe_prediction(
    smiles: str,
    *,
    record_id: str,
    task: TaskChoice,
    device: str,
) -> dict[str, object]:
    try:
        result = predict_smiles(smiles, task=task, device=device)
        return {"record_id": record_id, "status": "ok", **result}
    except (ValueError, RuntimeError) as exc:
        return {
            "record_id": record_id,
            "status": "error",
            "input_smiles": smiles,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _predict_csv(
    path: Path,
    *,
    smiles_column: str,
    id_column: str | None,
    task_column: str | None,
    task: TaskChoice,
    device: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if smiles_column not in fields:
            raise ValueError(f"missing SMILES column: {smiles_column}")
        if id_column and id_column not in fields:
            raise ValueError(f"missing ID column: {id_column}")
        if task_column and task_column not in fields:
            raise ValueError(f"missing task column: {task_column}")
        for row_number, row in enumerate(reader, start=1):
            record_id = row[id_column] if id_column else str(row_number)
            row_task = row[task_column].strip().lower() if task_column else task
            rows.append(
                _safe_prediction(
                    row[smiles_column],
                    record_id=record_id,
                    task=row_task,
                    device=device,
                )
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cube-pka",
        description="Predict site-resolved acidic and basic pKa with Cube-pKa V4.",
    )
    parser.add_argument("smiles", nargs="?", help="one input SMILES")
    parser.add_argument("--input-csv", type=Path, help="CSV file for batch prediction")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column")
    parser.add_argument(
        "--task-column",
        help="optional CSV column containing acid, base, or both per row",
    )
    parser.add_argument("--task", choices=("acid", "base", "both"), default="both")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output", type=Path, help="write JSON/JSONL to this path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.smiles) == bool(args.input_csv):
        raise SystemExit("provide exactly one of SMILES or --input-csv")
    if args.smiles:
        result = _safe_prediction(
            args.smiles,
            record_id="1",
            task=args.task,
            device=args.device,
        )
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    else:
        results = _predict_csv(
            args.input_csv,
            smiles_column=args.smiles_column,
            id_column=args.id_column,
            task_column=args.task_column,
            task=args.task,
            device=args.device,
        )
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
