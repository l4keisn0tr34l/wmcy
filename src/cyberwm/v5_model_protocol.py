"""Shared fail-closed helpers for the frozen V5 train/validation model protocol."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.cyberwm.graph_rssm import GraphFeatureScaler, fit_graph_feature_scaler

ROOT = Path(__file__).resolve().parents[2]
MODES = ("passive", "action", "passive_action")
EXPECTED_PROTOCOL_ID = "mvp_v5_train_validation_v1_1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    path = path or ROOT / "configs/mvp_v5_training_protocol.json"
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError(f"unexpected V5 protocol: {protocol.get('protocol_id')}")
    return protocol


def assert_test_sealed(sequences_dir: Path | None = None) -> None:
    sequences_dir = sequences_dir or ROOT / "outputs/mvp_v5/sequences"
    present = [str(sequences_dir / mode / "test") for mode in MODES
               if (sequences_dir / mode / "test").exists()]
    if present: raise PermissionError(f"V5 test exports must remain absent during training: {present}")


def load_export(mode: str, split: str, sequences_dir: Path | None = None
                ) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise PermissionError("V5 model training helpers expose train/validation only")
    if mode not in MODES: raise ValueError(mode)
    assert_test_sealed(sequences_dir)
    directory = (sequences_dir or ROOT / "outputs/mvp_v5/sequences") / mode / split
    with np.load(directory / f"{split}.npz") as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    manifest = pd.read_csv(directory / "sample_manifest.csv")
    metadata = json.loads((directory / "feature_metadata.json").read_text())
    if metadata.get("test_access") is not False or set(manifest.split) != {split}:
        raise ValueError(f"{mode}/{split}: export seal marker mismatch")
    return arrays, manifest, metadata


def unique_train_context_rows(sequences_dir: Path | None = None) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Deduplicate overlapping train contexts by episode/state; never use future rows."""
    rows: list[np.ndarray] = []; keys: list[tuple[str, int]] = []; seen: set[tuple[str, int]] = set()
    for mode in ("passive", "action"):
        arrays, manifest, _ = load_export(mode, "train", sequences_dir)
        context = arrays["context_states"]
        for sample_index, row in manifest.iterrows():
            first, last = int(row.context_first_state), int(row.context_last_state)
            if last - first + 1 != context.shape[1]: raise ValueError(f"{mode}: context index width")
            for local_index, state_index in enumerate(range(first, last + 1)):
                key = (str(row.episode_id), state_index)
                if key in seen: continue
                seen.add(key); keys.append(key); rows.append(context[sample_index, local_index])
    values = np.stack(rows).astype(np.float32)
    if values.shape != (440, 345) or not np.isfinite(values).all():
        raise ValueError(f"unexpected unique train context rows: {values.shape}")
    return values, keys


def fit_v5_scaler(sequences_dir: Path | None = None) -> tuple[GraphFeatureScaler, list[tuple[str, int]]]:
    values, keys = unique_train_context_rows(sequences_dir)
    scaler = fit_graph_feature_scaler(values[:, None, :], 15, 18, 5, 12, 20)
    return scaler, keys


def save_scaler(path: Path, scaler: GraphFeatureScaler, keys: list[tuple[str, int]]) -> None:
    episode = np.asarray([key[0] for key in keys])
    state = np.asarray([key[1] for key in keys], dtype=np.int32)
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_, episode_id=episode, state_index=state,
             n_samples_seen=np.asarray(scaler.n_samples_seen_))


def load_scaler(path: Path) -> GraphFeatureScaler:
    with np.load(path) as loaded:
        mean = loaded["mean"].copy(); scale = loaded["scale"].copy(); seen = int(loaded["n_samples_seen"])
    if mean.shape != (345,) or scale.shape != (345,) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("invalid V5 scaler")
    scaler = GraphFeatureScaler(); scaler.mean_ = mean; scaler.scale_ = scale
    scaler.var_ = scale ** 2; scaler.n_features_in_ = 345; scaler.n_samples_seen_ = seen
    return scaler


def compatible_parameters(model: torch.nn.Module, source: dict[str, torch.Tensor],
                          include_prefixes: tuple[str, ...] | None = None
                          ) -> tuple[dict[str, torch.Tensor], list[str]]:
    destination = model.state_dict(); selected: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        if include_prefixes is not None and not name.startswith(include_prefixes): continue
        if name in destination and destination[name].shape == value.shape:
            selected[name] = value
    if not selected: raise ValueError("no compatible initialization parameters")
    result = model.load_state_dict(selected, strict=False)
    if result.unexpected_keys: raise ValueError(f"unexpected initialization keys: {result.unexpected_keys}")
    return selected, list(result.missing_keys)
