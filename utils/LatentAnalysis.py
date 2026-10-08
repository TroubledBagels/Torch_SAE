"""
LatentAnalysis.py
=================
Interactive latent-space analysis for spiking autoencoders on URBAN-SED style
samples.

The program does the expensive work once at startup:
  1. deterministically chooses N dataset examples (default: 10),
  2. runs them through the trained autoencoder,
  3. records latent spikes and latent membrane potentials at every timestep,
  4. derives the active class/class-combination for every timestep from the
     corresponding annotation file,
  5. starts a Dash browser dashboard that recomputes PCA and separability
     metrics interactively without rerunning the network, including a centred
     per-recording rolling mean whose window is controlled from the dashboard.

Editor / notebook usage
-----------------------
If you already create and load the model yourself, pass the actual nn.Module
instance directly to main():

    from LatentAnalysis import main
    from Network import RecurrentSpikingAutoencoder

    model = RecurrentSpikingAutoencoder(
        input_size=20,
        recurrent_size=8,
        encoder_sizes=[32, 16],
        decoder_sizes=[16, 32],
    )
    model.load_state_dict(torch.load("./models/model.pth", map_location="cpu"))

    app, bundle = main(
        model=model,
        root="~/data/URBAN-SED",
        latent_module="recurrent_lif",  # optional; auto-detected when omitted
        samples=10,
        seed=1337,
    )

Set run_server=False to perform all checks and latent extraction without
starting the browser server.

Typical CLI usage
-----------------
A checkpoint containing a complete nn.Module:

    python LatentAnalysis.py \
        --root ~/data/URBAN-SED \
        --checkpoint ./models/model.pth

A state-dict checkpoint:

    python LatentAnalysis.py \
        --root ~/data/URBAN-SED \
        --checkpoint ./models/model.pth \
        --model-file ./Network.py \
        --model-class RecurrentSpikingAutoencoder \
        --model-kwargs '{"input_size":40,"recurrent_size":8,"encoder_sizes":[32,16],"decoder_sizes":[16,32]}'

If automatic latent-neuron detection is not correct for a custom model, pass a
module path such as:

    --latent-module recurrent_lif
    --latent-module lif_layers.2

Dependencies
------------
    pip install dash plotly scikit-learn pandas numpy torch torchaudio torchvision

The default transform intentionally mirrors plotting_latent.py:
Squeeze -> Resample(44100, 16000) -> Unsqueeze -> ToSpikeTransform(20)
-> Squeeze -> Squeeze.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import math
import pathlib
import random
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import torchaudio
    import torchvision.transforms as tv_transforms
except Exception as exc:  # pragma: no cover - environment-specific import error
    raise ImportError(
        "LatentAnalysis requires torchaudio and torchvision because the default "
        "URBAN preprocessing pipeline mirrors plotting_latent.py."
    ) from exc

try:
    from dash import Dash, Input, Output, dcc, html, dash_table
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Dash is required. Install it with `pip install dash`."
    ) from exc

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Reproducibility / data containers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Inference is normally deterministic already, but these settings make the
    # intent explicit. Some operations can still be hardware/version dependent.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@dataclass
class SampleSummary:
    dataset_index: int
    filename: str
    timesteps: int
    latent_dims: int
    labels: str


@dataclass
class LatentBundle:
    spikes: np.ndarray              # [all_timesteps, latent_dims]
    membrane: np.ndarray            # [all_timesteps, latent_dims]
    labels: np.ndarray              # [all_timesteps] strings
    dataset_indices: np.ndarray     # [all_timesteps]
    filenames: np.ndarray           # [all_timesteps] strings
    local_timesteps: np.ndarray     # [all_timesteps]
    time_seconds: np.ndarray        # [all_timesteps]
    sample_summaries: List[SampleSummary]
    selected_indices: List[int]
    seed: int
    latent_module_name: str

    @property
    def latent_dims(self) -> int:
        return int(self.spikes.shape[1])


# -----------------------------------------------------------------------------
# Dynamic loading helpers
# -----------------------------------------------------------------------------


def load_python_module(module_or_path: str, synthetic_name: str) -> Any:
    """Load either an importable module name or an arbitrary .py file path."""
    path = pathlib.Path(module_or_path).expanduser()
    if path.exists():
        spec = importlib.util.spec_from_file_location(synthetic_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Python module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[synthetic_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_or_path)


def load_model(
    checkpoint: str,
    device: torch.device,
    model_file: Optional[str],
    model_class: Optional[str],
    model_kwargs: Dict[str, Any],
) -> nn.Module:
    """Load a complete nn.Module checkpoint or instantiate and load a state dict."""
    checkpoint_path = pathlib.Path(checkpoint).expanduser()
    # Loading a full serialized nn.Module uses Python pickle; only use trusted
    # checkpoints. State-dict checkpoints are preferred.
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch versions
        loaded = torch.load(checkpoint_path, map_location=device)

    if isinstance(loaded, nn.Module):
        model = loaded
    else:
        if not model_file or not model_class:
            raise ValueError(
                "Checkpoint is not a serialized nn.Module. Supply --model-file "
                "and --model-class (plus --model-kwargs if required)."
            )

        model_module = load_python_module(model_file, "latent_analysis_model_module")
        cls = getattr(model_module, model_class)
        model = cls(**model_kwargs)

        if isinstance(loaded, dict) and "state_dict" in loaded:
            state_dict = loaded["state_dict"]
        elif isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded

        # Accommodate DataParallel checkpoints.
        if isinstance(state_dict, dict) and state_dict:
            if all(str(k).startswith("module.") for k in state_dict.keys()):
                state_dict = {str(k)[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def build_default_transform(dataset_module: Any) -> Any:
    """Mirror the transform sequence used in plotting_latent.py."""
    required = ["SqueezeTransform", "UnsqueezeTransform", "ToSpikeTransform"]
    missing = [name for name in required if not hasattr(dataset_module, name)]
    if missing:
        raise AttributeError(
            "The default transform requires these objects in the dataset module: "
            + ", ".join(missing)
            + ". Add them there or modify build_default_transform()."
        )

    return tv_transforms.Compose(
        [
            dataset_module.SqueezeTransform(0),
            torchaudio.transforms.Resample(44100, 16000),
            dataset_module.UnsqueezeTransform(dim=0),
            dataset_module.ToSpikeTransform(num_channels=20),
            dataset_module.SqueezeTransform(0),
            dataset_module.SqueezeTransform(0),
        ]
    )


def build_dataset(
    root: str,
    dataset_module_name: str,
    dataset_class_name: str,
    split_name: str,
) -> Any:
    dataset_module = load_python_module(dataset_module_name, "latent_analysis_dataset_module")
    dataset_cls = getattr(dataset_module, dataset_class_name)
    transform = build_default_transform(dataset_module)

    split = split_name
    if hasattr(dataset_module, "DatasetSplit"):
        split_enum = dataset_module.DatasetSplit
        if hasattr(split_enum, split_name.upper()):
            split = getattr(split_enum, split_name.upper())

    return dataset_cls(
        root,
        split=split,
        transform=transform,
        random_chunking=False,
        only_background=False,
    )


# -----------------------------------------------------------------------------
# Latent module detection and recording
# -----------------------------------------------------------------------------


def resolve_module_path(model: nn.Module, path: str) -> nn.Module:
    current: Any = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{path!r} does not resolve to a torch.nn.Module")
    return current


def auto_detect_latent_module(model: nn.Module) -> Tuple[str, nn.Module]:
    """Detect the neuron whose output is the latent representation.

    This intentionally targets the model patterns in Network.py / Network(1).py:
    recurrent_lif, rlif1, lif1, the final encoder LIF, or the final encoder_lif.
    """
    direct_candidates = ["recurrent_lif", "rlif1", "lif1"]
    for name in direct_candidates:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            return name, module

    encoder_layers = getattr(model, "encoder_layers", None)
    lif_layers = getattr(model, "lif_layers", None)
    if encoder_layers is not None and lif_layers is not None and len(encoder_layers) > 0:
        idx = len(encoder_layers) - 1
        return f"lif_layers.{idx}", lif_layers[idx]

    encoder_lifs = getattr(model, "encoder_lifs", None)
    if encoder_lifs is not None and len(encoder_lifs) > 0:
        idx = len(encoder_lifs) - 1
        return f"encoder_lifs.{idx}", encoder_lifs[idx]

    raise ValueError(
        "Could not auto-detect the latent spiking neuron. Pass --latent-module "
        "with a dotted module path, e.g. recurrent_lif or lif_layers.2."
    )


def tensor_to_time_feature(array: torch.Tensor, expected_timesteps: Optional[int] = None) -> np.ndarray:
    """Convert common latent tensor layouts into [T, D]."""
    x = array.detach().float().cpu()

    while x.ndim > 2 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(f"Cannot convert latent tensor with shape {tuple(array.shape)} to [T, D]")

    if expected_timesteps is not None:
        if x.shape[0] == expected_timesteps:
            return x.numpy()
        if x.shape[1] == expected_timesteps:
            return x.transpose(0, 1).numpy()

    # Time is usually the larger axis for these 10-second samples.
    if x.shape[1] > x.shape[0]:
        x = x.transpose(0, 1)
    return x.numpy()


def _extract_hook_pair(output: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Interpret an SNN-neuron output as (spikes, membrane)."""
    if isinstance(output, (tuple, list)):
        tensors = [v for v in output if torch.is_tensor(v)]
        if len(tensors) >= 2:
            # snntorch Leaky/RLeaky expose spike first and membrane last.
            return tensors[0], tensors[-1]
    return None


def run_and_record_latent(
    model: nn.Module,
    x: torch.Tensor,
    latent_module: nn.Module,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one sample and capture latent spikes + membrane at each timestep."""
    spike_steps: List[torch.Tensor] = []
    membrane_steps: List[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        pair = _extract_hook_pair(output)
        if pair is None:
            return
        spk, mem = pair
        spike_steps.append(spk.detach().cpu())
        membrane_steps.append(mem.detach().cpu())

    handle = latent_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            # Network.py uses ret_lat; plotting_latent.py's older model uses
            # ret_latent. We try both, then plain inference.
            try:
                model(x, ret_lat=True)
            except TypeError:
                try:
                    model(x, ret_latent=True)
                except TypeError:
                    model(x)
    finally:
        handle.remove()

    if not spike_steps or not membrane_steps:
        raise RuntimeError(
            "The latent module hook did not observe (spike, membrane) outputs. "
            "This model may not use a snntorch-style neuron at that module. "
            "Pass the correct --latent-module, or adapt run_and_record_latent()."
        )

    expected_t = int(x.shape[-1])

    # Normal case: selected LIF/RLIF is called once per timestep.
    if len(spike_steps) == expected_t:
        spk = torch.stack([v.squeeze(0) for v in spike_steps], dim=0)
        mem = torch.stack([v.squeeze(0) for v in membrane_steps], dim=0)
        return spk.numpy(), mem.numpy()

    # Fallback for a module that receives/emits an entire sequence in one call.
    if len(spike_steps) == 1:
        return (
            tensor_to_time_feature(spike_steps[0], expected_t),
            tensor_to_time_feature(membrane_steps[0], expected_t),
        )

    raise RuntimeError(
        f"Latent module was called {len(spike_steps)} times for an input with "
        f"{expected_t} timesteps. This is ambiguous; choose a more specific "
        "--latent-module."
    )


def infer_model_input_features(model: nn.Module) -> Optional[int]:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)
    return None


def prepare_model_input(sample: torch.Tensor, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Convert a dataset sample to [B, C, T]."""
    if not torch.is_tensor(sample):
        sample = torch.as_tensor(sample)
    sample = sample.float()

    if sample.ndim == 3 and sample.shape[0] == 1:
        # Already likely [1, C, T] or [1, T, C].
        sample = sample.squeeze(0)

    if sample.ndim != 2:
        raise ValueError(
            f"Expected the dataset waveform to be 2-D after transforms, got {tuple(sample.shape)}"
        )

    in_features = infer_model_input_features(model)
    if in_features is not None and sample.shape[0] == in_features:
        ct = sample
    elif in_features is not None and sample.shape[1] == in_features:
        ct = sample.transpose(0, 1)
    else:
        # Dataset __getitem__ in the prompt returns [T, features].
        # Fall back to treating the larger axis as time.
        ct = sample.transpose(0, 1) if sample.shape[0] >= sample.shape[1] else sample

    return ct.unsqueeze(0).contiguous().to(device)


# -----------------------------------------------------------------------------
# Per-timestep class labels
# -----------------------------------------------------------------------------


def annotation_path_for(dataset: Any, idx: int) -> pathlib.Path:
    if not hasattr(dataset, "file_list") or not hasattr(dataset, "ann_path"):
        raise AttributeError(
            "URBANDataset must expose file_list and ann_path so timestep labels "
            "can be reconstructed from the annotation files."
        )
    audio_file = pathlib.Path(dataset.file_list[idx])
    return pathlib.Path(dataset.ann_path) / f"{audio_file.stem}.txt"


def filename_for(dataset: Any, idx: int) -> str:
    if hasattr(dataset, "file_list"):
        return pathlib.Path(dataset.file_list[idx]).name
    return f"dataset[{idx}]"


def timestep_class_combinations(
    annotation_file: pathlib.Path,
    timesteps: int,
    clip_duration: float,
) -> np.ndarray:
    ann = pd.read_csv(
        annotation_file,
        sep="\t",
        header=None,
        names=["onset", "offset", "event_label"],
        dtype={"onset": float, "offset": float, "event_label": str},
    )

    active: List[List[str]] = [[] for _ in range(timesteps)]
    if timesteps == 0:
        return np.empty(0, dtype=object)

    for row in ann.itertuples(index=False):
        onset = max(0.0, float(row.onset))
        offset = min(float(clip_duration), float(row.offset))
        if offset <= onset:
            continue

        # Match plotting_latent.py exactly: integer binning for onset and offset.
        start = int((onset / clip_duration) * timesteps)
        stop = int((offset / clip_duration) * timesteps)
        start = max(0, min(start, timesteps))
        stop = max(start, min(stop, timesteps))
        label = str(row.event_label)
        for t in range(start, stop):
            active[t].append(label)

    labels = []
    for names in active:
        if not names:
            labels.append("none")
        else:
            labels.append("+".join(sorted(set(names))))
    return np.asarray(labels, dtype=object)


# -----------------------------------------------------------------------------
# Dataset run / cache
# -----------------------------------------------------------------------------


def choose_indices(dataset_size: int, n_samples: int, seed: int) -> List[int]:
    if dataset_size <= 0:
        raise ValueError("Dataset is empty")
    n = min(int(n_samples), int(dataset_size))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(dataset_size, size=n, replace=False))


def collect_latents(
    dataset: Any,
    model: nn.Module,
    device: torch.device,
    latent_module_name: Optional[str],
    n_samples: int,
    seed: int,
    clip_duration: float,
) -> LatentBundle:
    selected = choose_indices(len(dataset), n_samples, seed)

    if latent_module_name:
        latent_module = resolve_module_path(model, latent_module_name)
        detected_name = latent_module_name
    else:
        detected_name, latent_module = auto_detect_latent_module(model)

    all_spikes: List[np.ndarray] = []
    all_membrane: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []
    all_filenames: List[np.ndarray] = []
    all_timesteps: List[np.ndarray] = []
    all_time_seconds: List[np.ndarray] = []
    summaries: List[SampleSummary] = []

    print(f"Seed: {seed}")
    print(f"Selected dataset indices: {selected}")
    print(f"Latent neuron module: {detected_name}")

    latent_dim: Optional[int] = None

    for position, idx in enumerate(selected, start=1):
        item = dataset[idx]
        waveform = item[0] if isinstance(item, (tuple, list)) else item
        x = prepare_model_input(waveform, model, device)

        spikes, membrane = run_and_record_latent(model, x, latent_module)
        if spikes.shape != membrane.shape:
            raise RuntimeError(
                f"Spike/membrane shape mismatch for sample {idx}: "
                f"{spikes.shape} vs {membrane.shape}"
            )

        if latent_dim is None:
            latent_dim = int(spikes.shape[1])
        elif int(spikes.shape[1]) != latent_dim:
            raise RuntimeError("Latent dimensionality changed between samples")

        ann_file = annotation_path_for(dataset, idx)
        labels = timestep_class_combinations(ann_file, spikes.shape[0], clip_duration)

        t = min(len(labels), spikes.shape[0])
        spikes = spikes[:t]
        membrane = membrane[:t]
        labels = labels[:t]

        fname = filename_for(dataset, idx)
        unique_labels = sorted(set(str(x) for x in labels))

        all_spikes.append(spikes)
        all_membrane.append(membrane)
        all_labels.append(labels)
        all_indices.append(np.full(t, idx, dtype=np.int64))
        all_filenames.append(np.full(t, fname, dtype=object))
        all_timesteps.append(np.arange(t, dtype=np.int64))
        all_time_seconds.append(np.arange(t, dtype=float) * (clip_duration / max(t, 1)))

        summaries.append(
            SampleSummary(
                dataset_index=idx,
                filename=fname,
                timesteps=t,
                latent_dims=latent_dim,
                labels=", ".join(unique_labels),
            )
        )
        print(
            f"[{position:02d}/{len(selected):02d}] idx={idx:<5d} "
            f"file={fname} T={t} D={latent_dim} classes={len(unique_labels)}"
        )

    return LatentBundle(
        spikes=np.concatenate(all_spikes, axis=0),
        membrane=np.concatenate(all_membrane, axis=0),
        labels=np.concatenate(all_labels, axis=0),
        dataset_indices=np.concatenate(all_indices, axis=0),
        filenames=np.concatenate(all_filenames, axis=0),
        local_timesteps=np.concatenate(all_timesteps, axis=0),
        time_seconds=np.concatenate(all_time_seconds, axis=0),
        sample_summaries=summaries,
        selected_indices=selected,
        seed=seed,
        latent_module_name=detected_name,
    )


# -----------------------------------------------------------------------------
# PCA / separability calculations
# -----------------------------------------------------------------------------


def between_within_scatter_ratio(x: np.ndarray, labels: np.ndarray) -> float:
    """trace(S_between) / trace(S_within), weighted by class size."""
    if len(x) == 0:
        return float("nan")
    overall = np.mean(x, axis=0)
    between = 0.0
    within = 0.0

    for label in np.unique(labels):
        cls = x[labels == label]
        if len(cls) == 0:
            continue
        center = np.mean(cls, axis=0)
        between += len(cls) * float(np.sum((center - overall) ** 2))
        within += float(np.sum((cls - center) ** 2))

    return between / (within + 1e-12)


def safe_separability_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    result = {
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
        "between_within": float("nan"),
    }

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or len(x) <= len(unique):
        return result

    result["between_within"] = between_within_scatter_ratio(x, labels)

    try:
        result["davies_bouldin"] = float(davies_bouldin_score(x, labels))
    except Exception:
        pass

    try:
        result["calinski_harabasz"] = float(calinski_harabasz_score(x, labels))
    except Exception:
        pass

    # Exact silhouette is O(N^2). Use a fixed random subset for responsiveness.
    try:
        sample_size = min(5000, len(x))
        result["silhouette"] = float(
            silhouette_score(
                x,
                labels,
                sample_size=sample_size if sample_size < len(x) else None,
                random_state=seed,
            )
        )
    except Exception:
        pass

    return result


def pairwise_centroid_separation(x: np.ndarray, labels: np.ndarray) -> Tuple[List[str], np.ndarray]:
    classes = sorted(str(c) for c in np.unique(labels))
    n = len(classes)
    mat = np.zeros((n, n), dtype=float)

    stats: Dict[str, Tuple[np.ndarray, float]] = {}
    for cls in classes:
        pts = x[labels == cls]
        centroid = np.mean(pts, axis=0)
        radius = float(np.sqrt(np.mean(np.sum((pts - centroid) ** 2, axis=1))))
        stats[cls] = (centroid, radius)

    for i, a in enumerate(classes):
        ca, ra = stats[a]
        for j, b in enumerate(classes):
            if i == j:
                continue
            cb, rb = stats[b]
            dist = float(np.linalg.norm(ca - cb))
            mat[i, j] = dist / (0.5 * (ra + rb) + 1e-12)

    return classes, mat


def filter_for_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    exclude_none: bool,
    min_class_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(len(labels), dtype=bool)
    if exclude_none:
        mask &= labels != "none"

    temp_labels = labels[mask]
    unique, counts = np.unique(temp_labels, return_counts=True)
    allowed = set(unique[counts >= int(min_class_size)])
    mask &= np.asarray([lab in allowed for lab in labels], dtype=bool)
    return x[mask], labels[mask], mask


def rolling_mean_by_sample(
    data: np.ndarray,
    dataset_indices: np.ndarray,
    window: int,
) -> np.ndarray:
    """Apply a centred rolling mean independently within each recorded sample.

    ``data`` is the concatenated latent array with shape ``[T_total, D]``.
    The rolling window must never cross a recording boundary, otherwise the end
    of one soundscape would be averaged with the beginning of the next one.

    ``window=1`` returns the raw values (as a copy).  For spikes, the rolling
    mean is naturally interpretable as local spike density / firing fraction.
    """
    x = np.asarray(data)
    sample_ids = np.asarray(dataset_indices)

    if x.ndim != 2:
        raise ValueError(f"Expected latent data with shape [T, D], got {x.shape}")
    if len(sample_ids) != len(x):
        raise ValueError(
            "dataset_indices must contain one entry for every latent timestep"
        )

    window = max(1, int(window))
    if window == 1:
        return x.copy()

    # Float output avoids integer truncation when smoothing binary spike arrays.
    out = np.empty(x.shape, dtype=np.result_type(x.dtype, np.float32))

    # np.unique sorts the IDs, but assignment uses the original global indices,
    # so the concatenated ordering is preserved exactly.
    for sample_id in np.unique(sample_ids):
        idx = np.flatnonzero(sample_ids == sample_id)
        if len(idx) == 0:
            continue

        local = pd.DataFrame(x[idx])
        smoothed = local.rolling(
            window=window,
            center=True,
            min_periods=1,
        ).mean()
        out[idx] = smoothed.to_numpy(dtype=out.dtype, copy=False)

    return out


def rolling_source_name(signal_source: str, window: int) -> str:
    """Human-readable signal name used in figure titles."""
    window = max(1, int(window))
    if window == 1:
        return f"{signal_source} (raw)"
    return f"{signal_source} (rolling mean, window={window})"


def fit_pca(
    x: np.ndarray,
    n_components: int,
    standardise: bool,
) -> Tuple[np.ndarray, PCA, Optional[StandardScaler]]:
    if standardise:
        scaler = StandardScaler()
        xp = scaler.fit_transform(x)
    else:
        scaler = None
        xp = x

    max_components = min(xp.shape[0], xp.shape[1])
    n_components = max(1, min(int(n_components), max_components))
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(xp)
    return projected, pca, scaler


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def blank_figure(message: str, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title, height=420)
    return fig


def class_palette(labels: Sequence[str]) -> Dict[str, str]:
    ordered = sorted(set(str(x) for x in labels), key=lambda x: (x != "none", x))
    palette = (
        px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Alphabet
    )
    return {label: palette[i % len(palette)] for i, label in enumerate(ordered)}


def deterministic_plot_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        if len(idx) > max_per_class:
            idx = np.sort(rng.choice(idx, size=max_per_class, replace=False))
        chosen.extend(int(i) for i in idx)
    return np.asarray(sorted(chosen), dtype=np.int64)


def split_class_combination(label: str) -> Tuple[str, ...]:
    """Return the individual classes represented by a combination label.

    For example, ``"siren+dog_bark"`` becomes ``("siren", "dog_bark")``.
    ``"none"`` is retained as its own background class.  Repeated components
    are removed while preserving their order.
    """
    label = str(label)
    if label == "none":
        return ("none",)

    parts = [part.strip() for part in label.split("+") if part.strip()]
    if not parts:
        return (label,)
    return tuple(dict.fromkeys(parts))


def individual_class_memberships(labels: np.ndarray) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int]]:
    """Build overlapping timestep memberships for each individual class.

    A timestep can belong to more than one returned membership.  Therefore a
    ``siren+dog_bark`` timestep contributes to both the ``siren`` and
    ``dog_bark`` counts.
    """
    memberships: Dict[str, List[int]] = {}
    for i, label in enumerate(labels):
        for cls in split_class_combination(str(label)):
            memberships.setdefault(cls, []).append(i)

    classes = sorted(memberships, key=lambda x: (x != "none", x))
    indices = {
        cls: np.asarray(memberships[cls], dtype=np.int64)
        for cls in classes
    }
    counts = {cls: int(len(indices[cls])) for cls in classes}
    return classes, indices, counts


def deterministic_membership_plot_indices(
    labels: np.ndarray,
    max_per_class: int,
    seed: int,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int]]:
    """Deterministically downsample each overlapping individual-class membership."""
    classes, membership_indices, counts = individual_class_memberships(labels)
    rng = np.random.default_rng(seed)
    chosen: Dict[str, np.ndarray] = {}

    for cls in classes:
        idx = membership_indices[cls]
        if len(idx) > max_per_class:
            idx = np.sort(rng.choice(idx, size=max_per_class, replace=False))
        chosen[cls] = idx

    return classes, chosen, counts


def make_latent_dimension_grid(
    data: np.ndarray,
    labels: np.ndarray,
    dataset_indices: np.ndarray,
    timesteps: np.ndarray,
    palette: Dict[str, str],
    source_name: str,
    seed: int,
    max_dims: int = 32,
) -> go.Figure:
    """Interactive analogue of plotting_latent.py Figure 3."""
    dims = min(data.shape[1], max_dims)
    cols = min(4, dims)
    rows = math.ceil(dims / cols)
    titles = [f"z{d}" for d in range(dims)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    plot_idx = deterministic_plot_indices(labels, max_per_class=500, seed=seed)
    p_labels = labels[plot_idx]
    classes = sorted(np.unique(p_labels), key=lambda x: (x != "none", str(x)))
    class_to_row = {cls: i for i, cls in enumerate(classes)}

    for d in range(dims):
        r, c = divmod(d, cols)
        r += 1
        c += 1
        for cls in classes:
            idx = plot_idx[p_labels == cls]
            if len(idx) == 0:
                continue
            custom = np.column_stack(
                [dataset_indices[idx], timesteps[idx], np.full(len(idx), str(cls), dtype=object)]
            )
            fig.add_trace(
                go.Scattergl(
                    x=data[idx, d],
                    y=np.full(len(idx), class_to_row[cls]),
                    mode="markers",
                    name=str(cls),
                    legendgroup=str(cls),
                    showlegend=(d == 0),
                    marker={"size": 4, "opacity": 0.45, "color": palette[str(cls)]},
                    customdata=custom,
                    hovertemplate=(
                        f"z{d}=%{{x:.4g}}<br>class=%{{customdata[2]}}"
                        "<br>dataset idx=%{customdata[0]}"
                        "<br>timestep=%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(classes))),
            ticktext=[str(x) for x in classes],
            showticklabels=(c == 1),
            automargin=True,
            range=[-0.5, len(classes) - 0.5],
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text=f"z{d} value", automargin=True, row=r, col=c)

    title = f"Every latent dimension by timestep class — {source_name}"
    if data.shape[1] > max_dims:
        title += f" (showing first {max_dims}/{data.shape[1]} dims)"

    longest_label = max((len(str(x)) for x in classes), default=4)
    left_margin = min(300, max(90, longest_label * 7))
    fig.update_layout(
        title=title,
        height=max(420, 320 * rows),
        legend_title="Class combination",
        margin={"l": left_margin, "r": 30, "t": 75, "b": 50},
    )
    return fig


def make_individual_class_latent_grid(
    data: np.ndarray,
    labels: np.ndarray,
    dataset_indices: np.ndarray,
    timesteps: np.ndarray,
    palette: Dict[str, str],
    source_name: str,
    seed: int,
    max_dims: int = 32,
) -> go.Figure:
    """Plot latent values against overlapping individual-class memberships.

    Combination labels are expanded rather than made mutually exclusive.  A
    point labelled ``siren+dog_bark`` is plotted on *both* the ``siren`` row and
    the ``dog_bark`` row.  Hover text retains the original combination label.
    """
    dims = min(data.shape[1], max_dims)
    cols = min(4, dims)
    rows = math.ceil(dims / cols)
    titles = [f"z{d}" for d in range(dims)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    classes, membership_indices, counts = deterministic_membership_plot_indices(
        labels,
        max_per_class=500,
        seed=seed,
    )
    class_to_row = {cls: i for i, cls in enumerate(classes)}
    ticktext = [f"{cls} ({counts[cls]:,})" for cls in classes]

    for d in range(dims):
        r, c = divmod(d, cols)
        r += 1
        c += 1

        for cls in classes:
            idx = membership_indices[cls]
            if len(idx) == 0:
                continue

            custom = np.column_stack(
                [
                    dataset_indices[idx],
                    timesteps[idx],
                    labels[idx].astype(object),
                    np.full(len(idx), str(cls), dtype=object),
                ]
            )
            fig.add_trace(
                go.Scattergl(
                    x=data[idx, d],
                    y=np.full(len(idx), class_to_row[cls]),
                    mode="markers",
                    name=str(cls),
                    legendgroup=str(cls),
                    showlegend=(d == 0),
                    marker={"size": 4, "opacity": 0.45, "color": palette[str(cls)]},
                    customdata=custom,
                    hovertemplate=(
                        f"z{d}=%{{x:.4g}}"
                        "<br>individual class=%{customdata[3]}"
                        "<br>original combination=%{customdata[2]}"
                        "<br>dataset idx=%{customdata[0]}"
                        "<br>timestep=%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )

        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(classes))),
            ticktext=ticktext,
            showticklabels=(c == 1),
            automargin=True,
            range=[-0.5, len(classes) - 0.5],
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text=f"z{d} value", automargin=True, row=r, col=c)

    title = f"Every latent dimension by individual class membership — {source_name}"
    if data.shape[1] > max_dims:
        title += f" (showing first {max_dims}/{data.shape[1]} dims)"

    longest_label = max((len(x) for x in ticktext), default=4)
    left_margin = min(320, max(100, longest_label * 7))
    fig.update_layout(
        title=title,
        height=max(420, 320 * rows),
        legend_title="Individual class",
        margin={"l": left_margin, "r": 30, "t": 75, "b": 50},
    )
    return fig


def make_individual_class_count_figure(
    labels: np.ndarray,
    palette: Dict[str, str],
) -> go.Figure:
    """Count timesteps containing each individual class, including overlaps."""
    classes, _, counts = individual_class_memberships(labels)
    fig = go.Figure(
        go.Bar(
            x=classes,
            y=[counts[cls] for cls in classes],
            marker_color=[palette[str(cls)] for cls in classes],
            customdata=np.asarray([[counts[cls]] for cls in classes]),
            hovertemplate="class=%{x}<br>member timesteps=%{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Timestep membership count by individual class",
        xaxis_title="Individual class",
        yaxis_title="Timesteps containing class",
        height=420,
        margin={"l": 65, "r": 25, "t": 70, "b": 95},
    )
    fig.update_xaxes(tickangle=-35, automargin=True)
    return fig


def make_time_grid(
    data: np.ndarray,
    bundle: LatentBundle,
    sample_index: int,
    palette: Dict[str, str],
    source_name: str,
    max_dims: int = 32,
) -> go.Figure:
    mask = bundle.dataset_indices == int(sample_index)
    if not np.any(mask):
        return blank_figure("Selected sample was not recorded", "Latent activity over time")

    local_data = data[mask]
    local_labels = bundle.labels[mask]
    local_time = bundle.time_seconds[mask]
    local_steps = bundle.local_timesteps[mask]
    dims = min(local_data.shape[1], max_dims)
    cols = min(4, dims)
    rows = math.ceil(dims / cols)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"z{d}" for d in range(dims)])
    classes = sorted(np.unique(local_labels), key=lambda x: (x != "none", str(x)))

    for d in range(dims):
        r, c = divmod(d, cols)
        r += 1
        c += 1
        for cls in classes:
            cmask = local_labels == cls
            fig.add_trace(
                go.Scattergl(
                    x=local_time[cmask],
                    y=local_data[cmask, d],
                    mode="markers",
                    name=str(cls),
                    legendgroup=str(cls),
                    showlegend=(d == 0),
                    marker={"size": 4, "opacity": 0.65, "color": palette[str(cls)]},
                    customdata=np.column_stack([local_steps[cmask], local_labels[cmask]]),
                    hovertemplate=(
                        "time=%{x:.3f}s<br>value=%{y:.4g}"
                        "<br>timestep=%{customdata[0]}"
                        "<br>class=%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )
        fig.update_xaxes(title_text="Time (s)", row=r, col=c)
        fig.update_yaxes(title_text=f"z{d}", row=r, col=c)

    fname = bundle.filenames[np.flatnonzero(mask)[0]]
    fig.update_layout(
        title=f"Latent neuron activity over time — {source_name} — idx {sample_index}: {fname}",
        height=max(420, 300 * rows),
        legend_title="Class combination",
        margin={"l": 50, "r": 20, "t": 75, "b": 40},
    )
    return fig


def metric_card(label: str, value: str, help_text: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "0.78rem", "opacity": 0.75}),
            html.Div(value, style={"fontSize": "1.35rem", "fontWeight": 700}),
            html.Div(help_text, style={"fontSize": "0.7rem", "opacity": 0.65}),
        ],
        style={
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "minWidth": "155px",
            "flex": "1 1 155px",
        },
    )


def fmt_metric(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------


def create_app(bundle: LatentBundle) -> Dash:
    app = Dash(__name__)
    palette = class_palette(bundle.labels)
    individual_classes, _, _ = individual_class_memberships(bundle.labels)
    individual_palette = class_palette(individual_classes)
    individual_count_fig = make_individual_class_count_figure(bundle.labels, individual_palette)
    dim_options = [{"label": f"z{i}", "value": i} for i in range(bundle.latent_dims)]
    default_dims = list(range(min(bundle.latent_dims, 8)))
    sample_options = [
        {"label": f"idx {s.dataset_index} — {s.filename}", "value": s.dataset_index}
        for s in bundle.sample_summaries
    ]

    sample_rows = [
        {
            "Dataset index": s.dataset_index,
            "File": s.filename,
            "Timesteps": s.timesteps,
            "Latent dims": s.latent_dims,
            "Classes / combinations": s.labels,
        }
        for s in bundle.sample_summaries
    ]

    app.layout = html.Div(
        [
            html.H1("Spiking Autoencoder Latent Analysis", style={"marginBottom": "4px"}),
            html.Div(
                f"Seed {bundle.seed} • samples {bundle.selected_indices} • latent module {bundle.latent_module_name}",
                style={"opacity": 0.75, "marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("PCA controls"),
                            html.Label("Latent signal"),
                            dcc.RadioItems(
                                id="signal-source",
                                options=[
                                    {"label": "Membrane potential", "value": "membrane"},
                                    {"label": "Spikes", "value": "spikes"},
                                ],
                                value="membrane",
                                inline=True,
                            ),
                            html.Br(),
                            html.Label("Rolling mean window (timesteps)"),
                            html.Div(
                                "1 = raw latent values. Smoothing is centred and applied separately to each recording.",
                                style={"fontSize": "0.78rem", "opacity": 0.7, "marginBottom": "6px"},
                            ),
                            dcc.Slider(
                                id="rolling-window",
                                min=1,
                                max=201,
                                step=2,
                                value=49,
                                marks={
                                    1: "1 (raw)",
                                    25: "25",
                                    49: "49",
                                    101: "101",
                                    151: "151",
                                    201: "201",
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                                updatemode="mouseup",
                            ),
                            html.Br(),
                            html.Label("Original latent dimensions used for PCA"),
                            dcc.Dropdown(
                                id="latent-dims",
                                options=dim_options,
                                value=default_dims,
                                multi=True,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("Number of principal components"),
                            dcc.Dropdown(
                                id="n-components",
                                options=[
                                    {"label": str(i), "value": i}
                                    for i in range(2, max(3, min(bundle.latent_dims, 12)) + 1)
                                ],
                                value=min(3, max(2, bundle.latent_dims)),
                                clearable=False,
                            ),
                            dcc.Checklist(
                                id="pca-options",
                                options=[
                                    {"label": " Standardise selected latent dimensions before PCA", "value": "standardise"},
                                    {"label": " Exclude background ('none') from separability metrics", "value": "exclude_none"},
                                ],
                                value=["standardise"],
                            ),
                            html.Br(),
                            html.Label("Minimum timesteps per class for separability metrics"),
                            dcc.Slider(
                                id="min-class-size",
                                min=2,
                                max=100,
                                step=1,
                                value=10,
                                marks={2: "2", 10: "10", 25: "25", 50: "50", 100: "100"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style={
                            "padding": "14px",
                            "border": "1px solid #ddd",
                            "borderRadius": "10px",
                            "flex": "1 1 460px",
                        },
                    ),
                    html.Div(
                        [
                            html.H3("Recorded samples"),
                            dash_table.DataTable(
                                data=sample_rows,
                                columns=[{"name": c, "id": c} for c in sample_rows[0].keys()],
                                page_size=10,
                                style_table={"overflowX": "auto", "maxHeight": "360px"},
                                style_cell={
                                    "textAlign": "left",
                                    "fontSize": "12px",
                                    "whiteSpace": "normal",
                                    "height": "auto",
                                    "padding": "5px",
                                },
                                style_header={"fontWeight": "bold"},
                            ),
                        ],
                        style={
                            "padding": "14px",
                            "border": "1px solid #ddd",
                            "borderRadius": "10px",
                            "flex": "1 1 640px",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
            ),
            html.H2("PCA and class separability"),
            html.Div(id="metric-cards", style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
            html.Div(
                [
                    dcc.Graph(id="pca-scatter", style={"flex": "1 1 640px"}),
                    dcc.Graph(id="variance-graph", style={"flex": "1 1 420px"}),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
            dcc.Graph(id="separation-heatmap"),
            html.H2("Latent dimensions"),
            html.Div(
                "Each subplot places timestep values along one latent dimension and class combinations on rows. "
                "The rolling-mean control above is applied before these plots and before PCA/metrics. "
                "The visual is deterministically downsampled per class for browser performance; PCA/metrics use the full filtered data.",
                style={"opacity": 0.72, "marginBottom": "6px"},
            ),
            dcc.Graph(id="latent-grid"),
            html.H2("Latent dimensions by individual class membership"),
            html.Div(
                "Combination labels are expanded into overlapping memberships. For example, a timestep labelled "
                "'siren+dog_bark' contributes to both the 'siren' and 'dog_bark' rows. Counts in parentheses are "
                "the number of timesteps containing that individual class, so the class counts can sum to more "
                "than the total number of analysed timesteps.",
                style={"opacity": 0.72, "marginBottom": "6px"},
            ),
            dcc.Graph(id="individual-class-counts", figure=individual_count_fig),
            dcc.Graph(id="individual-class-latent-grid"),
            html.H2("One recorded sample over time"),
            dcc.Dropdown(
                id="sample-detail",
                options=sample_options,
                value=bundle.selected_indices[0],
                clearable=False,
                style={"maxWidth": "700px", "marginBottom": "8px"},
            ),
            dcc.Graph(id="time-grid"),
        ],
        style={
            "maxWidth": "1600px",
            "margin": "0 auto",
            "padding": "18px",
            "fontFamily": "Arial, sans-serif",
        },
    )

    @app.callback(
        Output("pca-scatter", "figure"),
        Output("variance-graph", "figure"),
        Output("separation-heatmap", "figure"),
        Output("metric-cards", "children"),
        Input("signal-source", "value"),
        Input("rolling-window", "value"),
        Input("latent-dims", "value"),
        Input("n-components", "value"),
        Input("pca-options", "value"),
        Input("min-class-size", "value"),
    )
    def update_pca(
        signal_source: str,
        rolling_window: int,
        selected_dims: Sequence[int],
        requested_components: int,
        pca_options: Sequence[str],
        min_class_size: int,
    ):
        raw_data = bundle.membrane if signal_source == "membrane" else bundle.spikes
        rolling_window = max(1, int(rolling_window or 1))
        data = rolling_mean_by_sample(raw_data, bundle.dataset_indices, rolling_window)
        display_source = rolling_source_name(signal_source, rolling_window)
        selected_dims = sorted(set(int(d) for d in (selected_dims or [])))
        selected_dims = [d for d in selected_dims if 0 <= d < data.shape[1]]
        if not selected_dims:
            blank = blank_figure("Select at least one latent dimension")
            return blank, blank, blank, [metric_card("Status", "No dimensions selected")]

        x_selected = data[:, selected_dims]
        standardise = "standardise" in (pca_options or [])
        exclude_none = "exclude_none" in (pca_options or [])

        x_metric, labels_metric, mask = filter_for_metrics(
            x_selected,
            bundle.labels,
            exclude_none=exclude_none,
            min_class_size=int(min_class_size),
        )

        if len(x_metric) < 3 or len(np.unique(labels_metric)) < 2:
            msg = "Not enough points/classes remain after filtering."
            blank = blank_figure(msg)
            return blank, blank, blank, [metric_card("Status", msg)]

        variances = np.var(x_metric, axis=0)
        variable_mask = variances > 1e-12
        effective_dims = [d for d, keep in zip(selected_dims, variable_mask) if keep]
        if not np.any(variable_mask):
            blank = blank_figure("All selected latent dimensions are constant for the filtered points")
            return blank, blank, blank, [metric_card("Status", "Selected dimensions are constant")]
        x_metric = x_metric[:, variable_mask]

        n_components = min(
            int(requested_components),
            x_metric.shape[1],
            x_metric.shape[0],
        )
        if n_components < 1:
            blank = blank_figure("PCA cannot be fit with the current selection")
            return blank, blank, blank, [metric_card("Status", "PCA unavailable")]

        projected, pca, _ = fit_pca(x_metric, n_components, standardise)
        original_metrics = safe_separability_metrics(x_metric, labels_metric, bundle.seed)
        pca_metrics = safe_separability_metrics(projected, labels_metric, bundle.seed)

        used_global = np.flatnonzero(mask)
        df = pd.DataFrame(
            {
                "class": labels_metric.astype(str),
                "dataset_index": bundle.dataset_indices[used_global],
                "filename": bundle.filenames[used_global],
                "timestep": bundle.local_timesteps[used_global],
                "time_s": bundle.time_seconds[used_global],
            }
        )
        for i in range(projected.shape[1]):
            df[f"PC{i+1}"] = projected[:, i]

        if projected.shape[1] >= 3:
            pca_fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="class",
                color_discrete_map=palette,
                hover_data=["dataset_index", "filename", "timestep", "time_s"],
                opacity=0.55,
                title=f"PCA of {display_source}: z{effective_dims}",
            )
            pca_fig.update_traces(marker={"size": 3})
        elif projected.shape[1] == 2:
            pca_fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="class",
                color_discrete_map=palette,
                hover_data=["dataset_index", "filename", "timestep", "time_s"],
                opacity=0.55,
                render_mode="webgl",
                title=f"PCA of {display_source}: z{effective_dims}",
            )
            pca_fig.update_traces(marker={"size": 4})
        else:
            df["row"] = pd.Categorical(df["class"]).codes
            pca_fig = px.scatter(
                df,
                x="PC1",
                y="row",
                color="class",
                color_discrete_map=palette,
                render_mode="webgl",
                title=f"PCA of {display_source}: one component",
            )

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        variance_fig = go.Figure()
        variance_fig.add_bar(
            x=[f"PC{i+1}" for i in range(len(explained))],
            y=explained,
            name="Explained variance ratio",
        )
        variance_fig.add_trace(
            go.Scatter(
                x=[f"PC{i+1}" for i in range(len(cumulative))],
                y=cumulative,
                mode="lines+markers",
                name="Cumulative",
            )
        )
        variance_fig.update_layout(
            title="PCA explained variance",
            yaxis_title="Variance ratio",
            yaxis_range=[0, 1.02],
        )

        classes = sorted(str(c) for c in np.unique(labels_metric))
        if len(classes) <= 30:
            classes, sep_mat = pairwise_centroid_separation(projected, labels_metric)
            heatmap = go.Figure(
                data=go.Heatmap(
                    z=sep_mat,
                    x=classes,
                    y=classes,
                    colorbar={"title": "centroid distance / mean radius"},
                    hovertemplate="%{y} vs %{x}<br>separation=%{z:.3f}<extra></extra>",
                )
            )
            heatmap.update_layout(title="Pairwise class separation in PCA space", height=max(430, 20 * len(classes) + 180))
        else:
            heatmap = blank_figure(
                f"{len(classes)} class combinations remain; heatmap hidden above 30 classes.",
                "Pairwise class separation in PCA space",
            )

        counts = pd.Series(labels_metric).value_counts()
        cards = [
            metric_card(
                "Temporal smoothing",
                "raw" if rolling_window == 1 else f"{rolling_window} steps",
                "centred rolling mean; per recording",
            ),
            metric_card("Timesteps analysed", f"{len(x_metric):,}", f"{len(counts)} classes/combinations"),
            metric_card("Selected dimensions", f"{len(effective_dims)}/{len(selected_dims)} variable", ", ".join(f"z{d}" for d in effective_dims)),
            metric_card("PCA variance retained", f"{cumulative[-1] * 100:.2f}%", f"{len(explained)} PCs"),
            metric_card("PC1 variance", f"{explained[0] * 100:.2f}%"),
            metric_card("PCA silhouette ↑", fmt_metric(pca_metrics["silhouette"]), "higher is better"),
            metric_card("PCA Davies–Bouldin ↓", fmt_metric(pca_metrics["davies_bouldin"]), "lower is better"),
            metric_card("PCA Calinski–Harabasz ↑", fmt_metric(pca_metrics["calinski_harabasz"], 2), "higher is better"),
            metric_card("PCA between/within ↑", fmt_metric(pca_metrics["between_within"]), "higher = tighter/further classes"),
            metric_card("Original silhouette ↑", fmt_metric(original_metrics["silhouette"]), "before PCA"),
            metric_card("Original between/within ↑", fmt_metric(original_metrics["between_within"]), "before PCA"),
        ]

        return pca_fig, variance_fig, heatmap, cards

    @app.callback(
        Output("latent-grid", "figure"),
        Output("individual-class-latent-grid", "figure"),
        Output("time-grid", "figure"),
        Input("signal-source", "value"),
        Input("rolling-window", "value"),
        Input("sample-detail", "value"),
    )
    def update_latent_views(signal_source: str, rolling_window: int, sample_index: int):
        raw_data = bundle.membrane if signal_source == "membrane" else bundle.spikes
        rolling_window = max(1, int(rolling_window or 1))
        data = rolling_mean_by_sample(raw_data, bundle.dataset_indices, rolling_window)
        display_source = rolling_source_name(signal_source, rolling_window)
        latent_fig = make_latent_dimension_grid(
            data,
            bundle.labels,
            bundle.dataset_indices,
            bundle.local_timesteps,
            palette,
            source_name=display_source,
            seed=bundle.seed,
        )
        individual_class_fig = make_individual_class_latent_grid(
            data,
            bundle.labels,
            bundle.dataset_indices,
            bundle.local_timesteps,
            individual_palette,
            source_name=display_source,
            seed=bundle.seed,
        )
        time_fig = make_time_grid(
            data,
            bundle,
            sample_index=int(sample_index),
            palette=palette,
            source_name=display_source,
        )
        return latent_fig, individual_class_fig, time_fig

    return app


# -----------------------------------------------------------------------------
# Programmatic/editor API + CLI
# -----------------------------------------------------------------------------


def _normalise_device(device: Optional[Any]) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device if isinstance(device, torch.device) else torch.device(str(device))


def validate_analysis_setup(
    model: nn.Module,
    dataset: Any,
    device: torch.device,
    latent_module_name: Optional[str] = None,
) -> Tuple[str, nn.Module]:
    """Run cheap structural checks before the ten analysis samples are recorded.

    The actual forward/hook compatibility check is intentionally left to
    ``collect_latents`` / ``run_and_record_latent`` because that requires a real
    transformed dataset sample.  This function catches the common editor-time
    mistakes first and gives a clearer error message.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"model must be a torch.nn.Module instance, got {type(model).__name__}. "
            "Instantiate your model before passing it to main()."
        )

    if dataset is None:
        raise ValueError("dataset must not be None")

    try:
        dataset_size = len(dataset)
    except Exception as exc:
        raise TypeError("dataset must implement __len__()") from exc
    if dataset_size <= 0:
        raise ValueError("Dataset is empty")

    if not hasattr(dataset, "__getitem__"):
        raise TypeError("dataset must implement __getitem__()")

    # Timestep labels are reconstructed from the annotation files rather than
    # from the dataset target tensor, matching plotting_latent.py.
    for attr in ("file_list", "ann_path"):
        if not hasattr(dataset, attr):
            raise AttributeError(
                f"Dataset is missing {attr!r}. URBANDataset-style file_list and "
                "ann_path are required to reconstruct per-timestep classes."
            )

    if latent_module_name:
        latent_module = resolve_module_path(model, latent_module_name)
        detected_name = latent_module_name
    else:
        detected_name, latent_module = auto_detect_latent_module(model)

    input_features = infer_model_input_features(model)
    print("LatentAnalysis pre-flight checks:")
    print(f"  model:         {model.__class__.__name__}")
    print(f"  device:        {device}")
    print(f"  dataset size:  {dataset_size}")
    print(f"  latent module: {detected_name}")
    if input_features is not None:
        print(f"  inferred model input features: {input_features}")
    else:
        print("  inferred model input features: unknown (no nn.Linear found)")
    print("  structural checks: PASS")

    return detected_name, latent_module


def run_analysis_dashboard(
    model: nn.Module,
    *,
    root: Optional[str] = None,
    dataset: Optional[Any] = None,
    dataset_module: str = "EDDataset",
    dataset_class: str = "URBANDataset",
    split: str = "train",
    latent_module: Optional[str] = None,
    samples: int = 10,
    seed: int = 1337,
    clip_duration: float = 10.0,
    device: Optional[Any] = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    run_server: bool = True,
) -> Tuple[Dash, LatentBundle]:
    """Run LatentAnalysis with an already-created model instance.

    This is the recommended entry point when running from an IDE/editor.  The
    supplied model is *not* recreated and no checkpoint is loaded here; whatever
    weights/state are present on ``model`` are the weights/state that are
    analysed.

    Parameters
    ----------
    model:
        Instantiated ``torch.nn.Module`` to analyse.
    root:
        URBAN-SED root. Required only when ``dataset`` is not passed.
    dataset:
        Optional already-created dataset. If supplied, ``root`` and the dataset
        module/class/split arguments are ignored.
    latent_module:
        Dotted path to the latent LIF/RLIF module, e.g. ``"recurrent_lif"`` or
        ``"lif_layers.2"``. Leave as None for architecture-based auto-detection.
    run_server:
        If False, run all checks/inference and build the Dash app, but do not
        block by starting the web server. Useful while debugging in an editor.

    Returns
    -------
    (app, bundle):
        The Dash application and all recorded spikes/membranes/labels.
    """
    set_seed(int(seed))
    device_obj = _normalise_device(device)

    if dataset is None:
        if root is None:
            raise ValueError(
                "root is required when dataset is not supplied. Example: "
                "main(model=my_model, root='~/data/URBAN-SED')"
            )
        dataset = build_dataset(
            root=root,
            dataset_module_name=dataset_module,
            dataset_class_name=dataset_class,
            split_name=split,
        )

    # Use the exact model object created by the caller, while normalising the
    # standard inference state expected by the analysis pipeline.
    model = model.to(device_obj)
    model.eval()

    detected_name, _ = validate_analysis_setup(
        model=model,
        dataset=dataset,
        device=device_obj,
        latent_module_name=latent_module,
    )

    # collect_latents repeats module resolution by name so the same path is used
    # for both the pre-flight check and the actual hook registration.
    bundle = collect_latents(
        dataset=dataset,
        model=model,
        device=device_obj,
        latent_module_name=detected_name,
        n_samples=int(samples),
        seed=int(seed),
        clip_duration=float(clip_duration),
    )

    print(
        f"Forward/latent checks: PASS. Recorded {len(bundle.labels):,} timesteps "
        f"from {len(bundle.selected_indices)} samples with "
        f"{bundle.latent_dims} latent dimensions."
    )

    app = create_app(bundle)

    if run_server:
        print(f"Open http://{host}:{port} in a browser.")
        # Dash 2.x supports app.run; older versions use run_server.
        if hasattr(app, "run"):
            app.run(host=host, port=port, debug=debug, use_reloader=False)
        else:  # pragma: no cover
            app.run_server(host=host, port=port, debug=debug, use_reloader=False)

    return app, bundle


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive PCA/separability dashboard for spiking-autoencoder latents")
    parser.add_argument("--root", required=True, help="URBAN dataset root containing audio/ and annotations/")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pth/.pt)")
    parser.add_argument("--model-file", default=None, help="Python module name or .py path defining the model class")
    parser.add_argument("--model-class", default=None, help="Model class name for state-dict checkpoints")
    parser.add_argument("--model-kwargs", default="{}", help="JSON dict passed to the model constructor")
    parser.add_argument("--dataset-module", default="EDDataset", help="Dataset module name or .py path")
    parser.add_argument("--dataset-class", default="URBANDataset", help="Dataset class name")
    parser.add_argument("--split", default="train", help="Dataset split, usually train/validation/test")
    parser.add_argument("--latent-module", default=None, help="Optional dotted module path for the latent LIF/RLIF")
    parser.add_argument("--samples", type=int, default=10, help="Number of random dataset examples")
    parser.add_argument("--seed", type=int, default=1337, help="Fixed seed for reproducible sample choice and plotting")
    parser.add_argument("--clip-duration", type=float, default=10.0, help="Annotation clip duration in seconds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(
    model: Optional[nn.Module] = None,
    *,
    root: Optional[str] = None,
    dataset: Optional[Any] = None,
    dataset_module: str = "EDDataset",
    dataset_class: str = "URBANDataset",
    split: str = "train",
    latent_module: Optional[str] = None,
    samples: int = 10,
    seed: int = 1337,
    clip_duration: float = 10.0,
    device: Optional[Any] = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    run_server: bool = True,
    argv: Optional[Sequence[str]] = None,
) -> Tuple[Dash, LatentBundle]:
    """Editor-friendly main function, with CLI fallback when model is omitted.

    Editor use::

        model = MySpikingAutoencoder(...)
        model.load_state_dict(torch.load("my_model.pth"))

        app, bundle = main(
            model=model,
            root="~/data/URBAN-SED",
            latent_module="recurrent_lif",  # optional
            samples=10,
            seed=1337,
        )

    When called as ``main()`` with no model, command-line arguments are parsed so
    the original CLI behaviour remains available.
    """
    if model is not None:
        return run_analysis_dashboard(
            model=model,
            root=root,
            dataset=dataset,
            dataset_module=dataset_module,
            dataset_class=dataset_class,
            split=split,
            latent_module=latent_module,
            samples=samples,
            seed=seed,
            clip_duration=clip_duration,
            device=device,
            host=host,
            port=port,
            debug=debug,
            run_server=run_server,
        )

    # CLI mode: preserve the previous behaviour and load the model from the
    # checkpoint/model-class arguments.
    args = parse_args(argv)
    set_seed(args.seed)

    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        raise ValueError("--model-kwargs must be valid JSON") from exc
    if not isinstance(model_kwargs, dict):
        raise ValueError("--model-kwargs must decode to a JSON object/dict")

    device_obj = _normalise_device(args.device)
    cli_model = load_model(
        checkpoint=args.checkpoint,
        device=device_obj,
        model_file=args.model_file,
        model_class=args.model_class,
        model_kwargs=model_kwargs,
    )

    return run_analysis_dashboard(
        model=cli_model,
        root=args.root,
        dataset_module=args.dataset_module,
        dataset_class=args.dataset_class,
        split=args.split,
        latent_module=args.latent_module,
        samples=args.samples,
        seed=args.seed,
        clip_duration=args.clip_duration,
        device=device_obj,
        host=args.host,
        port=args.port,
        debug=args.debug,
        run_server=True,
    )


if __name__ == "__main__":
    main()
