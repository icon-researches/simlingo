"""Persistent sample-index cache that avoids repeated route scans.

BaseDataset selects it with the dataset config key index_cache:
  null    scan the routes on every construction (upstream behaviour);
  strict  a prebuilt index is required;
  warn    use the prebuilt index if present, else scan;
  build   scan, then write the index.
Entries live under index_cache_dir (relative to the launch directory), in a
subdirectory keyed by the settings that decide which samples exist. Build them once
before launching DDP (scripts/build_index_cache.py in the training repository);
the ranks only read. Caching fixes the train/validation split and skips scan-time
RNG draws, so augmentation RNG differs from uncached runs. Rebuild with --force to
refresh. BaseDataset.__init__ still loads templates and augmenters; only the route
scan is skipped.

Each NPZ is unpacked once into an adjacent .mmap directory (extra disk space
equal to its unpacked arrays). Processes read the same immutable NPY files and
workers reopen those paths instead of receiving array contents through pickle.
Old generations stay available for workers already using a rebuilt cache.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock
from hydra.utils import get_original_cwd

DEFAULT_DIR = ".cache/dataset_index"
MODES = (None, "strict", "warn", "build")

# Only fields that change sample membership belong in the cache key.
_KEY_FIELDS = (
    "data_path",
    "use_old_towns",
    "use_only_old_towns",
    "use_town13",
    "skip_first_n_frames",
    "pred_len",
    "hist_len",
)
# The other dataset attributes the cache reads.
_LOCATION_FIELDS = ("split", "bucket_name", "bucket_path", "index_cache_dir")

_ARRAY_FIELDS = ("images", "boxes", "measurements", "sample_start", "augment_exists")
_INDEX_FIELDS = (*_ARRAY_FIELDS, "alternative_trajectories")

_ROUTE_COUNTS: dict[str, int] = {}


class _CacheMiss(Exception):
    """Raised internally when an entry is absent, stale, or inconsistent."""


def repo_path() -> str:
    """The launch directory: Hydra's original working directory, or the current one
    when Hydra is not running (e.g. under hydra.compose or in tests)."""
    try:
        return get_original_cwd()
    except ValueError:
        return os.getcwd()


def _class_name(dreamer: bool) -> str:
    """The upstream class name in entry file names, also for subclasses."""
    return "Data_Dreamer" if dreamer else "Data_Driving"


def _cache_root(cfg: Mapping[str, Any]) -> Path:
    root = Path(str(cfg.get("index_cache_dir") or DEFAULT_DIR))
    return root if root.is_absolute() else Path(repo_path()) / root


def _key_payload(cfg: Mapping[str, Any]) -> dict[str, Any]:
    payload = {field: cfg.get(field) for field in _KEY_FIELDS}
    # Include the resolved data path to invalidate the cache when a symlink changes.
    payload["data_root"] = os.path.realpath(
        os.path.join(repo_path(), str(cfg.get("data_path")))
    )
    return payload


def cache_dir_for(cfg: Mapping[str, Any]) -> Path:
    payload = _key_payload(cfg)
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return _cache_root(cfg) / digest


def entry_path(cfg: Mapping[str, Any], class_name: str) -> Path:
    """The entry of the dataset that cfg (key fields, split, bucket_name and
    index_cache_dir) describes. Use the upstream class name so cache builders and
    readers share filenames."""
    name = "{}__{}__{}.npz".format(class_name, cfg.get("split"), cfg.get("bucket_name"))
    return cache_dir_for(cfg) / name


def _dataset_cfg(dataset) -> dict[str, Any]:
    return {
        name: getattr(dataset, name, None) for name in (*_KEY_FIELDS, *_LOCATION_FIELDS)
    }


def _route_dir_count(cfg: Mapping[str, Any]) -> int:
    """Count route directories once per process to detect newly extracted data."""
    root = os.path.join(repo_path(), str(cfg.get("data_path")))
    if root not in _ROUTE_COUNTS:
        _ROUTE_COUNTS[root] = len(glob.glob(root + "/data/simlingo/*/*/*/Town*"))
    return _ROUTE_COUNTS[root]


def _bucket_fingerprint(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """Size and mtime of buckets_paths.pkl, or None for the unbucketed set."""
    bucket = cfg.get("bucket_name")
    if bucket in (None, "all"):
        return None
    path = os.path.join(repo_path(), str(cfg.get("bucket_path")), "buckets_paths.pkl")
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime": int(stat.st_mtime)}


def _write_dir_meta(root: Path, cfg: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "key_fields": _key_payload(cfg),
        "route_dir_count": _route_dir_count(cfg),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = root / f"meta.json.tmp{os.getpid()}"
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    os.replace(str(tmp), str(root / "meta.json"))


def _check_dir_meta(root: Path, cfg: Mapping[str, Any]) -> None:
    path = root / "meta.json"
    if not path.is_file():
        raise _CacheMiss(f"no cache directory at {root}")
    meta = json.loads(path.read_text())

    expected = json.dumps(_key_payload(cfg), sort_keys=True, default=str)
    stored = json.dumps(meta.get("key_fields"), sort_keys=True, default=str)
    if expected != stored:
        raise _CacheMiss("config fields differ from the ones the cache was built with")

    actual = _route_dir_count(cfg)
    if meta.get("route_dir_count") != actual:
        raise _CacheMiss(
            "dataset changed on disk: {} route dirs at build time, {} now".format(
                meta.get("route_dir_count"), actual
            )
        )


def _save_entry(entry: Path, dataset, cfg: Mapping[str, Any], class_name: str) -> None:
    arrays = {
        "images": dataset.images,
        "boxes": dataset.boxes,
        "measurements": dataset.measurements,
        "sample_start": np.asarray(dataset.sample_start),
        "augment_exists": np.asarray(dataset.augment_exists, dtype=bool),
    }
    # Only the dreamer set fills this in; upstream leaves it a plain list on
    # the driving set, where BaseDataset.__init__ recreates it.
    alternatives = getattr(dataset, "alternative_trajectories", None)
    if isinstance(alternatives, np.ndarray):
        arrays["alternative_trajectories"] = alternatives

    arrays["meta"] = np.array(
        json.dumps(
            {
                "class": class_name,
                "split": cfg.get("split"),
                "bucket_name": cfg.get("bucket_name"),
                "n_samples": int(dataset.images.shape[0]),
                "bucket_pkl": _bucket_fingerprint(cfg),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            sort_keys=True,
            default=str,
        )
    )

    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{entry}.tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, str(entry))


def _mmap_generation(entry: Path) -> str:
    stat = entry.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return hashlib.sha256(repr(identity).encode()).hexdigest()[:20]


def _prepare_mmap_entry(entry: Path) -> Path:
    """Unpack an existing NPZ once, without materializing its arrays in RAM.

    A lock and atomic directory rename keep simultaneous DDP readers from
    seeing partial files. Rebuilt NPZs get a new directory, so running workers
    can keep reading their original generation without changed sample indexes.
    """
    entry = entry.resolve()
    if not entry.is_file():
        raise _CacheMiss(f"no entry at {entry.name}")
    root = entry.with_suffix(".mmap")
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "convert.lock")):
        generation = _mmap_generation(entry)
        destination = root / generation
        if destination.is_dir():
            return destination

        with tempfile.TemporaryDirectory(prefix=".tmp-", dir=root) as tmp:
            staging = Path(tmp) / "arrays"
            staging.mkdir()
            with zipfile.ZipFile(entry) as archive:
                names = set(archive.namelist())
                required = {f"{name}.npy" for name in (*_ARRAY_FIELDS, "meta")}
                if not required.issubset(names):
                    raise _CacheMiss(f"entry is missing {sorted(required - names)}")
                for name in (*_INDEX_FIELDS, "meta"):
                    filename = f"{name}.npy"
                    if filename in names:
                        with (
                            archive.open(filename) as source,
                            (staging / filename).open("wb") as target,
                        ):
                            shutil.copyfileobj(source, target, length=1024 * 1024)
            if _mmap_generation(entry) != generation:
                raise _CacheMiss(
                    "entry changed while preparing memory maps; retry loading"
                )
            staging.rename(destination)
        print(f"[index-cache] prepared read-only memory maps for {entry.name}")
        return destination


def _load_entry(entry: Path, cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    directory = _prepare_mmap_entry(entry)
    meta = json.loads(str(np.load(directory / "meta.npy", allow_pickle=False).item()))

    stored_pkl = meta.get("bucket_pkl")
    current_pkl = _bucket_fingerprint(cfg)
    if stored_pkl != current_pkl:
        raise _CacheMiss("buckets_paths.pkl changed since the cache was built")

    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in _INDEX_FIELDS
        if (directory / f"{name}.npy").is_file()
    }
    missing = [name for name in _ARRAY_FIELDS if name not in arrays]
    if missing:
        raise _CacheMiss("entry is missing {}".format(", ".join(missing)))

    lengths = {name: len(arrays[name]) for name in arrays}
    if len(set(lengths.values())) > 1:
        raise _CacheMiss(f"entry arrays disagree on length: {lengths}")
    if lengths["images"] != int(meta.get("n_samples", -1)):
        raise _CacheMiss("entry is truncated")

    return arrays


def _mode(dataset) -> str | None:
    mode = getattr(dataset, "index_cache", None)
    if mode not in MODES:
        raise ValueError(
            f"index_cache must be null, strict, warn or build, not {mode!r}"
        )
    return mode


def load(dataset, dreamer: bool) -> bool:
    """Set the dataset's sample index from its cache entry, as read-only memory maps.

    Returns False when dataset.index_cache is None or "build" (the caller scans) and,
    in "warn" mode, when the entry is missing or stale; "strict" raises then.
    """
    mode = _mode(dataset)
    if mode in (None, "build"):
        return False
    cfg = _dataset_cfg(dataset)
    name = _class_name(dreamer)
    entry = entry_path(cfg, name)
    try:
        _check_dir_meta(entry.parent, cfg)
        arrays = _load_entry(entry, cfg)
    except (_CacheMiss, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        message = (
            f"index cache unusable for {name} split={cfg['split']} bucket={cfg['bucket_name']}: {exc}\n"
            f"  looked in : {entry.parent}\n"
            "  build with: python scripts/build_index_cache.py"
        )
        if mode == "strict":
            raise RuntimeError(message) from exc
        print(f"[index-cache] {message}\n[index-cache] falling back to a full scan")
        return False

    for field, value in arrays.items():
        setattr(dataset, field, value)
    return True


def save(dataset, dreamer: bool) -> None:
    """Write the cache entry (and its directory's meta.json) of a freshly scanned
    dataset, then switch the dataset to the entry's memory maps."""
    cfg = _dataset_cfg(dataset)
    name = _class_name(dreamer)
    entry = entry_path(cfg, name)
    _write_dir_meta(entry.parent, cfg)
    _save_entry(entry, dataset, cfg, name)
    for field, value in _load_entry(entry, cfg).items():
        setattr(dataset, field, value)
    print(f"[index-cache] wrote {entry.name} ({dataset.images.shape[0]} samples)")


class MemoryMappedIndex:
    """Reopen index files in workers; NumPy's default pickle copies memmap data."""

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        paths = {}
        for name in _INDEX_FIELDS:
            value = state.get(name)
            if isinstance(value, np.memmap):
                paths[name] = str(value.filename)
                del state[name]
        state["_index_mmap_paths"] = paths
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        paths = state.pop("_index_mmap_paths", {})
        self.__dict__.update(state)
        for name, path in paths.items():
            setattr(self, name, np.load(path, mmap_mode="r", allow_pickle=False))
