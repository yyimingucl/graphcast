from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import pathlib
import subprocess
import time
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


_LOG = logging.getLogger(__name__)


def _utc_now_iso() -> str:
  return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_path(p: Optional[str]) -> Optional[pathlib.Path]:
  if p is None:
    return None
  p = str(p).strip()
  if not p or p.lower() == "null":
    return None
  return pathlib.Path(p)


def _get_public_gcs_bucket(bucket_name: str):
  try:
    from google.cloud import storage  # type: ignore
  except Exception as e:
    raise RuntimeError(
        "google-cloud-storage is required for public GCS downloads. "
        "Install via: pip install -e '.[gcs]'"
    ) from e

  gcs_client = storage.Client.create_anonymous_client()
  gcs_bucket = gcs_client.get_bucket(bucket_name)
  return gcs_bucket


def _download_public_gcs_blob(*, bucket_name: str, blob_path: str, dest_path: pathlib.Path) -> None:
  gcs_bucket = _get_public_gcs_bucket(bucket_name)
  blob_path = blob_path.lstrip("/")
  dest_path.parent.mkdir(parents=True, exist_ok=True)
  blob = gcs_bucket.blob(blob_path)
  _LOG.info("Downloading gs://%s/%s -> %s", bucket_name, blob_path, dest_path)
  blob.download_to_filename(str(dest_path))


def _asset_dest_path(
    *,
    asset_root: pathlib.Path,
    dir_prefix: str,
    blob_relpath: str,
) -> pathlib.Path:
  """Map a blob relpath like 'stats/foo.nc' to a stable local cache path."""
  rel = pathlib.Path(str(blob_relpath).lstrip("/"))
  parts = rel.parts
  if not parts:
    raise ValueError("blob_relpath must be non-empty")
  kind = parts[0]
  rest = pathlib.Path(*parts[1:]) if len(parts) > 1 else pathlib.Path(rel.name)
  prefix = str(dir_prefix).strip("/").replace("/", "_") or "root"
  return asset_root / kind / prefix / rest


def _fetch_public_gcs_to_cache(
    *,
    bucket_name: str,
    blob_path: str,
    dest_path: pathlib.Path,
) -> pathlib.Path:
  blob_path = blob_path.lstrip("/")
  if dest_path.exists():
    _LOG.info("Cache hit: %s", dest_path)
    return dest_path
  _download_public_gcs_blob(bucket_name=bucket_name, blob_path=blob_path, dest_path=dest_path)
  return dest_path


def resolve_asset(
    *,
    source: str,
    local_path: Optional[str],
    bucket_name: str,
    dir_prefix: str,
    blob_relpath: Optional[str],
    base_dir: pathlib.Path,
    asset_root: pathlib.Path,
) -> pathlib.Path:
  source = str(source)
  if source not in {"gcs", "local"}:
    raise ValueError(f"Unsupported model.source={source!r}, expected 'gcs' or 'local'.")

  if source == "local":
    lp = _as_path(local_path)
    if lp is None:
      raise FileNotFoundError("Missing local_path for source='local'.")
    if not lp.is_absolute():
      lp = base_dir / lp
    if not lp.exists():
      raise FileNotFoundError(f"Local path does not exist: {str(lp)!r}")
    return lp

  if not blob_relpath:
    raise FileNotFoundError("Missing blob_relpath for source='gcs'.")
  blob_path = f"{str(dir_prefix)}{str(blob_relpath)}"
  dest_path = _asset_dest_path(
      asset_root=asset_root,
      dir_prefix=str(dir_prefix),
      blob_relpath=str(blob_relpath),
  )
  return _fetch_public_gcs_to_cache(
      bucket_name=str(bucket_name),
      blob_path=blob_path,
      dest_path=dest_path,
  )


def _load_netcdf(path: pathlib.Path):
  import xarray as xr

  # Prefer open_dataset(...).load() so we can explicitly control decoding.
  # Some xarray versions warn about future timedelta decoding behavior unless
  # decode_timedelta is set.
  try:
    ds = xr.open_dataset(path, decode_timedelta=True)  # type: ignore[call-arg]
    ds = ds.load()
    return ds
  except TypeError:
    # Older xarray without decode_timedelta kwarg.
    pass

  ds = xr.open_dataset(path)
  ds = ds.load()
  return ds


def _log_dataset_overview(name: str, ds) -> None:
  try:
    dims = dict(getattr(ds, "sizes", {}))
  except Exception:
    dims = {}
  _LOG.info("%s dims: %s", name, dims)
  try:
    keys = list(ds.data_vars.keys())
    _LOG.info("%s variables (%d): %s", name, len(keys), ", ".join(keys[:20]))
  except Exception:
    pass
  try:
    if "time" in ds.coords:
      time_vals = ds.coords["time"].values
      preview = list(time_vals[:10])
      _LOG.info("%s time preview: %s", name, preview)
  except Exception:
    pass


def _infer_uniform_timestep(time_td):
  import numpy as np
  import pandas as pd

  if len(time_td) < 2:
    raise ValueError("Need at least 2 timesteps to infer timestep.")
  diffs_ns = np.diff(time_td.astype("timedelta64[ns]").astype(np.int64))
  diffs_ns = diffs_ns[diffs_ns != 0]
  if diffs_ns.size == 0:
    raise ValueError("Cannot infer timestep (all time deltas identical).")
  if (diffs_ns < 0).any():
    raise ValueError("Dataset time coordinate must be non-decreasing.")
  if len(set(diffs_ns.tolist())) != 1:
    raise ValueError("Dataset time coordinate must be uniformly spaced for this mode.")
  return pd.to_timedelta(int(diffs_ns[0]), unit="ns")


def _parse_target_lead_times(cfg: DictConfig, dataset):
  """Return target lead times as an explicit list of pd.Timedelta.

  Note: `graphcast.data_utils.extract_input_target_times` shifts the dataset's
  time coordinate by `target_duration - time[-1]` and then selects:
    - targets at `target_lead_times`
    - inputs in the interval (-input_duration, 0]

  For the provided example datasets, the notebooks choose `target_duration`
  such that there are exactly the expected number of input frames after the
  shift. `mode=all_but_inputs` mirrors that behavior.
  """
  import pandas as pd

  mode = str(cfg.mode)
  if "time" not in dataset.coords:
    raise ValueError("Dataset is missing a 'time' coordinate required for target lead time selection.")

  time_td = pd.to_timedelta(dataset.coords["time"].values)
  step = _infer_uniform_timestep(time_td)

  if mode == "dataset_positive":
    positive_td = sorted([t for t in time_td if t > pd.Timedelta(0)])
    if not positive_td:
      raise ValueError("Dataset has no positive lead times in its 'time' coordinate.")
    _LOG.warning(
        "target_lead_times.mode=dataset_positive is deprecated; prefer all_but_inputs. "
        "Selected %d lead times.",
        len(positive_td),
    )
    return positive_td

  if mode == "all_but_inputs":
    num_frames = int(getattr(cfg, "input_frames", 2) or 2)
    if num_frames < 1:
      raise ValueError("target_lead_times.input_frames must be >= 1")
    if len(time_td) <= num_frames:
      raise ValueError(
          f"Dataset has {len(time_td)} frames but input_frames={num_frames}; need more frames than inputs."
      )
    stop = step
    start = step
    lead_times = []
    t = start
    while t <= stop:
      lead_times.append(t)
      t += step
    _LOG.info(
        "Selected target lead times (%d), step=%s stop=%s (dataset_frames=%d input_frames=%d)",
        len(lead_times),
        step,
        stop,
        len(time_td),
        num_frames,
    )
    return lead_times

  if mode == "range":
    start = pd.Timedelta(str(cfg.start))
    stop_raw = getattr(cfg, "stop", None)
    if stop_raw is None or str(stop_raw).lower() in {"null", "none", "dataset_max"}:
      stop = time_td[-1]
    else:
      stop = pd.Timedelta(str(stop_raw))
    step_cfg = pd.Timedelta(str(cfg.step))
    if step_cfg <= pd.Timedelta(0):
      raise ValueError(f"target_lead_times.step must be positive, got {cfg.step!r}")
    if stop < start:
      raise ValueError(f"target_lead_times.stop must be >= start, got {stop} < {start}")

    lead_times = []
    t = start
    while t <= stop:
      lead_times.append(t)
      t += step_cfg
    _LOG.info("Selected target lead times (%d) via range", len(lead_times))
    return lead_times

  if mode == "list":
    values = [pd.Timedelta(str(x)) for x in cfg.values]
    if not values:
      raise ValueError("target_lead_times.values must be non-empty for mode='list'.")
    _LOG.info("Selected target lead times (%d) via list", len(values))
    return values

  raise ValueError(f"Unsupported data.target_lead_times.mode={mode!r}")


def _maybe_set_env_var(name: str, value: str) -> None:
  if os.environ.get(name) is None:
    os.environ[name] = value


def _nvidia_smi_query() -> dict[str, Any]:
  # Best-effort; if nvidia-smi isn't present, return {}.
  try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
  except Exception:
    return {}

  # Single GPU line or multiple lines.
  gpus = []
  for line in out.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
      continue
    name, mem_used, mem_total, driver = parts
    gpus.append(
        {
            "name": name,
            "memory_used_mib": int(float(mem_used)),
            "memory_total_mib": int(float(mem_total)),
            "driver_version": driver,
        }
    )
  return {"gpus": gpus}


def _block_until_ready(predictions) -> None:
  import jax

  if not getattr(predictions, "data_vars", None):
    return
  first = next(iter(predictions.data_vars.values()))
  data = getattr(first, "data", None)
  try:
    jax.block_until_ready(data)
  except Exception:
    pass


def _percentile(values: list[float], q: float) -> float:
  if not values:
    return float("nan")
  values_sorted = sorted(values)
  if len(values_sorted) == 1:
    return values_sorted[0]
  idx = int(round((q / 100.0) * (len(values_sorted) - 1)))
  idx = max(0, min(len(values_sorted) - 1, idx))
  return values_sorted[idx]


def _to_pyfloat(x) -> float:
  """Convert scalar xarray/numpy/jax values (including xarray_jax wrappers) to float."""
  import numpy as np
  try:
    from graphcast import xarray_jax
  except Exception:
    xarray_jax = None

  # xarray.DataArray
  if hasattr(x, "data") and not isinstance(x, (np.ndarray, float, int)):
    data = x.data
  else:
    data = x

  if xarray_jax is not None:
    try:
      data = xarray_jax.unwrap_data(data, require_jax=False)
    except Exception:
      pass

  arr = np.asarray(data)
  if arr.size != 1:
    raise ValueError(f"Expected scalar convertible to float, got shape={arr.shape}")
  return float(arr.reshape(()))


def _weighted_mean(da, weights, dims: list[str], skipna: bool) -> Any:
  return da.weighted(weights).mean(dims, skipna=skipna)


def _weighted_std(da, weights, dims: list[str], skipna: bool) -> Any:
  mean = _weighted_mean(da, weights, dims, skipna=skipna)
  mean2 = _weighted_mean(da ** 2, weights, dims, skipna=skipna)
  var = mean2 - mean ** 2
  return var ** 0.5


def _compute_metrics_graphcast(*, targets, predictions, cfg_metrics: DictConfig) -> dict[str, Any]:
  import numpy as np
  from graphcast import losses

  skipna = True
  per_var: dict[str, Any] = {}
  overall_rmse = []
  overall_mae = []
  overall_nrmse = []

  variables = list(cfg_metrics.variables) if cfg_metrics.variables else list(targets.data_vars.keys())
  for var in variables:
    t = targets[var]
    p = predictions[var]
    err = p - t

    dims = [d for d in t.dims if d != "batch"]
    weights = None
    if cfg_metrics.compute_weighted_latitude and "lat" in t.dims:
      weights = losses.normalized_latitude_weights(t)
    if cfg_metrics.compute_weighted_level and "level" in t.dims:
      lvl_w = losses.normalized_level_weights(t)
      weights = (lvl_w if weights is None else weights * lvl_w)

    if weights is None:
      mae = np.abs(err).mean(dims, skipna=skipna)
      rmse = ((err ** 2).mean(dims, skipna=skipna)) ** 0.5
      denom = t.std(dims, skipna=skipna)
    else:
      mae = np.abs(err).weighted(weights).mean(dims, skipna=skipna)
      rmse = ((err ** 2).weighted(weights).mean(dims, skipna=skipna)) ** 0.5
      denom = _weighted_std(t, weights, dims, skipna=skipna)
    nrmse = rmse / denom

    per_var[var] = {
        "mae": _to_pyfloat(mae.mean(skipna=skipna)),
        "rmse": _to_pyfloat(rmse.mean(skipna=skipna)),
        "nrmse": _to_pyfloat(nrmse.mean(skipna=skipna)),
    }
    overall_mae.append(per_var[var]["mae"])
    overall_rmse.append(per_var[var]["rmse"])
    overall_nrmse.append(per_var[var]["nrmse"])

  overall = {
      "mae": float(np.mean(overall_mae)) if overall_mae else float("nan"),
      "rmse": float(np.mean(overall_rmse)) if overall_rmse else float("nan"),
      "nrmse": float(np.mean(overall_nrmse)) if overall_nrmse else float("nan"),
  }
  return {"overall": overall, "per_variable": per_var if cfg_metrics.report_per_variable else {}}


def _crps_bias_corrected(targets, predictions) -> Any:
  import numpy as np

  if predictions.sizes.get("sample", 1) < 2:
    raise ValueError("predictions must have dim 'sample' with size at least 2.")
  sum_dims = ["sample", "sample2"]
  preds2 = predictions.rename({"sample": "sample2"})
  num_samps = predictions.sizes["sample"]
  num_samps2 = num_samps - 1
  mean_abs_diff = np.abs(predictions - preds2).sum(dim=sum_dims, skipna=False) / (num_samps * num_samps2)
  mean_abs_err = np.abs(targets - predictions).sum(dim="sample", skipna=False) / num_samps
  return mean_abs_err - 0.5 * mean_abs_diff


def _compute_metrics_gencast(*, targets, predictions, cfg_metrics: DictConfig) -> dict[str, Any]:
  import numpy as np

  # Ensemble mean deterministic scores + CRPS.
  ens_mean = predictions.mean(dim="sample")

  base = _compute_metrics_graphcast(targets=targets, predictions=ens_mean, cfg_metrics=cfg_metrics)

  variables = list(cfg_metrics.variables) if cfg_metrics.variables else list(targets.data_vars.keys())
  per_var: dict[str, Any] = dict(base["per_variable"])
  overall_crps = []

  for var in variables:
    crps = _crps_bias_corrected(targets[var], predictions[var])
    per_var.setdefault(var, {})
    per_var[var]["crps"] = _to_pyfloat(crps.mean(skipna=True))
    overall_crps.append(per_var[var]["crps"])

  base["overall"]["crps"] = float(np.mean(overall_crps)) if overall_crps else float("nan")
  if cfg_metrics.report_per_variable:
    base["per_variable"] = per_var
  return base


def _maybe_save_visualizations(*, output_dir: pathlib.Path, targets, predictions, cfg_viz: DictConfig) -> list[str]:
  if not cfg_viz.save_visualizations:
    return []

  import numpy as np
  import matplotlib.pyplot as plt

  out = output_dir / "viz"
  out.mkdir(parents=True, exist_ok=True)

  variables = list(cfg_viz.variables) if cfg_viz.variables else list(targets.data_vars.keys())[:2]
  saved: list[str] = []
  max_steps = int(cfg_viz.max_steps)

  for var in variables:
    t = targets[var]
    p = predictions[var]
    # Drop batch if present.
    if "batch" in t.dims:
      t = t.isel(batch=0)
    if "batch" in p.dims:
      p = p.isel(batch=0)
    # Use ensemble mean if present.
    if "sample" in p.dims:
      p = p.mean(dim="sample")
    # Select first level if needed.
    if "level" in t.dims:
      t = t.isel(level=0)
    if "level" in p.dims:
      p = p.isel(level=0)
    if "time" in t.dims:
      t = t.isel(time=slice(0, max_steps))
    if "time" in p.dims:
      p = p.isel(time=slice(0, max_steps))

    # Only plot the first timestep for simplicity.
    if "time" in t.dims:
      t0 = t.isel(time=0)
    else:
      t0 = t
    if "time" in p.dims:
      p0 = p.isel(time=0)
    else:
      p0 = p

    e0 = p0 - t0
    data = {
        "targets": np.array(t0.data),
        "prediction": np.array(p0.data),
        "error": np.array(e0.data),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, arr) in zip(axes, data.items()):
      ax.set_title(title)
      ax.set_xticks([])
      ax.set_yticks([])
      im = ax.imshow(arr, origin="lower")
      fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(var)
    fig.tight_layout()

    fname = f"{var}.png".replace("/", "_")
    fpath = out / fname
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    saved.append(str(fpath))

  return saved


def _git_head_sha() -> Optional[str]:
  try:
    out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return out or None
  except Exception:
    return None


def _validate_graphcast_resolution(*, model_resolution: float, lon_size: int) -> None:
  # The notebooks treat resolution==0 as "unknown / random", allow it.
  if model_resolution == 0:
    return
  data_resolution = 360.0 / float(lon_size)
  if abs(model_resolution - data_resolution) > 1e-6:
    raise ValueError(
        "Model resolution does not match data resolution. "
        f"model_resolution={model_resolution}, data_resolution={data_resolution} (lon={lon_size})."
    )


def _validate_requested_lead_times_exist(*, dataset, requested_lead_times) -> None:
  """Raise a helpful error if requested lead times are not selectable."""
  import numpy as np
  import pandas as pd

  if "time" not in dataset.coords:
    raise ValueError("Dataset is missing a 'time' coordinate.")

  time_vals = dataset.coords["time"].values
  try:
    time_td = pd.to_timedelta(time_vals)
  except Exception as e:
    raise ValueError(
        "Expected dataset 'time' coordinate to be timedelta-like. "
        f"Got dtype={getattr(time_vals, 'dtype', None)!r}."
    ) from e

  req = [pd.Timedelta(x) for x in requested_lead_times]
  target_duration = max(req) if req else pd.Timedelta(0)
  shifted = time_td + target_duration - time_td[-1]

  shifted_set = set(shifted)
  missing = [t for t in req if t not in shifted_set]
  if not missing:
    return

  # Heuristics: infer timestep from positive diffs.
  diffs = np.diff(shifted.astype("timedelta64[ns]").astype(np.int64))
  diffs = diffs[diffs > 0]
  inferred_step = None
  if diffs.size:
    inferred_step = pd.to_timedelta(int(np.gcd.reduce(diffs)), unit="ns")

  positives = sorted([t for t in shifted if t > pd.Timedelta(0)])
  preview = ", ".join(str(t) for t in positives[:10])
  raise KeyError(
      "Requested target lead times are not present in the dataset time index. "
      f"Missing examples: {[str(t) for t in missing[:5]]}. "
      f"Inferred timestep: {inferred_step}. "
      f"Available positive lead times (preview): {preview}"
  )


def _ensure_batch_first(ds):
  """Ensure a leading batch dim exists and is first (size 1 if added)."""
  if ds is None:
    return None
  if "batch" not in ds.dims:
    ds = ds.expand_dims("batch", axis=0)
  return ds.transpose("batch", ...)


def _require_time_steps(ds, *, expected: int, name: str) -> None:
  if "time" not in ds.dims:
    raise ValueError(f"{name} is missing a 'time' dimension.")
  got = int(ds.sizes["time"])
  if got != expected:
    raise ValueError(f"{name} must have exactly {expected} timesteps, got {got}.")


def _expected_num_input_frames(*, dataset, input_duration: str) -> int:
  import pandas as pd

  if "time" not in dataset.coords:
    raise ValueError("Dataset is missing a 'time' coordinate.")
  time_td = pd.to_timedelta(dataset.coords["time"].values)
  step = _infer_uniform_timestep(time_td)
  dur = pd.Timedelta(str(input_duration))
  q, r = divmod(dur, step)
  if r != pd.Timedelta(0):
    raise ValueError(f"input_duration ({dur}) must be a multiple of dataset timestep ({step}).")
  return int(q)


def _first_chunk_template(*, targets_template, forcings, num_steps_per_chunk: int):
  """Match rollout.chunked_prediction_generator slicing for a compile warmup."""
  if num_steps_per_chunk < 1:
    raise ValueError("num_steps_per_chunk must be >= 1")
  if "time" not in targets_template.dims:
    return targets_template, forcings
  t_slice = slice(0, num_steps_per_chunk)
  targets_chunk_time = targets_template.time.isel(time=t_slice)
  tt = targets_template.isel(time=t_slice).assign_coords(time=targets_chunk_time).compute()
  ff = forcings.isel(time=t_slice).assign_coords(time=targets_chunk_time).compute() if forcings is not None else None
  return tt, ff


def _prepare_prediction_chunk_for_combine(chunk):
  """Ensure `sample` is a proper dimension for reliable concatenation."""
  if chunk is None:
    return chunk
  if "sample" in chunk.dims:
    return chunk
  if "sample" in chunk.coords:
    try:
      sample_val = int(chunk.coords["sample"].values)
    except Exception:
      sample_val = chunk.coords["sample"].values
    chunk = chunk.reset_coords("sample", drop=True)
    chunk = chunk.expand_dims({"sample": [sample_val]})
  return chunk


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  runtime = cfg.runtime
  if bool(runtime.get("xla_preallocate", False)):
    _maybe_set_env_var("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
  else:
    _maybe_set_env_var("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  _maybe_set_env_var("XLA_PYTHON_CLIENT_MEM_FRACTION", str(runtime.get("xla_mem_fraction", 0.85)))

  import jax
  import numpy as np
  import haiku as hk
  import xarray as xr

  from hydra.core.hydra_config import HydraConfig
  from hydra.utils import get_original_cwd

  from graphcast import checkpoint
  from graphcast import data_utils
  from graphcast import rollout

  base_dir = pathlib.Path(get_original_cwd())
  hydra_output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
  hydra_output_dir.mkdir(parents=True, exist_ok=True)

  # All persisted assets + outputs live under cfg.save_dir.
  save_dir = _as_path(getattr(cfg, "save_dir", None)) or pathlib.Path(".cache/graphcast_bench")
  if not save_dir.is_absolute():
    save_dir = base_dir / save_dir
  experiment_name = str(getattr(cfg, "experiment", {}).get("name", "default"))
  eval_root = save_dir / "eval" / experiment_name
  # Use Hydra's run directory structure for uniqueness, but store under save_dir.
  run_subdirs = hydra_output_dir.parts[-2:] if len(hydra_output_dir.parts) >= 2 else (hydra_output_dir.name,)
  run_dir = eval_root / "runs" / pathlib.Path(*run_subdirs)
  run_dir.mkdir(parents=True, exist_ok=True)
  asset_root = eval_root / "assets"
  asset_root.mkdir(parents=True, exist_ok=True)
  # Placeholder for future.
  (save_dir / "train" / experiment_name).mkdir(parents=True, exist_ok=True)
  _LOG.info("save_dir: %s", save_dir)
  _LOG.info("eval_root: %s", eval_root)
  _LOG.info("asset_root: %s", asset_root)
  _LOG.info("run_dir: %s", run_dir)

  model_kind = str(cfg.model.kind)
  data_kind = str(cfg.data.kind)
  preset_name = str(cfg.model.preset_name)

  _LOG.info("JAX platform: %s", jax.default_backend())
  _LOG.info("JAX devices: %s", [str(d) for d in jax.devices()])

  if bool(runtime.get("log_nvidia_smi", False)):
    try:
      _LOG.info("nvidia-smi:\n%s", subprocess.check_output(["nvidia-smi"], text=True))
    except Exception:
      pass

  mem_before = _nvidia_smi_query() if str(runtime.get("memory_probe")) == "nvidia_smi" else {}

  # Data loading (GCS only, anonymous client), per hydra.md.
  if not cfg.data.dataset.blob_relpath:
    raise ValueError("data.dataset.blob_relpath is required (GCS-only data loading).")
  dataset_blob_path = f"{str(cfg.data.gcs.dir_prefix)}{str(cfg.data.dataset.blob_relpath)}"
  dataset_dest = _asset_dest_path(
      asset_root=asset_root,
      dir_prefix=str(cfg.data.gcs.dir_prefix),
      blob_relpath=f"dataset/{pathlib.Path(str(cfg.data.dataset.blob_relpath)).name}",
  )
  dataset_path = _fetch_public_gcs_to_cache(
      bucket_name=str(cfg.data.gcs.bucket),
      blob_path=dataset_blob_path,
      dest_path=dataset_dest,
  )
  _LOG.info("Dataset source: gs://%s/%s", str(cfg.data.gcs.bucket), dataset_blob_path.lstrip("/"))
  _LOG.info("Dataset cached at: %s", dataset_path)
  example_batch = _load_netcdf(dataset_path)
  _log_dataset_overview("example_batch (raw)", example_batch)
  if "batch" in example_batch.dims:
    example_batch = example_batch.isel(batch=int(cfg.data.batch_index))
    _log_dataset_overview("example_batch (selected batch)", example_batch)
  max_time_frames = getattr(cfg.data.dataset, "max_time_frames", None)
  if max_time_frames not in (None, "null"):
    max_time_frames = int(max_time_frames)
    if "time" not in example_batch.dims:
      raise ValueError("data.dataset.max_time_frames was set but dataset has no 'time' dimension.")
    if max_time_frames < 3:
      raise ValueError("data.dataset.max_time_frames must be >= 3 (2 inputs + >=1 target).")
    if int(example_batch.sizes["time"]) < max_time_frames:
      raise ValueError(
          f"data.dataset.max_time_frames={max_time_frames} exceeds dataset time={int(example_batch.sizes['time'])}."
      )
    example_batch = example_batch.isel(time=slice(0, max_time_frames))
    _LOG.info("Sliced example_batch time to first %d frames", max_time_frames)
    _log_dataset_overview("example_batch (time-sliced)", example_batch)

  # Resolve checkpoint + stats.
  ckpt_path = resolve_asset(
      source=str(cfg.model.source),
      local_path=cfg.model.checkpoint.local_path,
      bucket_name=str(cfg.model.gcs.bucket),
      dir_prefix=str(cfg.model.gcs.dir_prefix),
      blob_relpath=cfg.model.checkpoint.blob_relpath,
      base_dir=base_dir,
      asset_root=asset_root,
  )

  if model_kind == "graphcast":
    from graphcast import autoregressive
    from graphcast import casting
    from graphcast import graphcast as graphcast_lib
    from graphcast import normalization

    with open(ckpt_path, "rb") as f:
      ckpt = checkpoint.load(f, graphcast_lib.CheckPoint)

    stats_diffs = _load_netcdf(
        pathlib.Path(
            resolve_asset(
                source=str(cfg.model.source),
                local_path=cfg.model.stats.diffs_stddev_by_level.local_path,
                bucket_name=str(cfg.model.gcs.bucket),
                dir_prefix=str(cfg.model.gcs.dir_prefix),
                blob_relpath=cfg.model.stats.diffs_stddev_by_level.blob_relpath,
                base_dir=base_dir,
                asset_root=asset_root,
            )
        )
    )
    stats_mean = _load_netcdf(
        pathlib.Path(
            resolve_asset(
                source=str(cfg.model.source),
                local_path=cfg.model.stats.mean_by_level.local_path,
                bucket_name=str(cfg.model.gcs.bucket),
                dir_prefix=str(cfg.model.gcs.dir_prefix),
                blob_relpath=cfg.model.stats.mean_by_level.blob_relpath,
                base_dir=base_dir,
                asset_root=asset_root,
            )
        )
    )
    stats_std = _load_netcdf(
        pathlib.Path(
            resolve_asset(
                source=str(cfg.model.source),
                local_path=cfg.model.stats.stddev_by_level.local_path,
                bucket_name=str(cfg.model.gcs.bucket),
                dir_prefix=str(cfg.model.gcs.dir_prefix),
                blob_relpath=cfg.model.stats.stddev_by_level.blob_relpath,
                base_dir=base_dir,
                asset_root=asset_root,
            )
        )
    )

    params = ckpt.params
    state: dict[str, Any] = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    target_lead_times = _parse_target_lead_times(cfg.data.target_lead_times, example_batch)
    _validate_requested_lead_times_exist(dataset=example_batch, requested_lead_times=target_lead_times)
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        input_variables=task_config.input_variables,
        target_variables=task_config.target_variables,
        forcing_variables=task_config.forcing_variables,
        pressure_levels=task_config.pressure_levels,
        input_duration=task_config.input_duration,
        target_lead_times=target_lead_times,
    )
    inputs = _ensure_batch_first(inputs)
    targets = _ensure_batch_first(targets)
    forcings = _ensure_batch_first(forcings)

    expected_inputs = _expected_num_input_frames(dataset=example_batch, input_duration=str(task_config.input_duration))
    _require_time_steps(inputs, expected=expected_inputs, name="inputs")

    if bool(cfg.data.validate_resolution_and_levels):
      _validate_graphcast_resolution(model_resolution=float(model_config.resolution), lon_size=int(inputs.sizes["lon"]))

    def construct_wrapped_graphcast():
      predictor = graphcast_lib.GraphCast(model_config, task_config)
      predictor = casting.Bfloat16Cast(predictor)
      predictor = normalization.InputsAndResiduals(
          predictor,
          diffs_stddev_by_level=stats_diffs,
          mean_by_level=stats_mean,
          stddev_by_level=stats_std,
      )
      predictor = autoregressive.Predictor(
          predictor, gradient_checkpointing=bool(cfg.model.graphcast.gradient_checkpointing)
      )
      return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
      predictor = construct_wrapped_graphcast()
      return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def apply_forward(rng, inputs, targets_template, forcings):
      preds, _ = run_forward.apply(params, state, rng, inputs, targets_template, forcings)
      return preds

    predictor_fn = jax.jit(apply_forward)

    # Warmup (compilation excluded from timed section).
    targets_template = targets * np.nan
    warm_targets_template, warm_forcings = _first_chunk_template(
        targets_template=targets_template,
        forcings=forcings,
        num_steps_per_chunk=int(cfg.benchmark.num_steps_per_chunk),
    )
    for _ in range(int(cfg.benchmark.warmup_steps)):
      warm = predictor_fn(jax.random.PRNGKey(int(cfg.seed)), inputs, warm_targets_template, warm_forcings)
      if bool(cfg.benchmark.block_until_ready):
        _block_until_ready(warm)

    # Timed runs.
    times_s: list[float] = []
    last_predictions = None
    for i in range(int(cfg.benchmark.timed_repeats)):
      start = time.perf_counter()
      preds = rollout.chunked_prediction(
          predictor_fn=predictor_fn,
          rng=jax.random.PRNGKey(int(cfg.seed)),
          inputs=inputs,
          targets_template=targets_template,
          forcings=forcings,
          num_steps_per_chunk=int(cfg.benchmark.num_steps_per_chunk),
      )
      if bool(cfg.benchmark.block_until_ready):
        _block_until_ready(preds)
      dt = time.perf_counter() - start
      _LOG.info("Repeat %d/%d: %.3fs", i + 1, int(cfg.benchmark.timed_repeats), dt)
      times_s.append(dt)
      last_predictions = preds

    assert last_predictions is not None
    metrics = _compute_metrics_graphcast(targets=targets, predictions=last_predictions, cfg_metrics=cfg.metrics)
    num_samples = 1.0

  elif model_kind == "gencast":
    from graphcast import gencast as gencast_lib
    from graphcast import nan_cleaning
    from graphcast import normalization
    from graphcast import xarray_jax

    with open(ckpt_path, "rb") as f:
      ckpt = checkpoint.load(f, gencast_lib.CheckPoint)

    stats_diffs = _load_netcdf(
        pathlib.Path(
            resolve_asset(
                source=str(cfg.model.source),
                local_path=cfg.model.stats.diffs_stddev_by_level.local_path,
                bucket_name=str(cfg.model.gcs.bucket),
                dir_prefix=str(cfg.model.gcs.dir_prefix),
                blob_relpath=cfg.model.stats.diffs_stddev_by_level.blob_relpath,
                base_dir=base_dir,
                asset_root=asset_root,
            )
        )
    )
    stats_mean = _load_netcdf(
        pathlib.Path(
            resolve_asset(
                source=str(cfg.model.source),
                local_path=cfg.model.stats.mean_by_level.local_path,
                bucket_name=str(cfg.model.gcs.bucket),
                dir_prefix=str(cfg.model.gcs.dir_prefix),
                blob_relpath=cfg.model.stats.mean_by_level.blob_relpath,
                base_dir=base_dir,
                asset_root=asset_root,
            )
        )
    )
    stats_std = _load_netcdf(
        pathlib.Path(
            resolve_asset(
                source=str(cfg.model.source),
                local_path=cfg.model.stats.stddev_by_level.local_path,
                bucket_name=str(cfg.model.gcs.bucket),
                dir_prefix=str(cfg.model.gcs.dir_prefix),
                blob_relpath=cfg.model.stats.stddev_by_level.blob_relpath,
                base_dir=base_dir,
                asset_root=asset_root,
            )
        )
    )
    stats_min = _load_netcdf(
        pathlib.Path(
            resolve_asset(
                source=str(cfg.model.source),
                local_path=cfg.model.stats.min_by_level.local_path,
                bucket_name=str(cfg.model.gcs.bucket),
                dir_prefix=str(cfg.model.gcs.dir_prefix),
                blob_relpath=cfg.model.stats.min_by_level.blob_relpath,
                base_dir=base_dir,
                asset_root=asset_root,
            )
        )
    )

    params = ckpt.params
    state: dict[str, Any] = {}
    task_config = ckpt.task_config
    sampler_config = ckpt.sampler_config
    noise_config = ckpt.noise_config
    noise_encoder_config = ckpt.noise_encoder_config
    denoiser_architecture_config = ckpt.denoiser_architecture_config

    if bool(cfg.model.gencast.force_gpu_compatible_attention) and jax.local_devices()[0].platform != "tpu":
      cfg_st = denoiser_architecture_config.sparse_transformer_config
      cfg_st.attention_type = "triblockdiag_mha"
      cfg_st.mask_type = "full"
      _LOG.info("Non-TPU backend detected; forcing attention_type=triblockdiag_mha and mask_type=full")

    target_lead_times = _parse_target_lead_times(cfg.data.target_lead_times, example_batch)
    _validate_requested_lead_times_exist(dataset=example_batch, requested_lead_times=target_lead_times)
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        input_variables=task_config.input_variables,
        target_variables=task_config.target_variables,
        forcing_variables=task_config.forcing_variables,
        pressure_levels=task_config.pressure_levels,
        input_duration=task_config.input_duration,
        target_lead_times=target_lead_times,
    )
    inputs = _ensure_batch_first(inputs)
    targets = _ensure_batch_first(targets)
    forcings = _ensure_batch_first(forcings)

    expected_inputs = _expected_num_input_frames(dataset=example_batch, input_duration=str(task_config.input_duration))
    _require_time_steps(inputs, expected=expected_inputs, name="inputs")

    def construct_wrapped_gencast():
      predictor = gencast_lib.GenCast(
          sampler_config=sampler_config,
          task_config=task_config,
          denoiser_architecture_config=denoiser_architecture_config,
          noise_config=noise_config,
          noise_encoder_config=noise_encoder_config,
      )
      predictor = normalization.InputsAndResiduals(
          predictor,
          diffs_stddev_by_level=stats_diffs,
          mean_by_level=stats_mean,
          stddev_by_level=stats_std,
      )
      predictor = nan_cleaning.NaNCleaner(
          predictor=predictor,
          reintroduce_nans=True,
          fill_value=stats_min,
          var_to_clean="sea_surface_temperature",
      )
      return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
      predictor = construct_wrapped_gencast()
      return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def apply_forward(rng, inputs, targets_template, forcings):
      preds, _ = run_forward.apply(params, state, rng, inputs, targets_template, forcings)
      return preds

    run_forward_jitted = jax.jit(apply_forward)

    targets_template = targets * np.nan
    ensemble_size = int(cfg.model.gencast.ensemble_size)
    num_samples = float(ensemble_size)

    # Warmup compile.
    warm_targets_template, warm_forcings = _first_chunk_template(
        targets_template=targets_template,
        forcings=forcings,
        num_steps_per_chunk=int(cfg.benchmark.num_steps_per_chunk),
    )
    for _ in range(int(cfg.benchmark.warmup_steps)):
      warm = run_forward_jitted(jax.random.PRNGKey(int(cfg.seed)), inputs, warm_targets_template, warm_forcings)
      if bool(cfg.benchmark.block_until_ready):
        _block_until_ready(warm)

    # Select predictor function for rollout. Optionally pmap across GPUs.
    if bool(cfg.runtime.use_pmap_for_gencast_ensemble):
      devices = jax.local_devices()
      if ensemble_size % len(devices) != 0:
        raise ValueError(
            f"ensemble_size ({ensemble_size}) must be a multiple of num_devices ({len(devices)}) when pmap is enabled."
        )
      predictor_fn = xarray_jax.pmap(run_forward_jitted, dim="sample", devices=devices)
      pmap_devices = devices
    else:
      predictor_fn = run_forward_jitted
      pmap_devices = None

    # Timed runs (end-to-end rollout).
    times_s: list[float] = []
    last_predictions = None
    for i in range(int(cfg.benchmark.timed_repeats)):
      rng = jax.random.PRNGKey(int(cfg.seed))
      rngs = np.stack([jax.random.fold_in(rng, k) for k in range(ensemble_size)], axis=0)

      start = time.perf_counter()
      chunks = []
      for chunk in rollout.chunked_prediction_generator_multiple_runs(
          predictor_fn=predictor_fn,
          rngs=rngs,
          inputs=inputs,
          targets_template=targets_template,
          forcings=forcings,
          num_steps_per_chunk=int(cfg.benchmark.num_steps_per_chunk),
          num_samples=ensemble_size,
          pmap_devices=pmap_devices,
      ):
        chunks.append(_prepare_prediction_chunk_for_combine(chunk))
      preds = xr.combine_by_coords(chunks)
      if bool(cfg.benchmark.block_until_ready):
        _block_until_ready(preds)
      dt = time.perf_counter() - start
      _LOG.info("Repeat %d/%d: %.3fs", i + 1, int(cfg.benchmark.timed_repeats), dt)
      times_s.append(dt)
      last_predictions = preds

    assert last_predictions is not None
    # Convert any device arrays / wrapped JAX arrays to host numpy arrays
    # before computing metrics.
    last_predictions = jax.device_get(last_predictions)
    metrics = _compute_metrics_gencast(targets=targets, predictions=last_predictions, cfg_metrics=cfg.metrics)

  else:
    raise ValueError(f"Unsupported model.kind={model_kind!r}")

  mem_after = _nvidia_smi_query() if str(runtime.get("memory_probe")) == "nvidia_smi" else {}

  median_s = _percentile(times_s, 50)
  p95_s = _percentile(times_s, 95)
  throughput = (num_samples / median_s) if median_s and median_s > 0 else float("nan")

  _LOG.info(
      "Summary: model=%s preset=%s data=%s median=%.3fs p95=%.3fs throughput=%.3f samples/s rmse=%.6f nrmse=%.6f",
      model_kind,
      preset_name,
      data_kind,
      median_s,
      p95_s,
      throughput,
      float(metrics["overall"].get("rmse", float("nan"))),
      float(metrics["overall"].get("nrmse", float("nan"))),
  )

  viz_paths = _maybe_save_visualizations(
      output_dir=run_dir, targets=targets, predictions=last_predictions, cfg_viz=cfg.visualization
  )

  results = {
      "run_metadata": {
          "timestamp_utc": _utc_now_iso(),
          "model_kind": model_kind,
          "preset_name": preset_name,
          "data_kind": data_kind,
          "seed": int(cfg.seed),
          "jax_platform": jax.default_backend(),
          "jax_devices": [str(d) for d in jax.devices()],
          "versions": {
              "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
              "git_head_sha": _git_head_sha(),
          },
      },
      "timing": {
          "warmup_steps": int(cfg.benchmark.warmup_steps),
          "timed_repeats": int(cfg.benchmark.timed_repeats),
          "repeat_seconds": times_s,
          "median_seconds": median_s,
          "p95_seconds": p95_s,
          "throughput_samples_per_second": throughput,
      },
      "memory": {
          "probe": str(runtime.get("memory_probe")),
          "before": mem_before,
          "after": mem_after,
      },
      "metrics": metrics,
      "artifacts": {"visualizations": viz_paths},
      "config": OmegaConf.to_container(cfg, resolve=True),
  }

  if bool(cfg.output.write_json):
    out_path = run_dir / str(cfg.output.json_filename)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    _LOG.info("Wrote %s", out_path)

  # Save resolved config snapshot alongside results for reproducibility.
  try:
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
  except Exception:
    pass


if __name__ == "__main__":
  main()
