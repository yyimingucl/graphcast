# Benchmarking GenCast vs GraphCast (GPU, Hydra)

This repo contains example benchmark code to measure **inference latency/throughput** and **forecast accuracy** for:

- **GraphCast** (deterministic autoregressive rollout)
- **GenCast** (diffusion-based ensemble rollout)

All knobs are configured via Hydra YAML files in `config/`.

## 1) Install (GPU)

1. Create and activate an environment.
2. Determine your CUDA major version:

```bash
nvidia-smi
```

3. Install JAX with CUDA wheels matching your CUDA major version:

- If `nvidia-smi` reports CUDA **12.x**, install JAX **CUDA12** wheels.
- If it reports CUDA **11.x**, install JAX **CUDA11** wheels.

Refer to JAX’s official installation docs for the exact command for your setup.

4. Install this repo:

```bash
pip install -e .
```

5. Optional (only if using `gcs_uri` download mode):

```bash
pip install -e .[gcs]
```

## 2) Acquire data + checkpoints

### Data (GCS only)

Per `hydra.md`, **ERA5/HRES test data must be fetched from the public `dm_graphcast` GCS bucket using an anonymous client**. The script downloads the NetCDF to `io.cache_dir` and then loads it from disk.

Configure the dataset via:

- `data.gcs.bucket` (default: `dm_graphcast`)
- `data.gcs.dir_prefix` (default: `gencast/`)
- `data.dataset.blob_relpath` (relative to `dir_prefix`, e.g. `dataset/source-era5_...nc`)

### Model checkpoints/stats (GCS or local)

Model assets support a Hydra toggle:

- `model.source=local`: use `*.local_path`
- `model.source=gcs`: use `model.gcs.{bucket,dir_prefix}` + `*.blob_relpath`

GCS downloads are anonymous and cached under `io.cache_dir`.

## 3) Run benchmarks

Tip: set `experiment.name=<name>` to group related runs under the same folder in `save_dir/eval/<name>/...`.

### GraphCast
```bash
python main.py model=graphcast data=era5 \\
  data.dataset.blob_relpath="dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc" \\
  model.source=local \\
  model.checkpoint.local_path=/path/to/GraphCast_small.npz \\
  model.stats.diffs_stddev_by_level.local_path=/path/to/diffs_stddev_by_level.nc \\
  model.stats.mean_by_level.local_path=/path/to/mean_by_level.nc \\
  model.stats.stddev_by_level.local_path=/path/to/stddev_by_level.nc \\
  model.gcs.dir_prefix="graphcast/"
```

### GenCast (GPU-compatible attention override)
```bash
python main.py model=gencast data=era5 \\
  data.dataset.blob_relpath="dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc" \\
  model.source=local \\
  model.checkpoint.local_path=/path/to/GenCast_1p0deg_Mini_2019.npz \\
  model.stats.diffs_stddev_by_level.local_path=/path/to/diffs_stddev_by_level.nc \\
  model.stats.mean_by_level.local_path=/path/to/mean_by_level.nc \\
  model.stats.stddev_by_level.local_path=/path/to/stddev_by_level.nc \\
  model.stats.min_by_level.local_path=/path/to/min_by_level.nc \\
  model.gencast.ensemble_size=8
```

Notes:
- By default `config/data/era5.yaml` uses `data.target_lead_times.mode=all_but_inputs` with `input_frames=2`, which mirrors `gencast_mini_demo.ipynb` (“all but the input frames”).

### Multirun comparison
```bash
python main.py -m model=graphcast,gencast
```

## 4) Outputs

Hydra still creates an `outputs/.../` directory for each run, but this benchmark writes *its* artifacts under `save_dir` (see `config/config.yaml`):

- `save_dir/eval/<experiment.name>/assets/{params,stats,dataset}/...` (downloaded + cached from GCS)
- `save_dir/eval/<experiment.name>/runs/<hydra date>/<hydra time>/results.json`
- `save_dir/eval/<experiment.name>/runs/<hydra date>/<hydra time>/viz/*.png` (if enabled)
- `save_dir/eval/<experiment.name>/runs/<hydra date>/<hydra time>/config.yaml` (resolved Hydra config)

To validate the JSON contract, see `benchmark_results_schema.json`.


python main.py model=gencast data=era5   model.source=gcs   'model.checkpoint.blob_relpath= "params/GenCast 1p0deg Mini <2019.npz"'  data.dataset.blob_relpath="dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc"   model.gencast.ensemble_size=2   benchmark.warmup_steps=1 benchmark.timed_repeats=1
