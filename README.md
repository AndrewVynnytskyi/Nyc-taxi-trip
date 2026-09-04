# NYC Taxi Trip Analytics

Production-grade pipeline that ingests NYC Yellow Taxi trip records into PostgreSQL
and trains a PyTorch regressor to predict **trip duration in minutes**.

## Architecture

```
configs/train.yaml          Single source of truth: every hyperparameter, path and business rule
src/nyc_taxi/
  config.py                 Config loading, override merging, validation
  utils.py                  Logging, seeding, device resolution
  database.py               Pooled SQLAlchemy engine, safe identifier handling
  etl.py                    TaxiETLPipeline: chunked CSV -> validated -> PostgreSQL
  data.py                   TaxiTripDataset / TaxiDataModule: features, splits, scalers, loaders
  model.py                  TripDurationMLP: residual MLP, config-driven architecture
  train.py                  Trainer: fit / evaluate / predict, AMP, early stopping, W&B
notebooks/
  00_etl_pipeline.ipynb     Thin orchestration of the ETL
  01_baseline_training.ipynb Thin orchestration of training + presentation-ready figures
sql/
  00_init_database.sql      Database and least-privilege role provisioning
  01_schema.sql             Tables, CHECK constraints mirroring etl.filters, indexes
  02_analytics.sql          EDA, data-quality audits and the heuristic RMSE baseline
tests/test_pipeline.py      Config, ETL transform, target round-trip, model contract
```

Notebooks contain **no** training loops and **no** hyperparameters — they import from
`nyc_taxi`, load the config, and call `trainer.fit()`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebooks,data,dev]"
pre-commit install

cp .env.example .env          # then set POSTGRES_PASSWORD
docker compose up -d postgres # or point .env at an existing instance

nyc-taxi-etl                  # CSV -> nyc_raw_taxi / nyc_clean_taxi
nyc-taxi-train                # train, evaluate, checkpoint
```

Open `sql/01_schema.sql` and `sql/02_analytics.sql` in DataGrip against the
`nyc_taxi` database to apply constraints and run the analytics queries.

## Configuration

Nothing is hardcoded. Override any value from the CLI or a notebook:

```bash
nyc-taxi-train optimizer.lr=3e-4 model.hidden_dims=[512,256] training.max_epochs=50
```

```python
cfg = load_config(overrides=["training.max_epochs=15"])
```

Secrets and machine-specific paths come from the environment (`.env.example` lists them).

## Docker

```bash
docker build -t nyc-taxi:latest .                       # CUDA 12.1 runtime
docker build --build-arg BASE_IMAGE=python:3.11-slim-bookworm \
             --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
             -t nyc-taxi:cpu .                          # CPU only
docker run --rm --gpus all --env-file .env nyc-taxi:latest nyc-taxi-train
```

Multi-stage build, dependency layer cached separately from source, non-root `mluser`,
`tini` as PID 1, healthcheck on the torch import.

## Quality gates

```bash
black --check --line-length=100 src/ tests/
isort --check-only --profile=black src/ tests/
flake8 src/ tests/
mypy src/            # strict mode
pytest
```

All five pass on the committed tree; `.pre-commit-config.yaml` runs them on every commit
and additionally strips notebook outputs and lints notebook code cells via `nbQA`.

## Model

`TripDurationMLP` — a residual MLP over 7 features (trip distance, fare, passenger count,
and sine/cosine encodings of pickup hour and day of week). The target is
`log1p`-transformed and standardised; metrics are inverse-transformed so RMSE and MAE are
reported in **minutes**. The reference to beat is the `(dayofweek, hour)` bucket mean,
computed by query 6 in `sql/02_analytics.sql` and reproduced in the notebook.
