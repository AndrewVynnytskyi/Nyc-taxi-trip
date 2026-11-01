# NYC Taxi Trip Analytics

[![CI](https://github.com/AndrewVynnytskyi/Nyc-taxi-trip/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewVynnytskyi/Nyc-taxi-trip/actions/workflows/ci.yml)

Production-grade pipeline that ingests NYC Yellow Taxi trip records into PostgreSQL
and trains a PyTorch regressor on a configurable per-trip target — `fare_amount` (USD)
by default, or `trip_duration_minutes`.

## Measured results

NYC TLC January 2015, first 2M rows ingested → 1,984,004 clean trips (99.2% retention),
split 1,388,802 / 297,601 / 297,601. Identical 6-feature input for both targets; CPU
training with early stopping. Reproduce with `python scripts/compare_targets.py`.

Every RMSE below is on the **same held-out test split**. All baselines are fitted on the
train split only (`nyc_taxi.baseline`):

- **bucket**: the mean target of the trip's `(dayofweek, hour)` bucket;
- **linear (distance)**: linear regression on `trip_distance` alone;
- **linear (all)**: linear regression on the MLP's exact 6 input features.

"Improvement" is the MLP's RMSE reduction over **linear (distance)**.

| target | σ | bucket RMSE | linear (distance) RMSE | linear (all) RMSE | MLP RMSE | improvement vs linear (distance) | MLP R² |
|---|---|---|---|---|---|---|---|
| `fare_amount` (USD) | 10.49 | 10.431 | 3.164 | 3.150 | **3.060** | 3.3% | **0.9150** |
| `trip_duration_minutes` | 8.81 | 8.742 | 5.269 | 5.211 | **4.195** | 20.4% | **0.7745** |

`fare_amount` is the easier target of the two: the 2015 NYC meter is a published function
of distance and waiting time, so most of the signal is already in `trip_distance`.
Trip duration is harder because it depends on effective speed, which varies with traffic
in ways the current feature set cannot observe — that gap is where geography (`PULocationID`
/ `DOLocationID`) would pay off most.

## Architecture

```
configs/train.yaml          Single source of truth: every hyperparameter, path and business rule
src/nyc_taxi/
  config.py                 Config loading, override merging, validation
  utils.py                  Logging, seeding, device resolution
  database.py               Pooled SQLAlchemy engine, safe identifier handling
  etl.py                    TaxiETLPipeline: chunked parquet (default) or CSV -> validated -> PostgreSQL
  data.py                   TaxiTripDataset / TaxiDataModule: features, splits, scalers, loaders
  model.py                  TripDurationMLP: residual MLP, config-driven architecture
  train.py                  Trainer: fit / evaluate / predict, AMP, early stopping, W&B
  baseline.py               Bucket-mean and linear baselines, fitted on train, scored on test
notebooks/
  00_etl_pipeline.ipynb     Thin orchestration of the ETL
  01_baseline_training.ipynb Thin orchestration of training + presentation-ready figures
scripts/
  run_pipeline.sh           End-to-end run: environment checks, ETL, training
  compare_targets.py        Benchmarks fare_amount vs trip_duration_minutes against the baselines
sql/
  00_init_database.sql      Database and least-privilege role provisioning
  01_schema.sql             Tables, CHECK constraints mirroring etl.filters, indexes
  02_analytics.sql          EDA, data-quality audits and an in-sample bucket-mean figure
tests/
  test_pipeline.py          Config, ETL transform, data splits, target round-trip, model contract
  test_baseline.py          Baseline correctness and train/test separation, no target leakage
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

nyc-taxi-etl                  # TLC parquet -> nyc_raw_taxi / nyc_clean_taxi
nyc-taxi-train                # train, evaluate, checkpoint
```

Open `sql/01_schema.sql` and `sql/02_analytics.sql` in DataGrip against the
`nyc_taxi` database to apply constraints and run the analytics queries.

## Pipeline

`scripts/run_pipeline.sh` runs the whole thing in one go: it checks that `.env` exists and
the console scripts are installed, runs the ETL, then trains and evaluates on the test
split. It can be called from any directory.

```bash
scripts/run_pipeline.sh                              # ETL + training
scripts/run_pipeline.sh --skip-etl                   # train on data already in PostgreSQL
scripts/run_pipeline.sh training.max_epochs=20       # key=value overrides go to training
scripts/run_pipeline.sh --help
```

It needs a `.env` in the repository root, a running PostgreSQL instance and the package
installed (`pip install -e ".[dev]"`). On failure it reports which step broke and exits
non-zero.

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

All five pass on the committed tree and run in CI on every push and pull request.
`.pre-commit-config.yaml` runs them on every commit and additionally lints notebook code
cells via `nbQA`; notebook outputs are deliberately kept, since the executed figures and
metric tables are part of the deliverable.

## Model and target

`TripDurationMLP` — a residual MLP over **6 features**: trip distance, passenger count,
and sine/cosine encodings of pickup hour and day of week. The target is `log1p`-transformed
and standardised; metrics are inverse-transformed, so RMSE and MAE are reported in the
target's own unit (`data.target_unit`).

`fare_amount` and `trip_duration_minutes` are **both** excluded from the features:
whichever one is not the target is unknown at pickup time, so using it as an input would
leak information the model would not have when serving a live quote.

Switching targets is a single override — the feature set is identical, so the runs are
directly comparable:

```bash
python scripts/compare_targets.py            # benchmarks both, side by side
nyc-taxi-train data.target=trip_duration_minutes data.target_unit=min
```

The reference to beat is the distance-only linear regression from `nyc_taxi.baseline`,
fitted on the train split and scored on the same test split as the MLP. The notebook and
the comparison script both use it. Query 6 in `sql/02_analytics.sql` is an in-sample,
descriptive bucket-mean figure, not a benchmark.

## Known limitations

- **The signal is mostly `trip_distance`.** Passenger count and pickup time add little, so
  the MLP's margin over a distance-only linear regression is the honest measure of what it
  learns.
- **No pickup/dropoff geography.** `PULocationID` / `DOLocationID` are not used, so airport
  flat fares, borough effects and route-specific traffic are invisible to the model.
- **One month of data.** Training and evaluation use January 2015 only; seasonality, tariff
  changes and drift across months or years are untested.
- **The linear baseline can predict negative values** for very short trips whenever its
  fitted intercept is negative: it is an unconstrained straight line, not forced through a
  minimum fare or duration. Check the logged intercept for each target.
- **The SQL bucket baseline is in-sample.** Query 6 in `sql/02_analytics.sql` scores bucket
  means on the rows they were computed from; it is descriptive only. The train/test version
  lives in `nyc_taxi.baseline`.

## License

Released under the [MIT License](LICENSE).
