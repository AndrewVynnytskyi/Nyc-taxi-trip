-- =============================================================================
-- NYC Taxi Trip Analytics -- physical schema.
--
-- Target : PostgreSQL 14+
-- Tooling: written to be run top-to-bottom from DataGrip (JetBrains) against
--          the `nyc_taxi` database. Idempotent: safe to re-run.
--
-- The ETL (`nyc-taxi-etl`) recreates `nyc_raw_taxi` and `nyc_clean_taxi` via
-- pandas `to_sql`, which infers column types. Running this script AFTER the ETL
-- adds the constraints, indexes and comments that pandas cannot express.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS public;

-- -----------------------------------------------------------------------------
-- Raw landing table -- a faithful, unfiltered copy of the source CSV.
-- Created by the ETL; this definition documents the contract and is used when
-- provisioning an empty database ahead of the first load.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.nyc_raw_taxi
(
    "VendorID"              INTEGER,
    tpep_pickup_datetime    TIMESTAMP        NOT NULL,
    tpep_dropoff_datetime   TIMESTAMP        NOT NULL,
    passenger_count         INTEGER,
    trip_distance           DOUBLE PRECISION,
    pickup_longitude        DOUBLE PRECISION,
    pickup_latitude         DOUBLE PRECISION,
    "RateCodeID"            INTEGER,
    store_and_fwd_flag      TEXT,
    dropoff_longitude       DOUBLE PRECISION,
    dropoff_latitude        DOUBLE PRECISION,
    payment_type            INTEGER,
    fare_amount             DOUBLE PRECISION,
    extra                   DOUBLE PRECISION,
    mta_tax                 DOUBLE PRECISION,
    tip_amount              DOUBLE PRECISION,
    tolls_amount            DOUBLE PRECISION,
    improvement_surcharge   DOUBLE PRECISION,
    total_amount            DOUBLE PRECISION
);

COMMENT ON TABLE public.nyc_raw_taxi IS
    'Unmodified NYC Yellow Taxi trip records as extracted from the source CSV.';

-- -----------------------------------------------------------------------------
-- Clean analytical table -- the training set consumed by nyc_taxi.data.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.nyc_clean_taxi
(
    tpep_pickup_datetime    TIMESTAMP        NOT NULL,
    tpep_dropoff_datetime   TIMESTAMP        NOT NULL,
    passenger_count         INTEGER          NOT NULL,
    trip_distance           DOUBLE PRECISION NOT NULL,
    trip_duration_minutes   DOUBLE PRECISION NOT NULL,
    pickup_hour             SMALLINT         NOT NULL,
    pickup_dayofweek        SMALLINT         NOT NULL,
    fare_amount             DOUBLE PRECISION NOT NULL
);

COMMENT ON TABLE public.nyc_clean_taxi IS
    'Validated, filtered and feature-enriched trips. One row per model training sample.';
COMMENT ON COLUMN public.nyc_clean_taxi.trip_duration_minutes IS
    'Regression target: dropoff - pickup, in minutes.';
COMMENT ON COLUMN public.nyc_clean_taxi.pickup_hour IS
    'Hour of day of pickup, 0-23. Encoded as a sin/cos pair before training.';
COMMENT ON COLUMN public.nyc_clean_taxi.pickup_dayofweek IS
    'Day of week of pickup, 0=Monday .. 6=Sunday.';

-- -----------------------------------------------------------------------------
-- Data-quality constraints mirroring `etl.filters` in configs/train.yaml.
-- Added post-load so a violated invariant fails loudly instead of silently
-- poisoning the training set.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_clean_duration_positive') THEN
        ALTER TABLE public.nyc_clean_taxi
            ADD CONSTRAINT chk_clean_duration_positive
                CHECK (trip_duration_minutes > 0 AND trip_duration_minutes <= 180);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_clean_fare_positive') THEN
        ALTER TABLE public.nyc_clean_taxi
            ADD CONSTRAINT chk_clean_fare_positive
                CHECK (fare_amount > 0 AND fare_amount <= 500);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_clean_passengers') THEN
        ALTER TABLE public.nyc_clean_taxi
            ADD CONSTRAINT chk_clean_passengers
                CHECK (passenger_count BETWEEN 1 AND 6);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_clean_distance') THEN
        ALTER TABLE public.nyc_clean_taxi
            ADD CONSTRAINT chk_clean_distance
                CHECK (trip_distance > 0 AND trip_distance <= 100);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_clean_pickup_hour') THEN
        ALTER TABLE public.nyc_clean_taxi
            ADD CONSTRAINT chk_clean_pickup_hour
                CHECK (pickup_hour BETWEEN 0 AND 23 AND pickup_dayofweek BETWEEN 0 AND 6);
    END IF;
END
$$;

-- -----------------------------------------------------------------------------
-- Indexes supporting the analytics queries in 02_analytics.sql.
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_clean_pickup_datetime
    ON public.nyc_clean_taxi (tpep_pickup_datetime);
CREATE INDEX IF NOT EXISTS idx_clean_pickup_hour
    ON public.nyc_clean_taxi (pickup_hour);
CREATE INDEX IF NOT EXISTS idx_clean_hour_dow
    ON public.nyc_clean_taxi (pickup_dayofweek, pickup_hour);
CREATE INDEX IF NOT EXISTS idx_raw_pickup_datetime
    ON public.nyc_raw_taxi (tpep_pickup_datetime);

ANALYZE public.nyc_clean_taxi;
