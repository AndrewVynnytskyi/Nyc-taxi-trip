-- =============================================================================
-- One-time provisioning. Run as a superuser against the `postgres` database
-- (in DataGrip: connect to `postgres`, then execute this file).
--
-- The password below is a placeholder: set the real one through the
-- POSTGRES_PASSWORD environment variable and keep it out of version control.
-- =============================================================================

CREATE DATABASE nyc_taxi
    WITH ENCODING 'UTF8'
         LC_COLLATE = 'en_US.UTF-8'
         LC_CTYPE = 'en_US.UTF-8'
         TEMPLATE = template0;

COMMENT ON DATABASE nyc_taxi IS 'NYC Yellow Taxi trip analytics: raw landing + clean training tables.';

-- Least-privilege application role used by the ETL and the training pipeline.
-- Replace the placeholder before executing.
CREATE ROLE nyc_taxi_app WITH LOGIN PASSWORD 'change-me-via-POSTGRES_PASSWORD';

GRANT CONNECT ON DATABASE nyc_taxi TO nyc_taxi_app;

-- The remaining grants must run while connected to `nyc_taxi`:
--   \c nyc_taxi
--   GRANT USAGE, CREATE ON SCHEMA public TO nyc_taxi_app;
--   ALTER DEFAULT PRIVILEGES IN SCHEMA public
--       GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO nyc_taxi_app;
