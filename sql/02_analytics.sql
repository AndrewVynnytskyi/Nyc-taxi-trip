-- =============================================================================
-- NYC Taxi Trip Analytics -- exploratory and monitoring queries.
--
-- Run individually from DataGrip (Ctrl+Enter on a statement). Each block is
-- self-contained and read-only.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. ETL health check: how much of the raw feed survived cleaning?
-- -----------------------------------------------------------------------------
SELECT (SELECT count(*) FROM public.nyc_raw_taxi)                       AS raw_rows,
       (SELECT count(*) FROM public.nyc_clean_taxi)                     AS clean_rows,
       round(100.0 * (SELECT count(*) FROM public.nyc_clean_taxi)
                   / NULLIF((SELECT count(*) FROM public.nyc_raw_taxi), 0), 2) AS retention_pct;

-- -----------------------------------------------------------------------------
-- 2. Target distribution: is the trip-duration tail as heavy as we assumed?
--    (Motivates data.log_transform_target=true in configs/train.yaml.)
-- -----------------------------------------------------------------------------
SELECT count(*)                                                                       AS trips,
       round(avg(trip_duration_minutes)::numeric, 2)                                  AS mean_minutes,
       round(stddev_samp(trip_duration_minutes)::numeric, 2)                          AS stddev_minutes,
       round(percentile_cont(0.50) WITHIN GROUP (ORDER BY trip_duration_minutes)::numeric, 2) AS p50,
       round(percentile_cont(0.90) WITHIN GROUP (ORDER BY trip_duration_minutes)::numeric, 2) AS p90,
       round(percentile_cont(0.99) WITHIN GROUP (ORDER BY trip_duration_minutes)::numeric, 2) AS p99,
       round(max(trip_duration_minutes)::numeric, 2)                                  AS max_minutes
FROM public.nyc_clean_taxi;

-- -----------------------------------------------------------------------------
-- 3. Demand and speed profile by hour of day.
-- -----------------------------------------------------------------------------
SELECT pickup_hour,
       count(*)                                                       AS trips,
       round(avg(trip_duration_minutes)::numeric, 2)                  AS avg_duration_min,
       round(avg(trip_distance)::numeric, 2)                          AS avg_distance_mi,
       round(avg(fare_amount)::numeric, 2)                            AS avg_fare_usd,
       round(avg(60.0 * trip_distance / NULLIF(trip_duration_minutes, 0))::numeric, 2)
                                                                      AS avg_speed_mph
FROM public.nyc_clean_taxi
GROUP BY pickup_hour
ORDER BY pickup_hour;

-- -----------------------------------------------------------------------------
-- 4. Weekday x hour demand heatmap (feeds the notebook's EDA figure).
-- -----------------------------------------------------------------------------
SELECT pickup_dayofweek,
       to_char(date '2015-01-05' + pickup_dayofweek, 'Dy')            AS day_name,
       pickup_hour,
       count(*)                                                       AS trips,
       round(avg(trip_duration_minutes)::numeric, 2)                  AS avg_duration_min
FROM public.nyc_clean_taxi
GROUP BY pickup_dayofweek, pickup_hour
ORDER BY pickup_dayofweek, pickup_hour;

-- -----------------------------------------------------------------------------
-- 5. Fare-per-mile bands: sanity check on pricing outliers that survived the
--    ETL filters.
-- -----------------------------------------------------------------------------
WITH priced AS (SELECT fare_amount / NULLIF(trip_distance, 0) AS usd_per_mile
                FROM public.nyc_clean_taxi)
SELECT width_bucket(usd_per_mile, 0, 20, 10) AS bucket,
       count(*)                              AS trips,
       round(min(usd_per_mile)::numeric, 2)  AS min_usd_per_mile,
       round(max(usd_per_mile)::numeric, 2)  AS max_usd_per_mile
FROM priced
WHERE usd_per_mile IS NOT NULL
GROUP BY bucket
ORDER BY bucket;

-- -----------------------------------------------------------------------------
-- 6. Baseline the model must beat: predict each trip's duration as the mean of
--    its (dayofweek, hour) bucket. Reports the RMSE of that heuristic so the
--    neural network's val_rmse has a meaningful reference point.
-- -----------------------------------------------------------------------------
WITH bucket_mean AS (SELECT pickup_dayofweek,
                            pickup_hour,
                            avg(trip_duration_minutes) AS predicted_minutes
                     FROM public.nyc_clean_taxi
                     GROUP BY pickup_dayofweek, pickup_hour)
SELECT round(sqrt(avg(power(t.trip_duration_minutes - b.predicted_minutes, 2)))::numeric, 3)
           AS baseline_rmse_minutes,
       round(avg(abs(t.trip_duration_minutes - b.predicted_minutes))::numeric, 3)
           AS baseline_mae_minutes
FROM public.nyc_clean_taxi t
         JOIN bucket_mean b
              ON b.pickup_dayofweek = t.pickup_dayofweek
                  AND b.pickup_hour = t.pickup_hour;

-- -----------------------------------------------------------------------------
-- 7. Null audit across the clean table -- should return all zeros.
-- -----------------------------------------------------------------------------
SELECT count(*) FILTER (WHERE tpep_pickup_datetime IS NULL)  AS null_pickup,
       count(*) FILTER (WHERE tpep_dropoff_datetime IS NULL) AS null_dropoff,
       count(*) FILTER (WHERE fare_amount IS NULL)           AS null_fare,
       count(*) FILTER (WHERE trip_duration_minutes IS NULL) AS null_duration,
       count(*) FILTER (WHERE passenger_count IS NULL)       AS null_passengers
FROM public.nyc_clean_taxi;
