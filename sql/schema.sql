-- Database layout for the building energy forecast.
-- DEV ONLY: the DROP lines wipe any existing data so the loader can be re-run.

DROP TABLE IF EXISTS forecasts;
DROP TABLE IF EXISTS readings;
DROP TABLE IF EXISTS buildings;

-- One row per building. `scale` is the building's mean load over the training
-- period; the model divides every reading by it.
CREATE TABLE buildings (
    building_id TEXT             PRIMARY KEY,
    site_id     TEXT             NOT NULL,
    scale       DOUBLE PRECISION NOT NULL
);

-- One row per building per hour. A masked (dead) hour is stored as NULL.
-- One building has many readings: building_id points back to `buildings`.
CREATE TABLE readings (
    building_id TEXT             NOT NULL REFERENCES buildings (building_id),
    ts          TIMESTAMP        NOT NULL,
    kwh         DOUBLE PRECISION,
    PRIMARY KEY (building_id, ts)
);

-- Every forecast the API has served, one row per building per target hour, in kWh.
-- Monitoring joins this with `readings` once the real values exist. Serving the
-- same hour again overwrites the row (the key is building + hour).
CREATE TABLE forecasts (
    building_id  TEXT             NOT NULL REFERENCES buildings (building_id),
    ts           TIMESTAMP        NOT NULL,
    forecast_kwh DOUBLE PRECISION NOT NULL,
    lower_kwh    DOUBLE PRECISION NOT NULL,
    upper_kwh    DOUBLE PRECISION NOT NULL,
    served_at    TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (building_id, ts)
);