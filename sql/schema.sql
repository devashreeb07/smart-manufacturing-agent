-- Smart Manufacturing Analytics Agent
-- Core schema

CREATE TABLE IF NOT EXISTS machines (
    machine_id      VARCHAR(10) PRIMARY KEY,
    machine_name    VARCHAR(50) NOT NULL,
    machine_type    VARCHAR(50) NOT NULL,   -- e.g. Cutter, Press, Folder
    install_date    DATE NOT NULL,
    rated_capacity  INT NOT NULL            -- target units/day at 100% efficiency
);

CREATE TABLE IF NOT EXISTS production_data (
    record_id           SERIAL PRIMARY KEY,
    date                DATE NOT NULL,
    machine_id          VARCHAR(10) NOT NULL REFERENCES machines(machine_id),
    shift               VARCHAR(10) NOT NULL,   -- Morning / Evening / Night
    target_units        INT NOT NULL,
    units_produced      INT NOT NULL,
    defective_units     INT NOT NULL,
    downtime_minutes    INT NOT NULL,
    shift_duration_min  INT NOT NULL DEFAULT 480,  -- 8 hr shift
    failure_flag        BOOLEAN DEFAULT FALSE,      -- true if a failure/breakdown occurred this record
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prod_date ON production_data(date);
CREATE INDEX IF NOT EXISTS idx_prod_machine ON production_data(machine_id);

-- Derived KPI view (computed on read, not stored)
CREATE OR REPLACE VIEW production_kpis AS
SELECT
    record_id,
    date,
    machine_id,
    shift,
    units_produced,
    target_units,
    defective_units,
    downtime_minutes,
    shift_duration_min,
    ROUND(100.0 * units_produced / NULLIF(target_units, 0), 2)          AS efficiency_pct,
    ROUND(100.0 * defective_units / NULLIF(units_produced, 0), 2)       AS defect_rate_pct,
    ROUND(100.0 * downtime_minutes / NULLIF(shift_duration_min, 0), 2)  AS downtime_pct,
    failure_flag
FROM production_data;
