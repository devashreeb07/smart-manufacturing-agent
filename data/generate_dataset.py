"""
Synthetic dataset generator for the Smart Manufacturing Analytics Agent.

Generates:
  - machines.csv        : 5 machines across different types
  - production_data.csv : ~180 days x 5 machines x 3 shifts of production records,
                           with realistic noise, weekly seasonality, and
                           deliberately injected anomaly events (breakdowns,
                           gradual degradation, defect spikes) so the
                           anomaly-detection / forecasting / failure-risk
                           components have real signal to find.

Run: python generate_dataset.py
Outputs land in this same /data folder.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Machines
# ---------------------------------------------------------------------------
machines = pd.DataFrame([
    {"machine_id": "M1", "machine_name": "Cutter Alpha",   "machine_type": "Cutter",  "install_date": "2022-03-15", "rated_capacity": 1800},
    {"machine_id": "M2", "machine_name": "Press Beta",     "machine_type": "Press",   "install_date": "2021-11-01", "rated_capacity": 1600},
    {"machine_id": "M3", "machine_name": "Cutter Gamma",   "machine_type": "Cutter",  "install_date": "2023-06-10", "rated_capacity": 1700},
    {"machine_id": "M4", "machine_name": "Folder Delta",   "machine_type": "Folder",  "install_date": "2020-08-20", "rated_capacity": 1500},
    {"machine_id": "M5", "machine_name": "Press Epsilon",  "machine_type": "Press",   "install_date": "2022-01-05", "rated_capacity": 1650},
])
machines.to_csv("machines.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Production data
# ---------------------------------------------------------------------------
START_DATE = date(2026, 1, 1)
NUM_DAYS = 180
SHIFTS = ["Morning", "Evening", "Night"]
SHIFT_DURATION = 480  # minutes

# Planned anomaly events: (machine_id, start_day_offset, duration_days, kind)
# kind: 'breakdown' = sudden severe event (like M3 blade failure in the spec)
#       'degradation' = gradual worsening over the window (early warning signal)
#       'defect_spike' = isolated quality issue, downtime mostly normal
ANOMALY_EVENTS = [
    ("M3", 40, 3,  "breakdown"),      # sudden cutter blade failure
    ("M2", 90, 10, "degradation"),    # bearing wear gradually worsening
    ("M5", 130, 2, "defect_spike"),   # bad raw material batch
    ("M1", 160, 4, "breakdown"),      # second breakdown late in the series
]

records = []
for day_offset in range(NUM_DAYS):
    current_date = START_DATE + timedelta(days=day_offset)
    weekday = current_date.weekday()  # 0=Mon .. 6=Sun

    for _, m in machines.iterrows():
        machine_id = m["machine_id"]
        base_capacity_per_shift = m["rated_capacity"] // 3

        for shift in SHIFTS:
            # --- baseline behaviour ---
            # weekend dip (lighter staffing), slight night-shift dip
            weekday_factor = 0.7 if weekday >= 5 else 1.0
            shift_factor = 0.92 if shift == "Night" else 1.0

            target_units = int(base_capacity_per_shift * weekday_factor * shift_factor)

            # normal efficiency ~ 90-98%, small random noise
            efficiency = np.random.normal(0.94, 0.03)
            efficiency = np.clip(efficiency, 0.75, 1.02)

            # normal defect rate ~ 1.5-3%
            defect_rate = np.random.normal(0.022, 0.006)
            defect_rate = max(defect_rate, 0.002)

            # normal downtime ~ 2-6% of shift
            downtime_pct = np.random.normal(0.04, 0.015)
            downtime_pct = max(downtime_pct, 0.0)

            failure_flag = False

            # --- apply anomaly events ---
            for (a_machine, a_start, a_dur, a_kind) in ANOMALY_EVENTS:
                if machine_id == a_machine and a_start <= day_offset < a_start + a_dur:
                    progress = (day_offset - a_start) / max(a_dur - 1, 1)  # 0..1 through event

                    if a_kind == "breakdown":
                        efficiency = np.random.normal(0.55, 0.08)
                        defect_rate = np.random.normal(0.09, 0.02)
                        downtime_pct = np.random.normal(0.35, 0.08)
                        failure_flag = True

                    elif a_kind == "degradation":
                        # gets steadily worse across the window
                        efficiency = np.random.normal(0.90 - 0.20 * progress, 0.03)
                        defect_rate = np.random.normal(0.025 + 0.05 * progress, 0.008)
                        downtime_pct = np.random.normal(0.05 + 0.12 * progress, 0.02)
                        failure_flag = progress > 0.7

                    elif a_kind == "defect_spike":
                        efficiency = np.random.normal(0.88, 0.04)
                        defect_rate = np.random.normal(0.12, 0.02)
                        downtime_pct = np.random.normal(0.05, 0.015)
                        failure_flag = False

            efficiency = np.clip(efficiency, 0.2, 1.02)
            defect_rate = np.clip(defect_rate, 0.0, 0.5)
            downtime_pct = np.clip(downtime_pct, 0.0, 0.9)

            units_produced = max(int(target_units * efficiency), 0)
            defective_units = int(units_produced * defect_rate)
            downtime_minutes = int(SHIFT_DURATION * downtime_pct)

            records.append({
                "date": current_date.isoformat(),
                "machine_id": machine_id,
                "shift": shift,
                "target_units": target_units,
                "units_produced": units_produced,
                "defective_units": defective_units,
                "downtime_minutes": downtime_minutes,
                "shift_duration_min": SHIFT_DURATION,
                "failure_flag": failure_flag,
            })

df = pd.DataFrame(records)
df.to_csv("production_data.csv", index=False)

print(f"Generated {len(df):,} production records across {len(machines)} machines and {NUM_DAYS} days.")
print(f"Files written: machines.csv, production_data.csv")
print(f"\nAnomaly events injected:")
for e in ANOMALY_EVENTS:
    print(f"  - {e[0]}: {e[3]} starting day {e[1]} for {e[2]} days")
