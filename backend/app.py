"""
Smart Manufacturing Analytics Agent -- Flask API

Exposes all five backend modules (KPIs, anomaly detection, forecasting,
failure risk, recommendations) as REST endpoints, reading from PostgreSQL.
Power BI connects to these endpoints via its Web data source connector.

Run locally:  python app.py
Runs on:      http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from db import query_to_dicts

app = Flask(__name__)
CORS(app)  # allow Power BI / any frontend to fetch these endpoints


@app.route("/")
def health_check():
    return jsonify({"status": "ok", "service": "Smart Manufacturing Analytics Agent API"})


# ---------------------------------------------------------------------------
# Machines
# ---------------------------------------------------------------------------

@app.route("/api/machines")
def get_machines():
    rows = query_to_dicts("SELECT * FROM machines ORDER BY machine_id;")
    return jsonify(rows)


# ---------------------------------------------------------------------------
# KPIs  (uses the production_kpis view created in schema.sql)
# ---------------------------------------------------------------------------

@app.route("/api/kpis")
def get_kpis():
    machine_id = request.args.get("machine_id")
    limit = request.args.get("limit", 500)

    sql = "SELECT * FROM production_kpis"
    params = []
    if machine_id:
        sql += " WHERE machine_id = %s"
        params.append(machine_id)
    sql += " ORDER BY date DESC LIMIT %s;"
    params.append(int(limit))

    rows = query_to_dicts(sql, tuple(params))
    return jsonify(rows)


@app.route("/api/kpis/summary")
def get_kpi_summary():
    """One row per machine: overall averages -- powers the top-level dashboard cards."""
    sql = """
        SELECT
            machine_id,
            ROUND(AVG(efficiency_pct), 2)  AS avg_efficiency_pct,
            ROUND(AVG(defect_rate_pct), 2) AS avg_defect_rate_pct,
            ROUND(AVG(downtime_pct), 2)    AS avg_downtime_pct,
            SUM(units_produced)            AS total_units_produced,
            SUM(defective_units)           AS total_defective_units,
            SUM(downtime_minutes)          AS total_downtime_minutes
        FROM production_kpis
        GROUP BY machine_id
        ORDER BY avg_efficiency_pct ASC;
    """
    return jsonify(query_to_dicts(sql))


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

@app.route("/api/anomalies")
def get_anomalies():
    severity = request.args.get("severity")  # optional filter: LOW / MEDIUM / HIGH
    sql = "SELECT * FROM anomaly_flags WHERE is_anomaly = TRUE"
    params = []
    if severity:
        sql += " AND severity = %s"
        params.append(severity.upper())
    sql += " ORDER BY date DESC;"
    return jsonify(query_to_dicts(sql, tuple(params)))


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------

@app.route("/api/forecasts")
def get_forecasts():
    machine_id = request.args.get("machine_id")
    metric = request.args.get("metric")  # units_produced / defective_units

    sql = "SELECT * FROM forecasts WHERE 1=1"
    params = []
    if machine_id:
        sql += " AND machine_id = %s"
        params.append(machine_id)
    if metric:
        sql += " AND metric = %s"
        params.append(metric)
    sql += " ORDER BY date ASC;"

    return jsonify(query_to_dicts(sql, tuple(params)))


# ---------------------------------------------------------------------------
# Failure risk
# ---------------------------------------------------------------------------

@app.route("/api/failure-risk")
def get_failure_risk():
    """Latest risk snapshot per machine -- powers the risk-band dashboard tiles."""
    sql = """
        SELECT DISTINCT ON (machine_id)
            machine_id, date, failure_probability, risk_band
        FROM failure_risk
        ORDER BY machine_id, date DESC;
    """
    return jsonify(query_to_dicts(sql))


@app.route("/api/failure-risk/history")
def get_failure_risk_history():
    machine_id = request.args.get("machine_id")
    sql = "SELECT machine_id, date, failure_probability, risk_band FROM failure_risk"
    params = []
    if machine_id:
        sql += " WHERE machine_id = %s"
        params.append(machine_id)
    sql += " ORDER BY date ASC;"
    return jsonify(query_to_dicts(sql, tuple(params)))


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@app.route("/api/recommendations")
def get_recommendations():
    severity = request.args.get("severity")
    limit = request.args.get("limit", 50)

    sql = "SELECT * FROM recommendations WHERE 1=1"
    params = []
    if severity:
        sql += " AND severity = %s"
        params.append(severity.upper())
    sql += " ORDER BY date DESC LIMIT %s;"
    params.append(int(limit))

    return jsonify(query_to_dicts(sql, tuple(params)))


# ---------------------------------------------------------------------------
# Combined dashboard summary -- one call for the whole overview screen
# ---------------------------------------------------------------------------

@app.route("/api/dashboard")
def get_dashboard():
    return jsonify({
        "kpi_summary": query_to_dicts("""
            SELECT machine_id, ROUND(AVG(efficiency_pct),2) AS avg_efficiency_pct,
                   ROUND(AVG(defect_rate_pct),2) AS avg_defect_rate_pct,
                   ROUND(AVG(downtime_pct),2) AS avg_downtime_pct
            FROM production_kpis GROUP BY machine_id ORDER BY avg_efficiency_pct ASC;
        """),
        "current_risk": query_to_dicts("""
            SELECT DISTINCT ON (machine_id) machine_id, failure_probability, risk_band
            FROM failure_risk ORDER BY machine_id, date DESC;
        """),
        "recent_recommendations": query_to_dicts("""
            SELECT date, machine_id, severity, recommendation
            FROM recommendations ORDER BY date DESC LIMIT 10;
        """),
        "high_severity_count": query_to_dicts("""
            SELECT COUNT(*) AS count FROM anomaly_flags WHERE severity = 'HIGH';
        """)[0]["count"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
