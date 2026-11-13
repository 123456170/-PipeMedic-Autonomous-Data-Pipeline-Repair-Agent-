"""PipeMedic — Autonomous Data Pipeline Repair Agent.
Single-file Streamlit app. No API keys. Auto-starts a live, seeded simulation.
"""
from __future__ import annotations

import html
import json
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

APP_NAME = "PipeMedic"
APP_VER = "v0.9.2"
REQUIREMENTS_TEXT = "streamlit\npandas\nnumpy\nplotly\n"
MANUAL_BASELINE_MIN = 252.0

st.set_page_config(page_title="PipeMedic · Autonomous Pipeline Repair Agent",
                   page_icon="⛑️", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- palette / css
PAL = {"cyan": "#0891B2", "blue": "#2563EB", "violet": "#7C3AED", "magenta": "#C026D3",
       "amber": "#D97706", "red": "#E11D48", "green": "#059669", "rose": "#F43F5E",
       "muted": "#5B6B87", "text": "#0F172A", "accent": "#4F46E5"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
:root{--bg:#F6F8FC;--surface:#FFFFFF;--surface2:#F1F5FB;--line:rgba(79,70,229,.16);
--text:#0F172A;--muted:#5B6B87;--accent:#4F46E5;--violet:#7C3AED;--amber:#D97706;--red:#E11D48;--green:#059669;}
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:radial-gradient(1200px 600px at 85% -10%,rgba(124,58,237,.10),transparent 60%),
radial-gradient(900px 500px at -10% 20%,rgba(79,70,229,.08),transparent 55%),
linear-gradient(180deg,#F6F8FC 0%,#EEF2F9 100%);color:var(--text);}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#FFFFFF,#F1F5FB);border-right:1px solid var(--line);}
*::-webkit-scrollbar{width:10px;height:10px}
*::-webkit-scrollbar-track{background:#EEF2F9}
*::-webkit-scrollbar-thumb{background:linear-gradient(180deg,#6366F1,#8B5CF6);border-radius:8px;border:2px solid #EEF2F9}
*::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,#4F46E5,#7C3AED)}
*{scrollbar-width:thin;scrollbar-color:#6366F1 #EEF2F9;}
div[data-testid="stMetric"]{background:linear-gradient(160deg,var(--surface),var(--surface2));
border:1px solid var(--line);border-radius:14px;padding:12px 14px;box-shadow:0 4px 18px rgba(15,23,42,.07);}
div[data-testid="stMetric"] label{color:var(--muted)!important;font-size:.7rem!important;
letter-spacing:.08em;text-transform:uppercase;font-weight:600;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;color:var(--text);}
button[data-baseweb="tab"]{font-weight:600;color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--accent);}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#4F46E5,#7C3AED);height:3px;border-radius:3px;}
.stButton>button,button[data-testid="stBaseButton-secondary"]{background:var(--surface);
border:1px solid var(--line);color:var(--text);border-radius:10px;font-weight:600;transition:all .15s ease;}
.stButton>button:hover,button[data-testid="stBaseButton-secondary"]:hover{border-color:#4F46E5;
color:#4338CA;transform:translateY(-1px);box-shadow:0 0 14px rgba(79,70,229,.18);}
button[data-testid="stBaseButton-primary"],.stButton>button[kind="primary"]{
background:linear-gradient(90deg,#4F46E5,#7C3AED);color:#FFFFFF;border:none;font-weight:800;
box-shadow:0 4px 16px rgba(79,70,229,.30);}
button[data-testid="stBaseButton-primary"]:hover{transform:translateY(-1px);box-shadow:0 6px 22px rgba(79,70,229,.45);}
div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:rgba(255,255,255,.75);}
div[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(255,255,255,.7);border:1px solid var(--line);border-radius:14px;}
code,pre{font-family:'JetBrains Mono',monospace!important;}
div[data-testid="stCode"] pre,.stCodeBlock pre{background:#F1F5FB!important;border:1px solid var(--line);border-radius:10px;}
div[data-testid="stProgress"]>div>div>div{background:linear-gradient(90deg,#4F46E5,#7C3AED);}
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:16px;padding:18px 22px;
background:linear-gradient(120deg,rgba(79,70,229,.08),rgba(124,58,237,.10) 55%,rgba(192,38,211,.06));margin-bottom:10px;}
.hero::after{content:"";position:absolute;top:0;left:-40%;width:40%;height:100%;
background:linear-gradient(90deg,transparent,rgba(79,70,229,.07),transparent);animation:scan 6s linear infinite;}
@keyframes scan{to{left:110%}}
.hero h1{margin:0;font-size:1.5rem;font-weight:800;color:var(--text);}
.grad{background:linear-gradient(90deg,#4F46E5,#7C3AED,#C026D3);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero .sub{color:var(--muted);font-size:.82rem;margin-top:2px;}
.hero .right{position:absolute;right:22px;top:16px;text-align:right;font-family:'JetBrains Mono',monospace;
font-size:.76rem;color:#334155;line-height:1.5;}
@media(max-width:760px){.hero .right{display:none}}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--green);
box-shadow:0 0 10px var(--green);animation:pulse 1.6s ease infinite;margin-right:6px;}
.dot.paused{background:var(--amber);box-shadow:0 0 10px var(--amber);animation:none;}
.dot.kill{background:var(--red);box-shadow:0 0 10px var(--red);animation:pulse .8s ease infinite;}
@keyframes pulse{50%{opacity:.35}}
.killbar{border:1px solid rgba(225,29,72,.45);background:rgba(225,29,72,.08);color:#9F1239;
border-radius:12px;padding:10px 14px;font-weight:700;margin-bottom:10px;font-size:.85rem;}
.tip{border:1px dashed rgba(79,70,229,.35);background:rgba(79,70,229,.05);border-radius:12px;
padding:9px 14px;font-size:.8rem;color:#334155;margin-bottom:12px;}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.7rem;font-weight:700;
letter-spacing:.04em;border:1px solid;margin-right:6px;font-family:'JetBrains Mono',monospace;}
.p-red{color:#BE123C;border-color:rgba(225,29,72,.45);background:rgba(225,29,72,.08);}
.p-amber{color:#B45309;border-color:rgba(217,119,6,.45);background:rgba(217,119,6,.10);}
.p-green{color:#047857;border-color:rgba(5,150,105,.45);background:rgba(5,150,105,.09);}
.p-cyan{color:#0E7490;border-color:rgba(8,145,178,.45);background:rgba(8,145,178,.09);}
.p-violet{color:#6D28D9;border-color:rgba(124,58,237,.45);background:rgba(124,58,237,.09);}
.p-magenta{color:#A21CAF;border-color:rgba(192,38,211,.45);background:rgba(192,38,211,.08);}
.p-blue{color:#1D4ED8;border-color:rgba(37,99,235,.45);background:rgba(37,99,235,.09);}
.evt{display:flex;gap:8px;align-items:baseline;padding:5px 8px;border-left:2px solid rgba(79,70,229,.30);
margin:3px 0;background:rgba(255,255,255,.75);border-radius:0 8px 8px 0;font-size:.79rem;}
.et{font-family:'JetBrains Mono',monospace;color:var(--muted);font-size:.68rem;min-width:44px;}
.ei{font-family:'JetBrains Mono',monospace;color:#1D4ED8;font-size:.66rem;min-width:66px;}
.em{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.64rem;padding:1px 6px;border-radius:6px;}
.m-M1{color:#0E7490;background:rgba(8,145,178,.12)}.m-M2{color:#6D28D9;background:rgba(124,58,237,.12)}
.m-M3{color:#B45309;background:rgba(217,119,6,.14)}.m-M4{color:#047857;background:rgba(5,150,105,.12)}
.m-HUMAN{color:#1D4ED8;background:rgba(37,99,235,.12)}.m-POLICY{color:#A21CAF;background:rgba(192,38,211,.12)}
.m-SYS{color:#5B6B87;background:rgba(91,107,135,.14)}
.ex{color:#334155;}
.pc{background:rgba(255,255,255,.8);border:1px solid var(--line);border-radius:12px;padding:10px 12px;
font-size:.76rem;margin-bottom:10px;line-height:1.6;}
.pc b{font-family:'JetBrains Mono',monospace;font-size:.8rem;}
.ok{color:var(--green);font-weight:700}.warn{color:var(--amber);font-weight:700}
.bad{color:var(--red);font-weight:700}.mut{color:var(--muted)}
.mono{font-family:'JetBrains Mono',monospace;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- domain data
PIPELINES = [
    {"id": "orders_fact_load", "model": "fct_orders", "table": "prod.analytics.fct_orders",
     "team": "Analytics Eng", "tier": 1, "sla_min": 30, "schedule": "*/15 * * * *", "rows_avg": 1_200_000,
     "upstream": ["raw_orders_ingest", "fx_rates_sync"], "downstream": ["revenue_daily", "finance_close"]},
    {"id": "payments_ledger", "model": "fct_payments", "table": "prod.finance.fct_payments",
     "team": "Platform SRE", "tier": 1, "sla_min": 15, "schedule": "*/10 * * * *", "rows_avg": 860_000,
     "upstream": ["psp_webhooks_ingest", "fx_rates_sync"], "downstream": ["settlement_report", "payout_recon"]},
    {"id": "finance_fx_rates", "model": "dim_fx_rates", "table": "prod.finance.dim_fx_rates",
     "team": "Platform SRE", "tier": 1, "sla_min": 60, "schedule": "0 * * * *", "rows_avg": 4_200,
     "upstream": ["ecb_feed_pull"], "downstream": ["orders_fact_load", "payments_ledger"]},
    {"id": "clickstream_sessions", "model": "fct_sessions", "table": "prod.product.fct_sessions",
     "team": "Data Eng", "tier": 2, "sla_min": 45, "schedule": "*/30 * * * *", "rows_avg": 5_400_000,
     "upstream": ["events_kafka_sink"], "downstream": ["marketing_attribution", "funnel_weekly"]},
    {"id": "crm_contacts_sync", "model": "dim_contacts", "table": "prod.crm.dim_contacts",
     "team": "Analytics Eng", "tier": 2, "sla_min": 120, "schedule": "0 */2 * * *", "rows_avg": 310_000,
     "upstream": ["salesforce_export"], "downstream": ["user_profiles_scd2", "ltv_model"]},
    {"id": "inventory_snapshot", "model": "fct_inventory", "table": "prod.ops.fct_inventory",
     "team": "Data Eng", "tier": 2, "sla_min": 60, "schedule": "0 * * * *", "rows_avg": 95_000,
     "upstream": ["erp_extract"], "downstream": ["stockout_alerts", "supply_dashboard"]},
    {"id": "user_profiles_scd2", "model": "dim_users", "table": "prod.core.dim_users",
     "team": "Data Eng", "tier": 2, "sla_min": 180, "schedule": "0 */4 * * *", "rows_avg": 2_100_000,
     "upstream": ["auth_events", "crm_contacts_sync"], "downstream": ["ltv_model", "churn_features"]},
    {"id": "marketing_attribution", "model": "fct_attribution", "table": "prod.growth.fct_attribution",
     "team": "Analytics Eng", "tier": 3, "sla_min": 240, "schedule": "0 */6 * * *", "rows_avg": 780_000,
     "upstream": ["ads_api_pull", "clickstream_sessions"], "downstream": ["roas_dashboard"]},
]
PIPE_BY_ID = {p["id"]: p for p in PIPELINES}

FAULTS = {
    "schema_drift": {
        "label": "Schema drift", "icon": "🧬", "color": "magenta", "difficulty": 0.88,
        "risk": (45, 75), "wrong_alt": "upstream_delay",
        "signal": "producer emitted `amount` as STRING (was NUMERIC); 3 downstream models failed compilation",
        "cause": "Upstream producer changed a column type without a contract version bump",
        "prevention": ["Add dbt source contract (data_type + not_null) with CI gate",
                       "Subscribe #data-contracts to the producer schema-change webhook",
                       "Introduce a defensive CAST layer in staging models"],
        "diff": """--- a/models/marts/__MODEL__.sql
+++ b/models/marts/__MODEL__.sql
@@ -14,7 +14,9 @@
     id,
     customer_id,
-    amount,
+    CAST(amount AS NUMERIC(18,4))  AS amount,
+    COALESCE(currency_code, 'USD') AS currency_code,
     created_at
 from {{ source('raw', '__MODEL__') }}
--- a/models/marts/schema.yml
+++ b/models/marts/schema.yml
@@
       - name: amount
+        data_type: numeric
+        tests: [not_null]
""",
    },
    "upstream_delay": {
        "label": "Upstream delay", "icon": "⏳", "color": "amber", "difficulty": 0.93,
        "risk": (15, 35), "wrong_alt": "warehouse_timeout",
        "signal": "ExternalTaskSensor 'wait_upstream' timed out after 30m; upstream DAG last success 74m ago",
        "cause": "Upstream DAG delayed beyond the sensor timeout (producer SLA breach)",
        "prevention": ["Freshness alert at 50% of sensor timeout",
                       "Reschedule-mode sensors with exponential backoff (in patch)",
                       "Link upstream-delay runbook in DAG docs"],
        "diff": """--- a/dags/__PIPE___dag.py
+++ b/dags/__PIPE___dag.py
@@ -22,8 +22,11 @@
 sensor = ExternalTaskSensor(
     task_id="wait_upstream",
-    timeout=1800,
-    poke_interval=60,
+    timeout=3300,
+    poke_interval=120,
+    mode="reschedule",          # free the worker slot while waiting
+    exponential_backoff=True,
 )
 run = PythonOperator(
     task_id="load___MODEL__",
-    execution_timeout=timedelta(minutes=30),
+    execution_timeout=timedelta(minutes=55),
 )
""",
    },
    "dq_null_spike": {
        "label": "DQ: null spike", "icon": "🕳️", "color": "violet", "difficulty": 0.86,
        "risk": (30, 55), "wrong_alt": "schema_drift",
        "signal": "dbt test not_null(amount) failed: null rate 18.4% vs 0.1% baseline (z=9.7)",
        "cause": "Upstream API began omitting `amount` for a subset of records",
        "prevention": ["Null-rate anomaly test (threshold 1%) at the source layer",
                       "Quarantine table + daily rejected-rows digest to the owner",
                       "Producer ticket: make `amount` required in the API schema"],
        "diff": """--- a/models/marts/__MODEL__.sql
+++ b/models/marts/__MODEL__.sql
@@ -9,6 +9,14 @@
+with quarantined as (
+    select * from {{ source('raw', '__MODEL__') }}
+    where amount is not null
+),
+rejected as (
+    select *, 'null_amount' as quarantine_reason
+    from {{ source('raw', '__MODEL__') }} where amount is null
+)
 select
     id,
-    amount,
+    coalesce(amount, 0) as amount,
+    amount is null      as amount_imputed
 from quarantined
--- a/models/marts/schema.yml
+++ b/models/marts/schema.yml
@@
+      - name: amount
+        tests:
+          - not_null:
+              severity: error
""",
    },
    "dq_duplicate_keys": {
        "label": "DQ: duplicate keys", "icon": "🧱", "color": "blue", "difficulty": 0.90,
        "risk": (25, 50), "wrong_alt": "late_data",
        "signal": "unique(id) test failed: 4,213 duplicate keys after the CDC merge",
        "cause": "CDC replay re-emitted events; the merge key lacked a tie-breaker",
        "prevention": ["unique(id) test at source, not just at the mart",
                       "Idempotent merge with updated_at tie-breaker (in patch)",
                       "CDC replay alert when lag > 5m"],
        "diff": """--- a/models/marts/__MODEL__.sql
+++ b/models/marts/__MODEL__.sql
@@ -6,8 +6,17 @@
+with deduped as (
+    select *,
+           row_number() over (
+             partition by id
+             order by updated_at desc, _ingested_at desc
+           ) as rn
+    from {{ source('raw', '__MODEL__') }}
+)
 select id, amount, currency_code, updated_at
 from deduped
-where 1 = 1
+where rn = 1
--- a/models/marts/schema.yml
+++ b/models/marts/schema.yml
@@
       - name: id
+        tests: [unique, not_null]
""",
    },
    "warehouse_timeout": {
        "label": "Warehouse timeout", "icon": "🐌", "color": "cyan", "difficulty": 0.92,
        "risk": (20, 45), "wrong_alt": "resource_oom",
        "signal": "query 01b2f44e cancelled: STATEMENT_TIMEOUT after 900s; queue depth 17",
        "cause": "Warehouse contention plus an unclustered merge exceeded the statement timeout",
        "prevention": ["Cluster staging by merge key; review auto-suspend policy",
                       "Queue-depth alert > 10 sustained 5m",
                       "Cost guard: auto-revert warehouse size after backfill"],
        "diff": """--- a/infra/warehouse/__PIPE__.sql
+++ b/infra/warehouse/__PIPE__.sql
@@
-ALTER WAREHOUSE wh_etl_small SET WAREHOUSE_SIZE = 'SMALL';
+ALTER WAREHOUSE wh_etl_small SET WAREHOUSE_SIZE = 'MEDIUM';  -- revert after backfill
+ALTER WAREHOUSE wh_etl_small SET STATEMENT_TIMEOUT_IN_SECONDS = 1800;
@@
-MERGE INTO __TABLE__ t USING staged s ON t.id = s.id
+MERGE INTO __TABLE__ t
+USING (SELECT * FROM staged CLUSTER BY id) s ON t.id = s.id
""",
    },
    "credential_expiry": {
        "label": "Credential expiry", "icon": "🔑", "color": "red", "difficulty": 0.95,
        "risk": (10, 30), "wrong_alt": "upstream_delay",
        "signal": "connection 'warehouse_legacy' auth failed: token expired 42m ago (rotated by IAM)",
        "cause": "Service credential expired after IAM rotation; the DAG still references the old secret",
        "prevention": ["30-day rotation policy + expiry alert at T-7d",
                       "Connection indirection via a secrets-manager alias",
                       "Synthetic auth canary every 6h"],
        "diff": """--- a/k8s/secrets/__PIPE__-conn.yaml
+++ b/k8s/secrets/__PIPE__-conn.yaml
@@
-  secretRef: warehouse-creds-2024
+  secretRef: warehouse-creds-rotated-__ID__
+  rotationPolicy: 30d
--- a/dags/__PIPE___dag.py
+++ b/dags/__PIPE___dag.py
@@
-conn = BaseHook.get_connection("warehouse_legacy")
+conn = BaseHook.get_connection("warehouse_rotated")
""",
    },
    "late_data": {
        "label": "Late-arriving data", "icon": "📦", "color": "green", "difficulty": 0.84,
        "risk": (20, 40), "wrong_alt": "upstream_delay",
        "signal": "freshness SLA breached: max(event_time) lag 5.2h; source watermark jumped",
        "cause": "Late-arriving events outside the incremental window were silently dropped",
        "prevention": ["6h lookback + idempotent merge (in patch)",
                       "Watermark monitor with a late-event histogram",
                       "Automated backfill when lag exceeds SLA"],
        "diff": """--- a/models/marts/__MODEL__.sql
+++ b/models/marts/__MODEL__.sql
@@
-    where event_time >= (select max(event_time) from {{ this }})
+    where event_time >= (
+        select dateadd(hour, -6, max(event_time)) from {{ this }}
+    )   -- 6h lookback absorbs late arrivals; merge is idempotent
--- a/dags/__PIPE___dag.py
+++ b/dags/__PIPE___dag.py
@@
-    catchup=False,
+    catchup=True,
+    max_active_runs=1,
""",
    },
    "resource_oom": {
        "label": "Executor OOM", "icon": "💥", "color": "rose", "difficulty": 0.90,
        "risk": (25, 50), "wrong_alt": "warehouse_timeout",
        "signal": "executor lost: OOMKilled (exit 137); shuffle spill 22GB on stage 4",
        "cause": "Skewed shuffle keys exhausted executor memory during repartition",
        "prevention": ["AQE + skew-join hints; repartition on a high-cardinality key",
                       "Right-size executor memory from peak-shuffle metrics",
                       "Load test at 2x peak volume in CI"],
        "diff": """--- a/k8s/jobs/__PIPE__-spark.yaml
+++ b/k8s/jobs/__PIPE__-spark.yaml
@@
-  executorMemory: "4g"
-  executorCores: 2
+  executorMemory: "8g"
+  executorCores: 2
+  shufflePartitions: 400
+  adaptiveExecution: true
--- a/jobs/__PIPE__.py
+++ b/jobs/__PIPE__.py
@@
-df.repartition(64).write.mode("append").saveAsTable("__TABLE__")
+df.repartition(400, "id").write.mode("append").saveAsTable("__TABLE__")
""",
    },
}
FAULT_KEYS = list(FAULTS)
FAULT_WEIGHTS = [0.18, 0.18, 0.14, 0.10, 0.12, 0.08, 0.12, 0.08]

FLOW = ["detect", "classify", "investigate", "diagnose", "propose", "gate",
        "staging", "apply", "verify", "postmortem", "closed"]
FLOW_INDEX = {s: i for i, s in enumerate(FLOW)}
BASE_MIN = {"detect": 1.2, "classify": 1.8, "investigate": 5.5, "diagnose": 2.2, "propose": 3.5,
            "gate": 0.4, "staging": 4.5, "apply": 2.5, "verify": 3.5, "postmortem": 5.0}
STAGE_LABEL = {"detect": "Detecting signal", "classify": "Classifying failure",
               "investigate": "Tracing root cause", "diagnose": "Forming diagnosis",
               "propose": "Drafting repair PR", "gate": "Policy gate",
               "staging": "Validating in staging", "apply": "Promoting to prod",
               "verify": "Verifying recovery", "postmortem": "Writing postmortem", "closed": "Closed"}
MODES = ["Suggest only", "Act with approval", "Act & notify", "Fully autonomous"]
KIND_ICON = {"plan": "🧭", "act": "🔧", "observe": "👁️", "reflect": "🪞", "write": "📝",
             "verify": "✅", "gate": "⏸️", "rollback": "↩️", "notify": "📣"}
STATUS_META = {"active": ("ACTIVE", "cyan"), "awaiting_approval": ("AWAITING APPROVAL", "amber"),
               "held_kill": ("HELD · KILL SWITCH", "magenta"), "held_budget": ("HELD · BUDGET", "magenta"),
               "escalated": ("ESCALATED", "red"), "rolling_back": ("ROLLING BACK", "magenta"),
               "recovered": ("RECOVERED · POSTMORTEM", "green"), "closed": ("CLOSED", "blue"),
               "manual_closed": ("CLOSED · MANUAL", "blue")}
SEV_COLOR = {"P1": "red", "P2": "amber", "P3": "blue"}
DEFAULT_POLICY = {"speed": 2.0, "rate": 3.0, "chaos": False, "mode": "Act & notify",
                  "conf_floor": 0.80, "max_risk": 40, "budget": 25.0, "kill": False}

# ---------------------------------------------------------------- small helpers
def clamp(x, a, b): return max(a, min(b, x))
def now_utc(): return datetime.now(timezone.utc).replace(microsecond=0)
def pill(txt, color): return f'<span class="pill p-{color}">{html.escape(str(txt))}</span>'
def get_rng(): return st.session_state.rng

def st_btn(label, key=None, kind="secondary", disabled=False, full=True):
    try:
        return st.button(label, key=key, type=kind, disabled=disabled,
                         width="stretch" if full else "content")
    except TypeError:
        return st.button(label, key=key, type=kind, disabled=disabled, use_container_width=full)

def st_download(label, data, fname, key, mime="text/plain", kind="secondary"):
    try:
        return st.download_button(label, data, fname, mime, key=key, type=kind, width="stretch")
    except TypeError:
        return st.download_button(label, data, fname, mime, key=key, type=kind, use_container_width=True)

def st_fig(fig, key):
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, key=key, config=cfg, width="stretch")
    except TypeError:
        try:
            st.plotly_chart(fig, key=key, config=cfg, use_container_width=True)
        except TypeError:
            st.plotly_chart(fig, config=cfg)

def st_frame(df, key=None, height=None):
    kw = {"height": height} if height else {}
    try:
        st.dataframe(df, key=key, hide_index=True, width="stretch", **kw)
    except TypeError:
        try:
            st.dataframe(df, key=key, hide_index=True, use_container_width=True, **kw)
        except TypeError:
            st.dataframe(df, hide_index=True, **kw)

def st_dot(dot, key):
    try:
        st.graphviz_chart(dot, key=key, width="stretch")
    except TypeError:
        try:
            st.graphviz_chart(dot, use_container_width=True)
        except TypeError:
            st.graphviz_chart(dot)

def style_fig(fig, h):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=h,
                      font=dict(family="JetBrains Mono, monospace", size=11, color="#334155"),
                      margin=dict(l=10, r=10, t=34, b=10),
                      legend=dict(orientation="h", y=1.14, x=0),
                      hoverlabel=dict(bgcolor="#FFFFFF"))
    fig.update_xaxes(gridcolor="rgba(100,116,139,.18)")
    fig.update_yaxes(gridcolor="rgba(100,116,139,.18)")
    return fig

def events_html(events, show_inc=False, limit=None):
    evs = events if limit is None else events[-limit:]
    rows = []
    for e in reversed(evs):
        t = e["t"].strftime("%H:%M") if isinstance(e["t"], datetime) else str(e["t"])[:16]
        chip = f'<span class="ei">{html.escape(str(e["inc"]))}</span>' if show_inc else ""
        rows.append(f'<div class="evt"><span class="et">{t}</span>{chip}'
                    f'<span class="em m-{e["module"]}">{e["module"]}</span>'
                    f'<span>{KIND_ICON.get(e["kind"], "•")}</span>'
                    f'<span class="ex">{html.escape(e["text"])}</span></div>')
    return "".join(rows) or '<div class="mut">No events yet.</div>'

# ---------------------------------------------------------------- simulation engine
def set_stage(inc, stage, rng):
    inc["stage"] = stage
    base = BASE_MIN.get(stage)
    inc["stage_dur"] = None if base is None else max(0.3, base * rng.uniform(0.75, 1.3))
    inc["stage_elapsed"] = 0.0

def emit(sim, inc, rng, module, kind, text, tool=None, tokens=0):
    ev = {"t": sim["clock"], "inc": inc["id"], "module": module, "kind": kind,
          "text": text, "tool": tool, "ms": rng.randint(40, 2400)}
    if tokens:
        cost = tokens / 1000.0 * 0.004
        sim["day_cost"] += cost
        sim["total_cost"] += cost
        sim["tokens"] += tokens
        inc["cost"] += cost
        inc["tokens"] += tokens
    inc["events"].append(ev)
    sim["feed"].append(ev)
    if len(sim["feed"]) > 140:
        sim["feed"] = sim["feed"][-140:]

def audit(sim, actor, action, inc_id, detail, result="ok"):
    sim["audit"].append({"t": sim["clock"], "actor": actor, "action": action,
                         "inc": inc_id, "detail": detail, "result": result})
    if len(sim["audit"]) > 200:
        sim["audit"] = sim["audit"][-200:]

def build_pr(rng, inc):
    f = FAULTS[inc["type"]]
    p = PIPE_BY_ID[inc["pipeline"]]
    diff = (f["diff"].replace("__MODEL__", p["model"]).replace("__TABLE__", p["table"])
            .replace("__PIPE__", p["id"]).replace("__ID__", inc["id"].lower()))
    v = inc["attempt"]
    if v > 1:
        diff = f"# strategy v{v}: quarantine-first, reduced blast radius\n" + diff
    risk = int(inc["risk"] * (0.7 if v > 1 else 1.0))
    return {"branch": f"fix/{inc['id'].lower()}-{inc['type'].replace('_', '-')}",
            "title": f"[PipeMedic] {f['label']} on {p['id']} — reversible patch (v{v})",
            "diff": diff, "files": 2, "adds": rng.randint(6, 18), "dels": rng.randint(2, 9),
            "risk": risk,
            "rollback_plan": f"Snapshot snap_{inc['id'].lower()}_pre → revert commit → restore → re-run DQ suite (ETA ~4 min)"}

def checks_for(inc):
    p = PIPE_BY_ID[inc["pipeline"]]
    silent = inc.get("silent_wrong", False)
    return [("Row-count delta ≤ 0.5%", "PASS", "0.03%"),
            ("dbt DQ suite", "PASS", "42/42"),
            ("Freshness SLA", "PASS", f"{p['sla_min']}m target met"),
            ("Partition checksums", "WARN" if silent else "PASS",
             "semantic mismatch → audit ticket" if silent else "128/128")]

def build_report_md(inc):
    p = PIPE_BY_ID[inc["pipeline"]]
    f = FAULTS[inc["type"]]
    marks = inc["marks"]
    rows = []
    for key, label in [("opened", "Detected"), ("classified", "Classified"), ("diagnosed", "Diagnosed"),
                       ("pr", "Repair PR opened"), ("approved", "Approved"), ("staged", "Staging verified"),
                       ("promoted", "Promoted to prod"), ("verified", "Recovery verified"), ("closed", "Postmortem filed")]:
        t = marks.get(key)
        if t:
            rows.append(f"| {label} | {t.strftime('%H:%M:%S')} UTC |")
    dur = ((marks["verified"] - marks["opened"]).total_seconds() / 60.0) if marks.get("verified") else 0.0
    checks = "\n".join(f"| {n} | {r} | {d} |" for n, r, d in checks_for(inc))
    return f"""# 🧾 Recovery Run Report — {inc['id']}

**Pipeline** `{p['id']}` · **Table** `{p['table']}` · **Owner** {p['team']} · **Tier** {p['tier']} · **Severity** {inc['severity']}

**Diagnosis:** {inc.get('diagnosis_text') or 'n/a'}  
**Confidence:** {inc.get('confidence', 0):.0%} · **Fix risk:** {inc.get('pr', {}).get('risk', 'n/a')}/100 · **Rollback occurred:** {'yes (auto)' if inc.get('rolled_back') else 'no'}

## Timeline
| Phase | At |
|---|---|
{chr(10).join(rows)}

## Verification
| Check | Result | Detail |
|---|---|---|
{checks}

**Outcome:** recovered in **{dur:.1f} min** · data downtime **{inc['downtime']:.1f} min** · agent cost **${inc['cost']:.2f}** · tokens {inc['tokens']:,}

*Repair branch:* `{inc.get('pr', {}).get('branch', 'n/a')}` · snapshot retained 24h · generated by PipeMedic {APP_VER} (simulated demo).
"""

def build_postmortem_md(inc):
    p = PIPE_BY_ID[inc["pipeline"]]
    f = FAULTS[inc["type"]]
    dur = ((inc["marks"]["verified"] - inc["marks"]["opened"]).total_seconds() / 60.0) if inc["marks"].get("verified") else 0.0
    prev = "\n".join(f"- [ ] {item}" for item in f["prevention"])
    silent_note = ("\n> ⚠️ **Post-hoc audit:** semantic checksum flagged a silent mismatch after close. "
                   "A follow-up incident and a contract test were filed automatically.\n"
                   if inc.get("silent_wrong") else "")
    rb_note = ("\n- Conservative auto-rollback fired when verification failed; the agent re-investigated "
               "with the rollback evidence and landed a correct fix.\n" if inc.get("rolled_back") else "")
    return f"""## 📜 Postmortem — {inc['id']} · {f['label']} on `{p['id']}`

### Impact
- Dataset `{p['table']}` stale for **{inc['downtime']:.0f} min** (SLA {p['sla_min']} min, tier {p['tier']}).
- Downstream affected: {', '.join('`' + d + '`' for d in p['downstream'])}.
- Detected → recovered: **{dur:.1f} min** (manual baseline ≈ 252 min).
{silent_note}
### Root cause
{f['cause']}.

**Evidence:** {f['signal']}.

### Repair
{inc.get('pr', {}).get('title', 'Manual fix by on-call.')}
Applied staging-first with a pre-change snapshot; four verification checks gated promotion.
{rb_note}
### What went well
- Signal grouped and classified in under 3 minutes; lineage bounded the blast radius before any write.
- Every write carried an undo plan; the snapshot made rollback a 4-minute operation.

### What got lucky
- The failure landed outside the finance-close freeze window.

### Prevention (auto-filed)
{prev}

*Owners: {p['team']} + Platform SRE · written by module M4, PipeMedic {APP_VER} (simulated demo).*
"""

def new_incident(sim, rng, pipe_id, fault, severity):
    p = PIPE_BY_ID[pipe_id]
    f = FAULTS[fault]
    inc = {"id": f"INC-{sim['next_id']}", "pipeline": pipe_id, "type": fault, "truth": fault,
           "severity": severity, "tier": p["tier"], "status": "active", "stage": "detect",
           "stage_dur": None, "stage_elapsed": 0.0, "attempt": 1, "re_diagnosis": False,
           "diagnosed": None, "diagnosis_text": None, "confidence": None, "risk": None,
           "pr": None, "feedback": None, "rolled_back": False, "silent_wrong": False,
           "gate_reason": None, "recovered_at": None, "closed_at": None,
           "downtime": 0.0, "cost": 0.0, "tokens": 0, "events": [],
           "marks": {"opened": sim["clock"]}, "report": None, "postmortem": None}
    sim["next_id"] += 1
    set_stage(inc, "detect", rng)
    emit(sim, inc, rng, "M1", "observe", f"Signal on {pipe_id}: {f['signal']}", tool="read_orchestrator_logs")
    sim["incidents"].append(inc)
    return inc

def gate_reason(sim, inc):
    p = sim["policy"]
    if p["kill"]:
        return ("held_kill", "Kill switch engaged — all write actions frozen")
    if sim["day_cost"] >= p["budget"]:
        return ("held_budget", f"Daily budget ${p['budget']:.0f} reached — writes paused")
    if p["mode"] == "Suggest only":
        return ("awaiting_approval", "Policy: suggest-only — a human must apply every repair")
    conf = inc.get("confidence") or 0.0
    risk = (inc.get("pr") or {}).get("risk", inc.get("risk") or 50)
    if conf < p["conf_floor"]:
        return ("awaiting_approval", f"Confidence {conf:.0%} below floor {p['conf_floor']:.0%}")
    if risk > p["max_risk"]:
        return ("awaiting_approval", f"Fix risk {risk} exceeds auto-cap {p['max_risk']}")
    if p["mode"] == "Act with approval":
        return ("awaiting_approval", "Policy: every repair plan needs approval")
    if p["mode"] == "Act & notify" and inc["tier"] == 1:
        return ("awaiting_approval", "Tier-1 pipeline — approval required at this autonomy level")
    return None

def start_rollback(sim, inc, rng, reason, actor):
    sim["rollbacks"] += 1
    inc["rolled_back"] = True
    inc["status"] = "rolling_back"
    inc["stage_elapsed"] = 0.0
    inc["stage_dur"] = 2.0
    inc["recovered_at"] = None
    inc["closed_at"] = None
    inc["marks"]["verified"] = None
    inc["marks"]["closed"] = None
    emit(sim, inc, rng, "M3", "rollback", f"ROLLBACK: restoring snapshot snap_{inc['id'].lower()}_pre — {reason}",
         tool="warehouse.revert_to_snapshot")
    audit(sim, actor, "rollback", inc["id"], reason, result="restored")

def finish_rollback(sim, inc, rng):
    emit(sim, inc, rng, "M3", "observe", "Snapshot restored; pipeline re-running from last good watermark")
    inc["attempt"] += 1
    inc["re_diagnosis"] = True
    inc["diagnosed"] = None
    inc["status"] = "active"
    emit(sim, inc, rng, "M2", "reflect", "Hypothesis rejected — expanding log window and re-investigating with rollback evidence")
    set_stage(inc, "investigate", rng)

def complete_stage(sim, inc, rng):
    f = FAULTS[inc["type"]]
    p = PIPE_BY_ID[inc["pipeline"]]
    stage = inc["stage"]
    agent = f"agent:{APP_VER}"

    if stage == "detect":
        emit(sim, inc, rng, "M1", "plan", "Grouping signals; fetching DAG metadata + recent orchestrator logs",
             tool="fetch_dag_metadata", tokens=350)
        set_stage(inc, "classify", rng)

    elif stage == "classify":
        inc["marks"]["classified"] = sim["clock"]
        emit(sim, inc, rng, "M1", "act",
             f"Classified: {f['label']} · {inc['severity']} · tier {inc['tier']} · dedupe group g-{inc['id'].lower()}",
             tokens=600)
        set_stage(inc, "investigate", rng)

    elif stage == "investigate":
        emit(sim, inc, rng, "M2", "act",
             f"Lineage trace: {len(p['upstream'])} upstream / {len(p['downstream'])} downstream nodes; blast radius bounded",
             tool="lineage.trace", tokens=1200)
        emit(sim, inc, rng, "M2", "observe", f"Log evidence: {f['signal']}", tool="read_orchestrator_logs", tokens=900)
        emit(sim, inc, rng, "M2", "act", "Scanned warehouse query history + DQ results for the failing window",
             tool="query_history.scan", tokens=1700)
        set_stage(inc, "diagnose", rng)

    elif stage == "diagnose":
        correct_p = 0.99 if inc["re_diagnosis"] else f["difficulty"]
        correct = rng.random() < correct_p
        inc["diagnosed"] = inc["truth"] if correct else f["wrong_alt"]
        believed = FAULTS[inc["diagnosed"]]
        mu = 0.95 if inc["re_diagnosis"] else (0.90 if correct else 0.77)
        inc["confidence"] = clamp(rng.gauss(mu, 0.055 if not inc["re_diagnosis"] else 0.02), 0.55, 0.99)
        inc["risk"] = rng.randint(*believed["risk"])
        inc["diagnosis_text"] = f"{believed['cause']} — evidence: {believed['signal']}"
        inc["marks"]["diagnosed"] = sim["clock"]
        emit(sim, inc, rng, "M2", "reflect",
             f"Root cause: {believed['cause']} (confidence {inc['confidence']:.0%})", tokens=2600)
        set_stage(inc, "propose", rng)

    elif stage == "propose":
        inc["pr"] = build_pr(rng, inc)
        inc["marks"]["pr"] = sim["clock"]
        emit(sim, inc, rng, "M3", "write",
             f"Opened PR {inc['pr']['branch']} (+{inc['pr']['adds']}/−{inc['pr']['dels']}, {inc['pr']['files']} files) · "
             f"risk {inc['pr']['risk']} · undo plan attached", tool="git.open_pr", tokens=3200)
        audit(sim, agent, "open_pr", inc["id"], inc["pr"]["branch"])
        set_stage(inc, "gate", rng)

    elif stage == "gate":
        reason = gate_reason(sim, inc)
        if reason:
            inc["status"], inc["gate_reason"] = reason
            inc["stage_dur"] = None
            emit(sim, inc, rng, "M3", "gate", f"Repair plan paused: {reason[1]}")
            return
        inc["marks"]["approved"] = sim["clock"]
        emit(sim, inc, rng, "POLICY", "gate",
             f"Auto-approved ({sim['policy']['mode']}): conf {inc['confidence']:.0%} ≥ floor, "
             f"risk {inc['pr']['risk']} ≤ cap {sim['policy']['max_risk']}")
        audit(sim, "policy-engine", "auto-approve", inc["id"], sim["policy"]["mode"])
        set_stage(inc, "staging", rng)

    elif stage == "staging":
        emit(sim, inc, rng, "M3", "write",
             f"Applied patch in staging (clone {p['table']} → staging_{inc['id'].lower()})",
             tool="warehouse.apply_patch_staging", tokens=500)
        emit(sim, inc, rng, "M3", "verify", "Staging rehearsal: 42/42 DQ checks · row-count Δ 0.03%", tokens=400)
        inc["marks"]["staged"] = sim["clock"]
        audit(sim, agent, "staging_apply", inc["id"], "42/42 checks pass")
        set_stage(inc, "apply", rng)

    elif stage == "apply":
        sim["applies"] += 1
        inc["marks"]["promoted"] = sim["clock"]
        emit(sim, inc, rng, "M3", "write",
             f"Promoted to production · snapshot snap_{inc['id'].lower()}_pre retained 24h",
             tool="warehouse.promote_patch", tokens=400)
        emit(sim, inc, rng, "SYS", "notify",
             f"Slack #data-incidents: recovery in progress for {inc['pipeline']} ({inc['id']})", tool="slack.notify")
        audit(sim, agent, "promote", inc["id"], "snapshot retained 24h")
        set_stage(inc, "verify", rng)

    elif stage == "verify":
        correct = inc["diagnosed"] == inc["truth"]
        ok = inc.pop("_force_ok", False) or rng.random() < (0.97 if correct else 0.12)
        if ok:
            if not correct:
                inc["silent_wrong"] = True
            inc["marks"]["verified"] = sim["clock"]
            inc["recovered_at"] = sim["clock"]
            inc["status"] = "recovered"
            dur = (inc["marks"]["verified"] - inc["marks"]["opened"]).total_seconds() / 60.0
            emit(sim, inc, rng, "M3", "verify",
                 "Verification green: freshness ≤ SLA · checksums 128/128 · DQ suite pass", tokens=700)
            emit(sim, inc, rng, "M2", "reflect", f"Repair validated · MTTR {dur:.1f}m · downtime {inc['downtime']:.1f}m")
            set_stage(inc, "postmortem", rng)
        else:
            why = "verification flake — conservative rollback" if correct else "verification failed — diagnosis hypothesis rejected"
            start_rollback(sim, inc, rng, why, agent)

    elif stage == "postmortem":
        inc["report"] = build_report_md(inc)
        inc["postmortem"] = build_postmortem_md(inc)
        inc["marks"]["closed"] = sim["clock"]
        inc["closed_at"] = sim["clock"]
        inc["status"] = "closed"
        if inc["silent_wrong"]:
            emit(sim, inc, rng, "M4", "reflect",
                 "Post-hoc semantic audit flagged a mismatch — follow-up ticket + contract test auto-filed")
            audit(sim, agent, "semantic_audit", inc["id"], "mismatch flagged", result="warn")
        emit(sim, inc, rng, "M4", "act", "Postmortem published; 3 prevention items filed to the runbook",
             tool="git.open_pr", tokens=4100)
        audit(sim, agent, "postmortem", inc["id"], "3 prevention items")
        set_stage(inc, "closed", rng)

def compute_metrics(sim):
    incs = sim["incidents"]
    closedish = ("closed", "manual_closed")
    active = [i for i in incs if i["status"] not in closedish]
    awaiting = [i for i in incs if i["status"] in ("awaiting_approval", "held_kill", "held_budget")]
    durs = [(i["marks"]["verified"] - i["marks"]["opened"]).total_seconds() / 60.0
            for i in incs if i["marks"].get("verified")]
    mttr = float(np.mean(durs[-20:])) if durs else 0.0
    pool = [i for i in incs if i.get("diagnosed") and i["marks"].get("verified")]
    good = [i for i in pool if i["diagnosed"] == i["truth"] and i.get("feedback") != "down"]
    acc = 100.0 * len(good) / len(pool) if pool else 100.0
    rb = 100.0 * sim["rollbacks"] / max(1, sim["applies"])
    win = sim["clock"] - timedelta(hours=48)
    down = sum(i["downtime"] for i in incs if i["marks"]["opened"] >= win) / 60.0
    return {"mttr": mttr, "acc": acc, "rb": rb, "down": down, "active": len(active),
            "awaiting": len(awaiting), "cost": sim["day_cost"]}

def advance(sim, rng, dt):
    sim["clock"] += timedelta(minutes=dt)
    if sim["clock"].date() != sim["day"]:
        sim["day"] = sim["clock"].date()
        sim["day_cost"] = 0.0
    p = sim["policy"]
    closedish = ("closed", "manual_closed")

    for inc in sim["incidents"]:
        s = inc["status"]
        if s in closedish:
            continue
        if s in ("held_kill", "held_budget"):
            r = gate_reason(sim, inc)
            if r is None:
                inc["status"] = "active"
                inc["gate_reason"] = None
                set_stage(inc, "gate", rng)
                emit(sim, inc, rng, "POLICY", "gate", "Hold released — resuming repair plan")
            elif r[0] != s:
                inc["status"], inc["gate_reason"] = r
            inc["downtime"] += dt
            continue
        if s in ("awaiting_approval", "escalated"):
            inc["downtime"] += dt
            continue
        if s == "rolling_back":
            inc["stage_elapsed"] += dt
            inc["downtime"] += dt
            if inc["stage_elapsed"] >= 2.0:
                finish_rollback(sim, inc, rng)
            continue
        if s == "recovered":
            inc["stage_elapsed"] += dt
            if inc["stage_dur"] is not None and inc["stage_elapsed"] >= inc["stage_dur"]:
                complete_stage(sim, inc, rng)
            continue
        # active
        inc["stage_elapsed"] += dt
        guard = 0
        while (inc["status"] == "active" and inc["stage_dur"] is not None
               and inc["stage_elapsed"] >= inc["stage_dur"] and guard < 12):
            inc["stage_elapsed"] -= inc["stage_dur"]
            complete_stage(sim, inc, rng)
            guard += 1
        if inc["status"] not in ("recovered", "closed", "manual_closed"):
            inc["downtime"] += dt

    # spawn new incidents
    rate = p["rate"] * (2.5 if p["chaos"] else 1.0)
    active_n = sum(1 for i in sim["incidents"] if i["status"] not in closedish)
    if rate > 0 and active_n < 7 and rng.random() < rate * dt / 60.0:
        pipe = rng.choice(PIPELINES)["id"]
        fault = rng.choices(FAULT_KEYS, FAULT_WEIGHTS)[0]
        dup = any(i["pipeline"] == pipe and i["type"] == fault and i["status"] not in closedish
                  for i in sim["incidents"])
        if dup:
            sim["suppressed"] += 1
            sim["feed"].append({"t": sim["clock"], "inc": "SYS", "module": "M1", "kind": "observe",
                                "text": f"Deduplicated repeated {FAULTS[fault]['label']} signal on {pipe}",
                                "tool": None, "ms": 12})
        else:
            sev = rng.choices(["P1", "P2", "P3"], [0.2, 0.5, 0.3])[0]
            new_incident(sim, rng, pipe, fault, sev)
            sim["spawned"] += 1

    m = compute_metrics(sim)
    sim["history"].append({"t": sim["clock"], **m})
    if len(sim["history"]) > 600:
        sim["history"] = sim["history"][-600:]

# ---------------------------------------------------------------- human actions
def mutate_approve(sim, inc, actor="human:you"):
    rng = get_rng()
    inc["marks"]["approved"] = sim["clock"]
    inc["status"] = "active"
    inc["gate_reason"] = None
    set_stage(inc, "staging", rng)
    emit(sim, inc, rng, "HUMAN", "act", f"Repair plan approved by {actor} — proceeding to staging")
    audit(sim, actor, "approve", inc["id"], inc["pr"]["branch"] if inc["pr"] else "plan")

def mutate_reject(sim, inc):
    rng = get_rng()
    inc["attempt"] += 1
    audit(sim, "human:you", "reject", inc["id"], f"strategy v{inc['attempt'] - 1} rejected")
    if inc["attempt"] >= 3:
        inc["status"] = "escalated"
        inc["stage_dur"] = None
        emit(sim, inc, rng, "HUMAN", "gate", "Second rejection — escalated to on-call; agent keeps monitoring")
    else:
        inc["status"] = "active"
        emit(sim, inc, rng, "M3", "reflect",
             f"Plan rejected — replanning with lower blast radius (strategy v{inc['attempt']})")
        set_stage(inc, "propose", rng)

def rollbackable(sim, inc):
    return (inc.get("recovered_at") is not None and not inc.get("rolled_back")
            and inc["status"] in ("recovered", "closed")
            and (sim["clock"] - inc["recovered_at"]) <= timedelta(minutes=60))

def mutate_escalate(sim, inc):
    inc["status"] = "escalated"
    inc["stage_dur"] = None
    emit(sim, inc, get_rng(), "HUMAN", "gate", "Escalated to on-call by operator")
    audit(sim, "human:you", "escalate", inc["id"], "manual escalation")

def mutate_resolve(sim, inc):
    rng = get_rng()
    inc["marks"]["verified"] = sim["clock"]
    inc["marks"]["closed"] = sim["clock"]
    inc["recovered_at"] = sim["clock"]
    inc["closed_at"] = sim["clock"]
    inc["status"] = "closed"
    if not inc.get("diagnosed"):
        inc["diagnosed"] = inc["truth"]
        inc["diagnosis_text"] = FAULTS[inc["truth"]]["cause"] + " (confirmed by on-call)"
        inc["confidence"] = 1.0
    inc["report"] = build_report_md(inc)
    inc["postmortem"] = build_postmortem_md(inc)
    emit(sim, inc, rng, "HUMAN", "verify", "Manual fix applied by on-call — agent verified freshness + DQ suite")
    audit(sim, "human:you", "manual_resolve", inc["id"], "verified by agent")

def after_action(msg):
    st.toast(msg)
    st.rerun()

# ---------------------------------------------------------------- seeding
def _seed_events(sim, inc, rng):
    f = FAULTS[inc["type"]]
    m = inc["marks"]
    seq = [("opened", "M1", "observe", f"Signal: {f['signal']}"),
           ("classified", "M1", "act", f"Classified: {f['label']} · {inc['severity']} · group g-{inc['id'].lower()}"),
           ("diagnosed", "M2", "reflect", f"Root cause: {FAULTS[inc['diagnosed']]['cause']} (confidence {inc['confidence']:.0%})"),
           ("pr", "M3", "write", f"Opened PR {inc['pr']['branch']} · risk {inc['pr']['risk']} · undo plan attached"),
           ("approved", "POLICY", "gate", "Auto-approved: confidence ≥ floor, risk ≤ cap"),
           ("staged", "M3", "verify", "Staging rehearsal passed 42/42 DQ checks"),
           ("promoted", "M3", "write", "Promoted patch to production; snapshot retained 24h"),
           ("verified", "M3", "verify", "Recovery verified: freshness + row counts + checksums green"),
           ("closed", "M4", "act", "Postmortem published; 3 prevention items filed")]
    for key, mod, kind, text in seq:
        if m.get(key):
            inc["events"].append({"t": m[key], "inc": inc["id"], "module": mod, "kind": kind,
                                  "text": text, "tool": None, "ms": rng.randint(40, 2400)})
    if inc["rolled_back"]:
        t = m["promoted"] + timedelta(minutes=2)
        inc["events"].append({"t": t, "inc": inc["id"], "module": "M3", "kind": "rollback",
                              "text": "Verification failed — auto-rollback to snapshot; re-investigating",
                              "tool": "warehouse.revert_to_snapshot", "ms": 800})
    if inc["silent_wrong"]:
        inc["events"].append({"t": m["closed"], "inc": inc["id"], "module": "M4", "kind": "reflect",
                              "text": "Post-hoc audit flagged semantic mismatch — follow-up opened",
                              "tool": None, "ms": 300})
    inc["events"].sort(key=lambda e: e["t"])
    sim["feed"].extend(inc["events"])

def seed_sim(seed):
    rng = random.Random(seed)
    now = now_utc()
    sim = {"seed": seed, "clock": now, "day": now.date(), "running": True,
           "incidents": [], "next_id": 1000, "feed": [], "history": [], "audit": [],
           "rollbacks": 0, "applies": 0, "day_cost": 0.0, "total_cost": 0.0, "tokens": 0,
           "spawned": 0, "suppressed": 0, "policy": dict(DEFAULT_POLICY)}

    closed = []
    for k in range(16):
        p = rng.choice(PIPELINES)
        fault = rng.choices(FAULT_KEYS, FAULT_WEIGHTS)[0]
        f = FAULTS[fault]
        rolled = k in (3, 9)
        silent = k == 6
        t_open = now - timedelta(minutes=rng.uniform(150, 2860))
        dur = rng.uniform(14, 38)
        marks = {"opened": t_open,
                 "classified": t_open + timedelta(minutes=2),
                 "diagnosed": t_open + timedelta(minutes=dur * 0.3),
                 "pr": t_open + timedelta(minutes=dur * 0.42),
                 "approved": t_open + timedelta(minutes=dur * 0.47),
                 "staged": t_open + timedelta(minutes=dur * 0.6),
                 "promoted": t_open + timedelta(minutes=dur * 0.72),
                 "verified": t_open + timedelta(minutes=dur),
                 "closed": t_open + timedelta(minutes=dur + 5)}
        tokens = rng.randint(9000, 16000)
        inc = {"id": "", "pipeline": p["id"], "type": fault, "truth": fault,
               "severity": rng.choices(["P1", "P2", "P3"], [0.2, 0.5, 0.3])[0],
               "tier": p["tier"], "status": "closed", "stage": "closed",
               "stage_dur": None, "stage_elapsed": 0.0, "attempt": 2 if rolled else 1,
               "re_diagnosis": False,
               "diagnosed": f["wrong_alt"] if silent else fault,
               "confidence": clamp(rng.gauss(0.78 if silent else 0.9, 0.05), 0.6, 0.98),
               "risk": rng.randint(*f["risk"]), "pr": None, "feedback": None,
               "rolled_back": rolled, "silent_wrong": silent, "gate_reason": None,
               "recovered_at": marks["verified"], "closed_at": marks["closed"],
               "downtime": dur * 0.85 + (6 if rolled else 0),
               "cost": tokens / 1000.0 * 0.004 + 0.02, "tokens": tokens,
               "events": [], "marks": marks, "report": None, "postmortem": None}
        inc["diagnosis_text"] = FAULTS[inc["diagnosed"]]["cause"] + " — evidence: " + FAULTS[inc["diagnosed"]]["signal"]
        closed.append(inc)

    closed.sort(key=lambda i: i["marks"]["opened"])
    for inc in closed:
        inc["id"] = f"INC-{sim['next_id']}"
        sim["next_id"] += 1
        inc["pr"] = build_pr(rng, inc)
        inc["report"] = build_report_md(inc)
        inc["postmortem"] = build_postmortem_md(inc)
        _seed_events(sim, inc, rng)
        sim["incidents"].append(inc)
        sim["tokens"] += inc["tokens"]
        sim["total_cost"] += inc["cost"]
        if inc["marks"]["opened"] >= now - timedelta(hours=24):
            sim["day_cost"] += inc["cost"]
        actor = "human:on-call" if rng.random() < 0.4 else f"agent:{APP_VER}"
        audit(sim, actor, "promote", inc["id"], "snapshot retained 24h")
        if inc["rolled_back"]:
            audit(sim, f"agent:{APP_VER}", "rollback", inc["id"], "verification failed", result="restored")
    sim["applies"] = 20
    sim["rollbacks"] = 2

    # live incidents via the real engine, fast-forwarded
    saved = dict(sim["policy"])

    def ff(pipe, fault, sev, age_min, target, mode):
        sim["policy"].update(mode=mode, conf_floor=0.0, max_risk=100, kill=False, budget=1e9)
        sim["clock"] = now - timedelta(minutes=age_min)
        inc = new_incident(sim, rng, pipe, fault, sev)
        g = 0
        while g < 40 and inc["status"] == "active" and inc["stage"] != target:
            sim["clock"] += timedelta(minutes=(inc["stage_dur"] or 0.5) * 0.42)
            complete_stage(sim, inc, rng)
            g += 1
        inc["stage_elapsed"] = rng.uniform(0, (inc["stage_dur"] or 1.0) * 0.5)
        return inc

    ff("payments_ledger", "schema_drift", "P1", 11, "__await__", "Act & notify")     # waits for you
    ff("clickstream_sessions", "upstream_delay", "P2", 4, "investigate", "Act & notify")
    ff("marketing_attribution", "dq_null_spike", "P3", 10, "staging", "Act & notify")
    d = ff("inventory_snapshot", "late_data", "P2", 18, "postmortem", "Fully autonomous")
    d["_force_ok"] = True  # already recovered before you arrived; rollback button live

    sim["policy"] = saved
    sim["clock"] = now
    sim["feed"].sort(key=lambda e: e["t"])
    sim["feed"] = sim["feed"][-140:]

    # 48h of metric history so charts open populated
    for h in range(48, 0, -1):
        t = now - timedelta(hours=h)
        rec = [i for i in closed if i["marks"]["verified"] <= t]
        durs = [(i["marks"]["verified"] - i["marks"]["opened"]).total_seconds() / 60.0 for i in rec]
        mttr = float(np.mean(durs[-20:])) if durs else 24.0
        good = [i for i in rec if i["diagnosed"] == i["truth"]]
        acc = 100.0 * len(good) / len(rec) if rec else 94.0
        rolled = sum(1 for i in rec if i["rolled_back"])
        rb = 100.0 * rolled / max(1, len(rec) + rolled)
        down = sum(i["downtime"] for i in closed if i["marks"]["opened"] <= t) / 60.0
        sim["history"].append({"t": t, "mttr": mttr, "acc": acc, "rb": rb, "down": down,
                               "active": rng.randint(0, 2), "awaiting": 0,
                               "cost": sim["day_cost"] * (48 - h) / 48.0})
    sim["history"].append({"t": now, **compute_metrics(sim)})
    return sim, rng

# ---------------------------------------------------------------- init / reset
if st.session_state.pop("_do_reset", False):
    seed = int(st.session_state.get("sb_seed", 7))
    for k in list(st.session_state.keys()):
        if k.startswith(("sb_", "f_", "inc_", "rep_", "sh_", "det_", "ops_", "in_")) and k != "sb_seed":
            del st.session_state[k]
    sim, rng = seed_sim(seed)
    st.session_state.sim = sim
    st.session_state.rng = rng
    st.session_state["_last_real"] = time.time()

if "sim" not in st.session_state:
    sim, rng = seed_sim(7)
    st.session_state.sim = sim
    st.session_state.rng = rng
    st.session_state["_last_real"] = time.time()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div style="font-size:1.45rem;font-weight:800;">⛑️ <span class="grad">PipeMedic</span></div>',
                unsafe_allow_html=True)
    st.caption("Autonomous Data Pipeline Repair Agent · live sandbox")
    sim = st.session_state.sim
    if st_btn("⏸ Pause simulation" if sim["running"] else "▶️ Resume simulation", key="sb_run", kind="primary"):
        sim["running"] = not sim["running"]
        st.session_state["_last_real"] = time.time()
        st.rerun()
    sb_speed = st.slider("Simulation speed (×)", 0.5, 4.0, 2.0, 0.25, key="sb_speed")
    sb_rate = st.slider("Fault injection rate (per sim-hour)", 0, 12, 3, 1, key="sb_rate")
    sb_chaos = st.toggle("🐒 Chaos monkey (2.5× faults)", value=False, key="sb_chaos")
    st.caption("1 real second ≈ 1 sim-minute at 1×")
    st.divider()
    st.markdown("**🛡 Autonomy policy**")
    sb_mode = st.selectbox("Autonomy level", MODES, index=2, key="sb_mode")
    sb_conf = st.slider("Confidence floor", 0.50, 0.99, 0.80, 0.01, key="sb_conf")
    sb_risk = st.slider("Max auto-fix risk", 0, 100, 40, 5, key="sb_risk")
    sb_budget = st.number_input("Daily agent budget ($)", 1, 500, 25, 1, key="sb_budget")
    sb_kill = st.toggle("🛑 KILL SWITCH — freeze writes", value=False, key="sb_kill")
    if sb_kill:
        st.markdown('<span class="bad">All write actions frozen. Diagnostics continue.</span>',
                    unsafe_allow_html=True)
    st.divider()
    st.markdown("**🧪 Inject an incident**")
    sb_pipe = st.selectbox("Pipeline", [p["id"] for p in PIPELINES], key="sb_pipe")
    sb_fault = st.selectbox("Failure type", FAULT_KEYS,
                            format_func=lambda k: f"{FAULTS[k]['icon']} {FAULTS[k]['label']}", key="sb_fault")
    sb_sev = st.selectbox("Severity", ["P1", "P2", "P3"], index=1, key="sb_sev")
    if st_btn("💥 Inject now", key="sb_inject", kind="primary"):
        dup = any(i["pipeline"] == sb_pipe and i["type"] == sb_fault
                  and i["status"] not in ("closed", "manual_closed") for i in sim["incidents"])
        if dup:
            sim["suppressed"] += 1
            st.toast("Duplicate signal — M1 deduplicated it (watch the feed)")
        else:
            inc = new_incident(sim, get_rng(), sb_pipe, sb_fault, sb_sev)
            sim["spawned"] += 1
            st.toast(f"💥 Injected {FAULTS[sb_fault]['label']} on {sb_pipe} → {inc['id']}")
        st.rerun()
    st.divider()
    sb_seed = st.number_input("Demo seed", 0, 9999, 7, 1, key="sb_seed")
    if st_btn("🔁 Reset & reseed", key="sb_reset"):
        st.session_state["_do_reset"] = True
        st.rerun()
    st.caption(f"agent {APP_VER} · model router: cost/latency/conf · source & clipboard in 📤 Share & Export")

sim["policy"].update({"speed": float(sb_speed), "rate": float(sb_rate), "chaos": bool(sb_chaos),
                      "mode": sb_mode, "conf_floor": float(sb_conf), "max_risk": int(sb_risk),
                      "budget": float(sb_budget), "kill": bool(sb_kill)})

# ---------------------------------------------------------------- charts & views
def fig_trend(sim):
    df = pd.DataFrame(sim["history"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["mttr"], name="MTTR (min)",
                             line=dict(color=PAL["cyan"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(8,145,178,.10)"))
    fig.add_trace(go.Scatter(x=df["t"], y=df["acc"], name="Diagnosis %", yaxis="y2",
                             line=dict(color=PAL["green"], width=2, dash="dot")))
    fig.update_layout(hovermode="x unified",
                      yaxis=dict(title="min"),
                      yaxis2=dict(overlaying="y", side="right", range=[60, 101],
                                  showgrid=False, title="%"))
    return style_fig(fig, 280)

def fig_donut(sim):
    win = sim["clock"] - timedelta(hours=48)
    incs = [i for i in sim["incidents"] if i["marks"]["opened"] >= win]
    counts = pd.Series([FAULTS[i["type"]]["label"] for i in incs]).value_counts()
    colors = [PAL[FAULTS[k]["color"]] for k in FAULT_KEYS
              if FAULTS[k]["label"] in counts.index]
    color_map = {FAULTS[k]["label"]: PAL[FAULTS[k]["color"]] for k in FAULT_KEYS}
    fig = go.Figure(go.Pie(labels=list(counts.index), values=list(counts.values), hole=0.62,
                           marker=dict(colors=[color_map[l] for l in counts.index]),
                           textinfo="label+value", textfont=dict(size=10), sort=False))
    fig.update_layout(showlegend=False)
    return style_fig(fig, 280)

def fig_downtime(sim):
    win = sim["clock"] - timedelta(hours=48)
    rows = {}
    for i in sim["incidents"]:
        if i["marks"]["opened"] >= win:
            rows[i["pipeline"]] = rows.get(i["pipeline"], 0.0) + i["downtime"] / 60.0
    s = pd.Series(rows).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=list(s.index), orientation="h",
                           marker=dict(color=s.values,
                                       colorscale=[[0, PAL["amber"]], [1, PAL["red"]]]),
                           text=[f"{v:.2f}h" for v in s.values], textposition="outside"))
    fig.update_layout(xaxis_title="hours")
    return style_fig(fig, 280)

def fig_gauge(conf, floor):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=(conf or 0) * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#5B6B87"},
               "bar": {"color": PAL["violet"]},
               "bgcolor": "rgba(0,0,0,0)",
               "steps": [{"range": [0, floor * 100], "color": "rgba(225,29,72,.14)"},
                         {"range": [floor * 100, 100], "color": "rgba(5,150,105,.10)"}],
               "threshold": {"line": {"color": PAL["red"], "width": 3},
                             "thickness": 0.8, "value": floor * 100}}))
    return style_fig(fig, 210)

def lineage_dot(inc):
    p = PIPE_BY_ID[inc["pipeline"]]
    ok = inc["status"] in ("recovered", "closed", "manual_closed")
    tgt_fill = "#D1FAE5" if ok else "#FFE4E6"
    tgt_edge = "#059669" if ok else "#E11D48"
    lines = ["digraph G {", "rankdir=LR;", 'bgcolor="transparent";',
             'node [shape=box,style="rounded,filled",fillcolor="#FFFFFF",color="#94A3B8",'
             'fontcolor="#1E293B",fontname="monospace",fontsize=11];',
             'edge [color="#94A3B8"];']
    cause_up = (p["upstream"][0] if inc["type"] in ("upstream_delay", "credential_expiry")
                and p["upstream"] else None)
    for u in p["upstream"]:
        extra = ',fillcolor="#FEF3C7",color="#D97706"' if u == cause_up else ""
        lines.append(f'"{u}" [shape=box{extra}];')
        lines.append(f'"{u}" -> "{p["id"]}";')
    lines.append(f'"{p["id"]}" [fillcolor="{tgt_fill}",color="{tgt_edge}",penwidth=2];')
    for d in p["downstream"]:
        lines.append(f'"{p["id"]}" -> "{d}";')
        lines.append(f'"{d}" [style="rounded,filled,dashed"];')
    lines.append("}")
    return "\n".join(lines)

def hero_html(sim):
    p = sim["policy"]
    dot = "kill" if p["kill"] else ("" if sim["running"] else "paused")
    state = "KILL SWITCH" if p["kill"] else ("LIVE SIMULATION" if sim["running"] else "PAUSED")
    clock = sim["clock"].strftime("%Y-%m-%d %H:%M:%S UTC")
    return (f'<div class="hero"><h1>⛑️ <span class="grad">PipeMedic</span> '
            f'<span style="font-size:.85rem;color:#5B6B87;font-weight:600;">· Autonomous Data Pipeline '
            f'Repair Agent {APP_VER}</span></h1>'
            f'<div class="sub">monitor → diagnose → repair (staged, reversible) → verify → postmortem · '
            f'for Data Eng · Analytics Eng · Platform SRE</div>'
            f'<div class="right"><span class="dot {dot}"></span>{state}<br>{clock}<br>'
            f'<span class="mut">{html.escape(p["mode"])} · {p["speed"]:.2g}×</span></div></div>')

def inc_header_html(inc, sim):
    f = FAULTS[inc["type"]]
    p = PIPE_BY_ID[inc["pipeline"]]
    label, color = STATUS_META[inc["status"]]
    age = (sim["clock"] - inc["marks"]["opened"]).total_seconds() / 60.0
    conf = f' · conf <b class="mono">{inc["confidence"]:.0%}</b>' if inc.get("confidence") else ""
    risk = f' · risk <b class="mono">{inc["pr"]["risk"]}</b>' if inc.get("pr") else ""
    return (f'<div style="margin-bottom:4px;"><b class="mono">{inc["id"]}</b> '
            f'{pill(f["icon"] + " " + f["label"], f["color"])}{pill(inc["severity"], SEV_COLOR[inc["severity"]])}'
            f'{pill(label, color)}<span class="mut">{conf}{risk}</span></div>'
            f'<div class="mut" style="font-size:.78rem;">{p["id"]} · {p["team"]} · tier {p["tier"]} · '
            f'opened {inc["marks"]["opened"].strftime("%H:%M")} UTC · age {age:.0f}m · '
            f'downtime {inc["downtime"]:.0f}m · ${inc["cost"]:.2f}</div>')

def pipe_card_html(sim, p):
    act = [i for i in sim["incidents"] if i["pipeline"] == p["id"]
           and i["status"] not in ("closed", "manual_closed")]
    unresolved = [i for i in act if i["status"] != "recovered"]
    if unresolved:
        cls, label = ("bad", "DOWN") if any(i["severity"] == "P1" for i in unresolved) else ("warn", "DEGRADED")
        fresh = f'{int(unresolved[0]["downtime"])}m stale'
    elif act:
        cls, label, fresh = "ok", "RECOVERING", "backfilling"
    else:
        cls, label = "ok", "HEALTHY"
        fresh = f'{(sum(ord(c) for c in p["id"]) % 9) + 3}m ago'
    return (f'<div class="pc"><b>{p["id"]}</b><br><span class="mut">{p["team"]} · T{p["tier"]} · '
            f'SLA {p["sla_min"]}m</span><br><span class="{cls}">● {label}</span> '
            f'<span class="mut">· {fresh}</span></div>')

def render_active_card(sim, inc):
    with st.container(border=True):
        st.markdown(inc_header_html(inc, sim), unsafe_allow_html=True)
        idx = FLOW_INDEX.get(inc["stage"], 0)
        frac = clamp(idx / 10.0, 0.02, 1.0)
        label = STAGE_LABEL.get(inc["stage"], inc["stage"])
        if inc["status"] in ("awaiting_approval", "held_kill", "held_budget"):
            label = f"⏸ {inc['gate_reason'] or 'awaiting approval'}"
        elif inc["status"] == "escalated":
            label = "🚨 Escalated to on-call — human resolution required"
        elif inc["status"] == "rolling_back":
            label = "↩️ Restoring pre-change snapshot"
        st.progress(frac, text=f"{label} · {frac:.0%}")
        c1, c2, c3, _ = st.columns([1, 1, 1, 1.6])
        if inc["status"] in ("awaiting_approval", "held_kill", "held_budget"):
            if c1.button("✅ Approve", key=f"ops_ap_{inc['id']}", type="primary"):
                mutate_approve(sim, inc)
                after_action(f"✅ Approved {inc['id']} — staging rehearsal starting")
            if c2.button("✖ Reject", key=f"ops_rj_{inc['id']}"):
                mutate_reject(sim, inc)
                after_action(f"✖ Rejected {inc['id']} — agent replanning")
        if rollbackable(sim, inc) and c3.button("↩️ Rollback", key=f"ops_rb_{inc['id']}"):
            start_rollback(sim, inc, get_rng(), "operator-initiated rollback", "human:you")
            after_action(f"↩️ Rolling back {inc['id']} to snapshot")
        with st.expander("🔍 Patch & recent trace"):
            if inc.get("pr"):
                st.code(inc["pr"]["diff"], language="diff")
                st.caption(f"Rollback plan: {inc['pr']['rollback_plan']}")
            st.markdown(events_html(inc["events"], limit=6), unsafe_allow_html=True)

def status_group(inc):
    s = inc["status"]
    if s in ("closed", "manual_closed"): return "Closed"
    if s == "recovered": return "Recovered"
    if s == "awaiting_approval": return "Awaiting approval"
    if s in ("held_kill", "held_budget"): return "Held"
    if s == "escalated": return "Escalated"
    return "Active"

def render_ops(sim):
    m = compute_metrics(sim)
    st.markdown('<div class="tip">🚀 <b>60-second demo</b> — ① approve a repair in the <b>Approval Inbox</b> '
                '② inject a fault from the sidebar ③ watch verify → auto-rollback → postmortem. '
                'Full source + one-click clipboard live in <b>📤 Share & Export</b>.</div>',
                unsafe_allow_html=True)
    cols = st.columns(6)
    cols[0].metric("Active incidents", m["active"], f"{m['awaiting']} awaiting approval", delta_color="off")
    cols[1].metric("MTTR", f"{m['mttr']:.1f}m", f"{m['mttr'] - MANUAL_BASELINE_MIN:.0f}m vs manual",
                   delta_color="inverse")
    cols[2].metric("Correct diagnosis", f"{m['acc']:.1f}%", f"{m['acc'] - 90:+.1f} pts vs 90% floor")
    cols[3].metric("Rollback rate", f"{m['rb']:.1f}%", f"{m['rb'] - 10:+.1f} pts vs 10% ceiling",
                   delta_color="inverse")
    cols[4].metric("Data downtime", f"{m['down']:.2f}h", "48h window", delta_color="off")
    cols[5].metric("Agent spend today", f"${m['cost']:.2f}", f"cap ${sim['policy']['budget']:.0f}/day",
                   delta_color="off")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("**📈 Recovery trend**")
        st_fig(fig_trend(sim), "fig_trend")
    with c2:
        st.markdown("**🧬 Incidents by cause (48h)**")
        st_fig(fig_donut(sim), "fig_donut")
    c3, c4 = st.columns([1, 1.2])
    with c3:
        st.markdown("**⏱ Data downtime by pipeline**")
        st_fig(fig_downtime(sim), "fig_down")
    with c4:
        st.markdown("**📡 Live agent feed**")
        with st.container(height=300):
            st.markdown(events_html(sim["feed"], show_inc=True, limit=14), unsafe_allow_html=True)
        st.caption(f"agent loop: plan → act → observe → reflect · spawned {sim['spawned']} · "
                   f"signals deduped {sim['suppressed']} · tokens {sim['tokens']:,}")
    st.markdown("### 🚨 Active incident board")
    sev_order = {"P1": 0, "P2": 1, "P3": 2}
    prio = {"awaiting_approval": 0, "held_kill": 0, "held_budget": 0, "escalated": 1,
            "rolling_back": 2, "active": 3, "recovered": 4}
    active = [i for i in sim["incidents"] if i["status"] not in ("closed", "manual_closed")]
    active.sort(key=lambda i: (prio.get(i["status"], 5), sev_order[i["severity"]], i["marks"]["opened"]))
    if not active:
        st.success("🌲 All pipelines healthy — agent is patrolling. Inject a fault or enable the chaos monkey.")
    for inc in active:
        render_active_card(sim, inc)
    st.markdown("### 🗺 Pipeline health")
    cols = st.columns(4)
    for idx, p in enumerate(PIPELINES):
        with cols[idx % 4]:
            st.markdown(pipe_card_html(sim, p), unsafe_allow_html=True)

def render_incidents(sim):
    c1, c2, c3, c4, c5 = st.columns([1.6, 1.5, 1.1, 1.4, 1.3])
    q = c1.text_input("Search", "", key="f_q", placeholder="id / pipeline / cause…")
    f_status = c2.multiselect("Status", ["Active", "Awaiting approval", "Held", "Escalated",
                                         "Recovered", "Closed"], key="f_status")
    f_sev = c3.multiselect("Severity", ["P1", "P2", "P3"], key="f_sev")
    f_pipe = c4.selectbox("Pipeline", ["All pipelines"] + [p["id"] for p in PIPELINES], key="f_pipe")
    f_sort = c5.selectbox("Sort", ["Newest first", "Oldest first", "Highest risk",
                                   "Lowest confidence", "Longest open"], key="f_sort")
    rows = []
    for i in sim["incidents"]:
        if q and q.lower() not in (i["id"] + i["pipeline"] + FAULTS[i["type"]]["label"]).lower():
            continue
        if f_status and status_group(i) not in f_status:
            continue
        if f_sev and i["severity"] not in f_sev:
            continue
        if f_pipe != "All pipelines" and i["pipeline"] != f_pipe:
            continue
        rows.append(i)
    age = lambda i: (sim["clock"] - i["marks"]["opened"]).total_seconds() / 60.0
    if f_sort == "Newest first": rows.sort(key=lambda i: i["marks"]["opened"], reverse=True)
    elif f_sort == "Oldest first": rows.sort(key=lambda i: i["marks"]["opened"])
    elif f_sort == "Highest risk": rows.sort(key=lambda i: (i.get("pr") or {}).get("risk", i.get("risk") or 0), reverse=True)
    elif f_sort == "Lowest confidence": rows.sort(key=lambda i: i.get("confidence") or 1.0)
    else: rows.sort(key=age, reverse=True)
    st.caption(f"{len(rows)} / {len(sim['incidents'])} incidents match")
    if rows:
        df = pd.DataFrame([{"ID": i["id"], "Pipeline": i["pipeline"],
                            "Type": FAULTS[i["type"]]["icon"] + " " + FAULTS[i["type"]]["label"],
                            "Sev": i["severity"], "Status": STATUS_META[i["status"]][0],
                            "Stage": STAGE_LABEL.get(i["stage"], i["stage"]),
                            "Conf %": round((i.get("confidence") or 0) * 100),
                            "Risk": (i.get("pr") or {}).get("risk", i.get("risk")),
                            "Age (m)": round(age(i)), "Downtime (m)": round(i["downtime"]),
                            "Cost $": round(i["cost"], 2)} for i in rows])
        st_frame(df, key="df_inc", height=280)
    else:
        st.info("No incidents match the current filters.")
    pool = rows or sim["incidents"]
    ids = [i["id"] for i in pool]
    if "inc_pick" in st.session_state and st.session_state.inc_pick not in ids:
        del st.session_state["inc_pick"]
    by_id = {i["id"]: i for i in sim["incidents"]}
    default = next((i["id"] for i in pool if i["status"] == "awaiting_approval"), ids[0])
    pick = st.selectbox("🔎 Inspect incident", ids, index=ids.index(default), key="inc_pick",
                        format_func=lambda x: f"{x} · {by_id[x]['pipeline']} · "
                                              f"{FAULTS[by_id[x]['type']]['label']} "
                                              f"({STATUS_META[by_id[x]['status']][0].lower()})")
    inc = by_id[pick]
    p = PIPE_BY_ID[inc["pipeline"]]
    st.markdown(inc_header_html(inc, sim), unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.3, 1, 1.3])
    with c1:
        st.markdown(f"**Table** `{p['table']}`  \n**Schedule** `{p['schedule']}`  \n"
                    f"**SLA** {p['sla_min']}m · **~rows/run** {p['rows_avg']:,}")
        st.markdown(f"**Diagnosis:** {inc.get('diagnosis_text') or '🔍 investigating…'}")
        if inc.get("gate_reason"):
            st.warning(f"⏸ {inc['gate_reason']}")
    with c2:
        st_fig(fig_gauge(inc.get("confidence"), sim["policy"]["conf_floor"]), "fig_gauge")
    with c3:
        st.caption("Lineage · red = failed node · amber = suspected upstream")
        st_dot(lineage_dot(inc), f"dot_{inc['id']}")
    b = st.columns(6)
    can_decide = inc["status"] in ("awaiting_approval", "held_kill", "held_budget", "escalated")
    if b[0].button("✅ Approve", key="det_ap", type="primary", disabled=not can_decide):
        mutate_approve(sim, inc); after_action(f"✅ Approved {inc['id']}")
    if b[1].button("✖ Reject", key="det_rj", disabled=not can_decide):
        mutate_reject(sim, inc); after_action(f"✖ Rejected {inc['id']} — replanning")
    if b[2].button("↩️ Rollback", key="det_rb", disabled=not rollbackable(sim, inc)):
        start_rollback(sim, inc, get_rng(), "operator-initiated rollback", "human:you")
        after_action(f"↩️ Rolling back {inc['id']}")
    if b[3].button("🚨 Escalate", key="det_esc",
                   disabled=inc["status"] in ("closed", "manual_closed", "escalated", "recovered")):
        mutate_escalate(sim, inc); after_action(f"🚨 {inc['id']} escalated to on-call")
    if b[4].button("🧑‍🔧 Human resolve", key="det_fix", disabled=inc["status"] != "escalated"):
        mutate_resolve(sim, inc); after_action(f"🧑‍🔧 {inc['id']} resolved manually — agent verified")
    with b[5]:
        fb = inc.get("feedback")
        st.caption(f"Diagnosis feedback: {'👍' if fb == 'up' else '👎' if fb == 'down' else '—'}")
        f1, f2 = st.columns(2)
        if f1.button("👍", key="det_up", disabled=not inc.get("diagnosed")):
            inc["feedback"] = "up"
            emit(sim, inc, get_rng(), "HUMAN", "reflect", "Operator confirmed the diagnosis (👍)")
            after_action("👍 Feedback recorded")
        if f2.button("👎", key="det_down", disabled=not inc.get("diagnosed")):
            inc["feedback"] = "down"
            emit(sim, inc, get_rng(), "HUMAN", "reflect", "Operator flagged the diagnosis as wrong (👎) — counts against accuracy")
            after_action("👎 Feedback recorded — accuracy metric updated")
    with st.expander("🧪 Simulation oracle (ground truth — hidden from the agent)"):
        match = inc.get("diagnosed") == inc["truth"] if inc.get("diagnosed") else None
        st.markdown(f"Truth: **{FAULTS[inc['truth']]['label']}** · Agent: "
                    f"**{FAULTS[inc['diagnosed']]['label'] if inc.get('diagnosed') else 'pending'}** · "
                    f"{'✅ match' if match else '❌ mismatch' if match is False else '…'}")
    if inc.get("pr"):
        with st.expander(f"🛠 Repair PR · {inc['pr']['branch']}", expanded=inc["status"] == "awaiting_approval"):
            st.markdown(f"**{inc['pr']['title']}** · risk {inc['pr']['risk']}/100 · "
                        f"+{inc['pr']['adds']}/−{inc['pr']['dels']} across {inc['pr']['files']} files")
            st.code(inc["pr"]["diff"], language="diff")
            st.caption(f"Rollback plan: {inc['pr']['rollback_plan']}")
    with st.expander("🧵 Full agent trace"):
        with st.container(height=320):
            st.markdown(events_html(inc["events"], limit=80), unsafe_allow_html=True)
    if inc["marks"].get("verified"):
        with st.expander("🧾 Recovery report & postmortem", expanded=inc["status"] == "closed"):
            st_frame(pd.DataFrame(checks_for(inc), columns=["Check", "Result", "Detail"]), key="df_checks")
            st.markdown(inc.get("report") or build_report_md(inc))
            d1, d2 = st.columns(2)
            with d1:
                st_download("⬇️ Report (JSON)", json.dumps(inc, default=str, indent=2),
                            f"{inc['id']}_report.json", "det_json", "application/json")
            with d2:
                st_download("⬇️ Postmortem (Markdown)", inc.get("postmortem") or build_postmortem_md(inc),
                            f"{inc['id']}_postmortem.md", "det_md", "text/markdown")

def render_inbox(sim):
    pending = [i for i in sim["incidents"]
               if i["status"] in ("awaiting_approval", "held_kill", "held_budget")]
    if not pending:
        st.success("🎉 Inbox zero — the agent is operating within policy. "
                   "New approval requests appear here the moment a gate trips.")
        auto_n = sum(1 for a in sim["audit"] if a["action"] == "auto-approve")
        st.caption(f"Auto-approved within policy so far: {auto_n}")
    else:
        low = [i for i in pending if (i.get("pr") or {}).get("risk", 100) < 40
               and i["status"] == "awaiting_approval"]
        if low and st_btn(f"✅ Approve all low-risk plans ({len(low)})", key="in_bulk", kind="primary"):
            for i in low:
                mutate_approve(sim, i)
            after_action(f"✅ Approved {len(low)} low-risk plans")
        for inc in pending:
            p = PIPE_BY_ID[inc["pipeline"]]
            with st.container(border=True):
                st.markdown(inc_header_html(inc, sim), unsafe_allow_html=True)
                st.warning(f"⏸ {inc['gate_reason']}")
                st.markdown(f"**Why this fix:** {inc.get('diagnosis_text')}  \n"
                            f"Blast radius ≈ **{p['rows_avg']:,} rows** · snapshot-first · "
                            f"rollback ETA ~4 min · confidence **{inc['confidence']:.0%}**")
                if inc.get("pr"):
                    with st.container(height=200):
                        st.code(inc["pr"]["diff"], language="diff")
                c1, c2, _ = st.columns([1, 1, 2])
                if c1.button("✅ Approve plan", key=f"in_ap_{inc['id']}", type="primary"):
                    mutate_approve(sim, inc); after_action(f"✅ {inc['id']} approved — staging → prod")
                if c2.button("✖ Reject", key=f"in_rj_{inc['id']}"):
                    mutate_reject(sim, inc); after_action(f"✖ {inc['id']} rejected — agent replanning")
    st.markdown("#### Recent decisions")
    dec = [a for a in sim["audit"] if a["action"] in ("approve", "reject", "auto-approve")][-6:]
    if dec:
        st_frame(pd.DataFrame([{"Time": a["t"].strftime("%H:%M:%S"), "Actor": a["actor"],
                                "Decision": a["action"], "Incident": a["inc"], "Detail": a["detail"]}
                               for a in reversed(dec)]), key="df_decisions")
    else:
        st.caption("No decisions yet.")

def render_reports(sim):
    done = [i for i in sim["incidents"] if i["status"] in ("closed", "manual_closed", "recovered")
            and i["marks"].get("verified")]
    done.sort(key=lambda i: i["marks"]["verified"], reverse=True)
    if not done:
        st.info("No completed recoveries yet — approve a repair or let the agent auto-heal one.")
        return
    ids = [i["id"] for i in done]
    if "rep_pick" in st.session_state and st.session_state.rep_pick not in ids:
        del st.session_state["rep_pick"]
    by_id = {i["id"]: i for i in done}
    pick = st.selectbox("Recovery run", ids, key="rep_pick",
                        format_func=lambda x: f"{x} · {by_id[x]['pipeline']} · "
                                              f"{FAULTS[by_id[x]['type']]['label']} · "
                                              f"{by_id[x]['marks']['verified'].strftime('%H:%M')} UTC")
    inc = by_id[pick]
    marks = inc["marks"]
    tl = [{"Phase": lbl, "At": marks[k].strftime("%H:%M:%S") + " UTC"}
          for k, lbl in [("opened", "Detected"), ("classified", "Classified"), ("diagnosed", "Diagnosed"),
                         ("pr", "PR opened"), ("approved", "Approved"), ("staged", "Staging verified"),
                         ("promoted", "Promoted"), ("verified", "Recovered"), ("closed", "Postmortem")]
          if marks.get(k)]
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("**Timeline**")
        st_frame(pd.DataFrame(tl), key="df_rep_timeline")
        st.markdown("**Verification**")
        st_frame(pd.DataFrame(checks_for(inc), columns=["Check", "Result", "Detail"]), key="df_rep_checks")
        st_download("⬇️ Report (JSON)", json.dumps(inc, default=str, indent=2),
                    f"{inc['id']}_report.json", "rep_json", "application/json")
        st_download("⬇️ Postmortem (Markdown)", inc.get("postmortem") or build_postmortem_md(inc),
                    f"{inc['id']}_postmortem.md", "rep_md", "text/markdown")
        st_download("⬇️ All reports (JSON)", json.dumps(done, default=str, indent=2),
                    "pipemedic_recovery_reports.json", "rep_all", "application/json")
    with c2:
        with st.container(border=True):
            st.markdown(inc.get("postmortem") or build_postmortem_md(inc))

def render_policy(sim):
    p = sim["policy"]
    st.markdown(pill(p["mode"], "cyan") + pill(f"conf floor {p['conf_floor']:.0%}", "violet")
                + pill(f"risk cap {p['max_risk']}", "amber") + pill(f"budget ${p['budget']:.0f}/day", "green")
                + pill("KILL SWITCH ON" if p["kill"] else "writes enabled", "red" if p["kill"] else "blue"),
                unsafe_allow_html=True)
    st.progress(clamp(sim["day_cost"] / max(p["budget"], 0.01), 0.0, 1.0),
                text=f"Budget used: ${sim['day_cost']:.2f} / ${p['budget']:.0f} · {sim['tokens']:,} tokens lifetime")
    st.markdown("#### Autonomy matrix (action class × level)")
    matrix = pd.DataFrame([
        ["Read logs / lineage / DQ", "auto", "auto", "auto", "auto"],
        ["Open repair PR (no prod effect)", "suggest", "auto+notify", "auto+notify", "auto"],
        ["Apply to staging", "human", "human", "auto+notify", "auto*"],
        ["Promote to production", "human", "human", "T1: human / else auto*", "auto*"],
        ["Backfill > 1M rows", "human", "human", "human", "auto*"],
        ["Rollback to snapshot", "human", "auto+notify", "auto+notify", "auto+notify"],
    ], columns=["Action class"] + MODES)
    st_frame(matrix, key="df_matrix")
    st.caption("* still gated by confidence floor, risk cap, daily budget and the kill switch.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Tool registry")
        tools = pd.DataFrame([
            ["read_orchestrator_logs", "READ", 10, 2, "—"],
            ["fetch_dag_metadata", "READ", 10, 2, "—"],
            ["query_history.scan", "READ", 20, 2, "—"],
            ["dq_results.fetch", "READ", 10, 2, "—"],
            ["lineage.trace", "READ", 15, 2, "—"],
            ["git.create_branch", "WRITE", 15, 1, "delete_branch"],
            ["git.open_pr", "WRITE", 15, 1, "close_pr"],
            ["warehouse.apply_patch_staging", "WRITE", 60, 1, "drop_staging_clone"],
            ["warehouse.promote_patch", "WRITE", 120, 0, "revert_to_snapshot"],
            ["orchestrator.backfill", "WRITE", 300, 0, "cancel_and_purge"],
            ["slack.notify", "WRITE", 5, 2, "follow_up_msg"],
        ], columns=["Tool", "Access", "Timeout (s)", "Retries", "Undo / compensation"])
        st_frame(tools, key="df_tools", height=320)
    with c2:
        st.markdown("#### Risk → control → test")
        risks = pd.DataFrame([
            ["Data corruption by faulty fix", "Snapshot + staging rehearsal + 4-check verify + auto-rollback",
             "Chaos replay: bad patch → restore ≤ 5 min, 0 row loss"],
            ["Privilege escalation", "Per-task scoped creds; write allow-list; kill switch",
             "Red-team token on prod scope → deny + audit entry"],
            ["Silent wrong repair", "Semantic checksums + post-hoc audit + human feedback",
             "Golden replay of 25 wrong-fix cases → ≥ 95% caught"],
        ], columns=["Risk", "Control", "Acceptance test"])
        st_frame(risks, key="df_risks", height=320)
        with st.expander("🧠 Memory layers"):
            st.markdown("- **Working** — Redis scratchpad per incident, TTL 15 min, never persisted.\n"
                        "- **Episodic** — PostgreSQL trajectories (90d), replayed for offline evals.\n"
                        "- **Semantic** — curated runbooks/policies in a vector store; writes require "
                        "provenance + review (poisoning defence). Tool output is treated as untrusted data.")
    st.markdown("#### Audit log — who changed what, with which agent version")
    entries = sim["audit"][-14:]
    st_frame(pd.DataFrame([{"Time": a["t"].strftime("%H:%M:%S"), "Actor": a["actor"],
                            "Action": a["action"], "Incident": a["inc"],
                            "Detail": a["detail"], "Result": a["result"]}
                           for a in reversed(entries)]), key="df_audit", height=280)

def render_dashboard():
    sim = st.session_state.sim
    rng = st.session_state.rng
    now = time.time()
    last = st.session_state.get("_last_real", now)
    st.session_state["_last_real"] = now
    if sim["running"]:
        dt = clamp(now - last, 0.0, 3.0) * sim["policy"]["speed"]
        if dt > 0:
            advance(sim, rng, dt)
    st.markdown(hero_html(sim), unsafe_allow_html=True)
    if sim["policy"]["kill"]:
        st.markdown('<div class="killbar">🛑 KILL SWITCH ENGAGED — all write actions frozen. '
                    'Diagnostics continue. Release it in the sidebar to resume repairs.</div>',
                    unsafe_allow_html=True)
    m = compute_metrics(sim)
    t1, t2, t3, t4, t5 = st.tabs(["📡 Live Ops", "🚨 Incidents",
                                  f"✅ Approval Inbox ({m['awaiting']})",
                                  "📜 Recovery Reports", "🛡️ Policy & Audit"])
    with t1: render_ops(sim)
    with t2: render_incidents(sim)
    with t3: render_inbox(sim)
    with t4: render_reports(sim)
    with t5: render_policy(sim)

# ---------------------------------------------------------------- share & export
@st.cache_data(show_spinner=False)
def _read_source(path_str, mtime):
    return Path(path_str).read_text(encoding="utf-8")

def get_app_source():
    try:
        p = Path(__file__).resolve()
        return _read_source(str(p), p.stat().st_mtime)
    except Exception:
        return "# Source read unavailable — copy app.py from your project folder."

def copy_button(label, payload):
    pay = json.dumps(payload).replace("<", "\\u003c")
    doc = f"""
<style>html,body{{margin:0;padding:0;background:transparent;font-family:'JetBrains Mono',monospace;}}
button{{width:100%;padding:10px 12px;border-radius:10px;border:1px solid rgba(79,70,229,.45);
background:linear-gradient(90deg,rgba(79,70,229,.10),rgba(124,58,237,.12));color:#0F172A;
font-weight:700;font-size:13px;cursor:pointer;transition:all .15s;}}
button:hover{{border-color:#4F46E5;box-shadow:0 0 14px rgba(79,70,229,.30);transform:translateY(-1px);}}</style>
<button id="cp">📋 {label}</button>
<script>
const PAYLOAD = {pay};
const btn = document.getElementById('cp');
function ok(){{btn.textContent='✅ Copied to clipboard';setTimeout(function(){{btn.textContent='📋 {label}';}},2200);}}
function fallback(){{
  const ta=document.createElement('textarea');ta.value=PAYLOAD;
  ta.style.cssText='position:fixed;left:0;top:0;opacity:0;';
  document.body.appendChild(ta);ta.focus();ta.select();
  let done=false;try{{done=document.execCommand('copy');}}catch(e){{}}
  ta.remove();
  if(done){{ok();}}else{{btn.textContent='⚠️ Auto-copy blocked — use the copy icon on the code block';}}
}}
btn.addEventListener('click',function(){{
  if(navigator.clipboard&&navigator.clipboard.writeText){{navigator.clipboard.writeText(PAYLOAD).then(ok).catch(fallback);}}
  else{{fallback();}}
}});
</script>"""
    components.html(doc, height=54)

def share_summary(sim):
    m = compute_metrics(sim)
    p = sim["policy"]
    return (f"⛑️ {APP_NAME} — live demo snapshot ({sim['clock'].strftime('%Y-%m-%d %H:%M')} UTC)\n"
            f"• Active incidents: {m['active']} ({m['awaiting']} awaiting approval)\n"
            f"• MTTR: {m['mttr']:.1f} min (manual baseline 252 min)\n"
            f"• Correct-diagnosis rate: {m['acc']:.1f}%\n"
            f"• Rollback rate: {m['rb']:.1f}%\n"
            f"• Data downtime (48h): {m['down']:.2f} h\n"
            f"• Agent spend today: ${m['cost']:.2f}\n"
            f"Autonomy: {p['mode']} · conf floor {p['conf_floor']:.0%} · kill switch {'ON' if p['kill'] else 'off'}\n"
            f"Simulated demo · single-file Streamlit · no API keys")

def render_share(sim):
    st.markdown("### 📤 Share & Export")
    st.info("Custom theme hiding Streamlit's ⋮ menu? No problem — copy or download everything right here. "
            "The app.py preview is read from this file's own source at runtime, so it is always in sync.")
    app_src = get_app_source()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📄 requirements.txt")
        st.code(REQUIREMENTS_TEXT, language="text")
        copy_button("Copy requirements.txt", REQUIREMENTS_TEXT)
        st_download("⬇️ Download requirements.txt", REQUIREMENTS_TEXT, "requirements.txt", "sh_req")
    with c2:
        st.markdown("#### 🐍 app.py  <span class='mut' style='font-size:.75rem;'>(complete single file)</span>",
                    unsafe_allow_html=True)
        with st.container(height=300):
            st.code(app_src, language="python")
        copy_button("Copy app.py (full source)", app_src)
        st_download("⬇️ Download app.py", app_src, "app.py", "sh_app", "text/x-python")
    st.divider()
    st.markdown("#### 📊 Share a live snapshot")
    summary = share_summary(sim)
    st.text_area("Snapshot text (editable before copying)", summary, height=170, key="sh_summary")
    copy_button("Copy snapshot", st.session_state.get("sh_summary", summary))
    st_download("⬇️ Full simulation state (JSON)", json.dumps(sim, default=str, indent=2),
                "pipemedic_simulation.json", "sh_json", "application/json")
    st.caption("Snapshot reflects the last full page interaction; the live console keeps simulating in the background.")
    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### ▶️ Run locally")
        run_cmd = "pip install -r requirements.txt\nstreamlit run app.py"
        st.code(run_cmd, language="bash")
        copy_button("Copy run commands", run_cmd)
    with r2:
        st.markdown("#### 🔗 Deploy & share a URL")
        st.markdown("1. Push `app.py` + `requirements.txt` to a GitHub repo.\n"
                    "2. On **Streamlit Community Cloud** (share.streamlit.io) → *New app* → pick the repo "
                    "and `app.py`.\n"
                    "3. Deploy — the public URL is your share link; the demo starts live for every visitor.\n\n"
                    "No secrets, no API keys, nothing to configure.")

# ---------------------------------------------------------------- mount
tab_live, tab_share = st.tabs(["🛰️ Live Console", "📤 Share & Export"])
with tab_live:
    if hasattr(st, "fragment"):
        interval = 0.9 if st.session_state.sim["running"] else None
        st.fragment(run_every=interval)(render_dashboard)()
    else:
        render_dashboard()
with tab_share:
    render_share(st.session_state.sim)
if not hasattr(st, "fragment") and st.session_state.sim["running"]:
    time.sleep(0.9)
    st.rerun()