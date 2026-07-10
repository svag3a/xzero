import os
import json
import uuid
import sqlite3
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from datalab_engine import (
    GENERIC_COLUMNS, GENERIC_LABELS, SIMULATION_RULES,
    suggest_mapping_heuristic, parse_file, compute_dataset_meta,
    run_assessment, engineer_features, get_adapters, run_replay,
    apply_rule, calculate_elir, run_forecast,
)

router = APIRouter()

DATA_DIR    = Path(os.environ.get("DATA_DIR", Path(__file__).parent))
DB_PATH     = DATA_DIR / "scans.db"
LAB_DIR     = DATA_DIR / "datalab"
LAB_DIR.mkdir(exist_ok=True)

_MODEL_DIRECT  = "claude-opus-4-8"
_MODEL_BEDROCK = "us.anthropic.claude-sonnet-4-6"

def _get_claude():
    """Return (client, model_id) — Bedrock if AWS creds available, else direct."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        try:
            client = anthropic.AnthropicBedrock(
                aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                aws_region=os.environ.get("AWS_DEFAULT_REGION", "eu-west-1"),
            )
            return client, _MODEL_BEDROCK
        except Exception:
            pass
    return anthropic.Anthropic(), _MODEL_DIRECT


def _db():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def _now():
    return datetime.now(timezone.utc).isoformat()


def _session_path(sid: str) -> Path:
    p = LAB_DIR / sid
    p.mkdir(exist_ok=True)
    return p


def _get_session(sid: str) -> dict:
    con = _db()
    row = con.execute("SELECT * FROM datalab_sessions WHERE id=?", (sid,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(404, "Session hittades inte")
    d = dict(row)
    for f in ("dataset_meta","generic_labels","hypothesis_json","mapping","assessment","benchmark","replay","simulation","elir"):
        if d.get(f):
            try:
                d[f] = json.loads(d[f])
            except Exception:
                pass
    return d


def _update_session(sid: str, **fields):
    con = _db()
    fields["updated_at"] = _now()
    for k, v in fields.items():
        if isinstance(v, (dict, list)):
            fields[k] = json.dumps(v, ensure_ascii=False)
    set_clause = ", ".join(f"{k}=?" for k in fields)
    con.execute(f"UPDATE datalab_sessions SET {set_clause} WHERE id=?",
                list(fields.values()) + [sid])
    con.commit()
    con.close()


# ── Claude helpers ────────────────────────────────────────────────────────────

async def _claude_suggest_mapping(
    columns: list[str], sample_rows: list,
    hypothesis: str = "", company_name: str = "",
    data_requirements: list = None, prediction_target: str = "",
) -> tuple[dict, dict]:
    """Returns (generic_labels, mapping)."""
    data_req_section = ""
    if data_requirements:
        data_req_section = f"""
Datakrav identifierade i workshop-analysen:
{json.dumps(data_requirements, ensure_ascii=False, indent=2)}
Prediktionsmål: {prediction_target or "ej angivet"}

Använd datakraven som utgångspunkt för den generiska datamodellen.
"""

    prompt = f"""Du är ett datamappningsverktyg för xZero Opportunity Scan.

Kund: {company_name or "okänd"}
Hypotes: {hypothesis or "ej angiven"}
{data_req_section}
Kundens uppladdade kolumner: {json.dumps(columns)}
Exempelrader (max 3): {json.dumps(sample_rows[:3])}

Din uppgift:
1. Identifiera verksamhetstypen och vad som ska predikteras.
2. Definiera en generisk datamodell med 5–12 kolumner anpassad till JUST DENNA verksamhet.
   Använd snake_case-nycklar (t.ex. "date", "location_id", "glass_kg").
   Etiketterna ska vara på svenska och beskrivande.
3. Mappa kundens kolumner mot din generiska modell (null om ingen passande kolumn finns).

Returnera ENBART detta JSON-objekt, inget annat:
{{
  "generic_labels": {{"nyckel": "Etikett på svenska", ...}},
  "mapping": {{"nyckel": "kundens_kolumn_eller_null", ...}}
}}"""

    try:
        client, model = _get_claude()
        logging.info(f"[datalab] calling model={model}")
        resp = client.messages.create(
            model=model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        labels = result.get("generic_labels", {})
        if not labels:
            logging.warning("[datalab] claude returned empty generic_labels")
        return labels or GENERIC_LABELS, result.get("mapping", {})
    except Exception as e:
        logging.error(f"[datalab] claude mapping failed ({type(e).__name__}): {e}")
        heuristic = suggest_mapping_heuristic(columns)
        return {"_error": str(e), **GENERIC_LABELS}, heuristic


async def _claude_interpret_assessment(assessment: dict, mapping: dict) -> str:
    prompt = f"""Du är en dataanalytiker för xZero Data Lab.

Bedömning av dataset:
{json.dumps(assessment, ensure_ascii=False, indent=2)}

Datamappning:
{json.dumps({k: v for k, v in mapping.items() if v}, ensure_ascii=False)}

Skriv en kort (3-5 meningar), konkret tolkning på svenska:
- Vad är datasetets styrkor?
- Vilka risker finns för modellkvaliteten?
- Är datasetet tillräckligt för att träna en prediktiv modell?
Inga rubriker, bara löpande text."""

    try:
        client, model = _get_claude()
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logging.warning(f"[datalab] claude assessment failed: {e}")
        return "Kunde inte generera tolkning."


async def _claude_elir_narrative(elir: dict, scan_info: dict, target_col: str, rule: str) -> str:
    prompt = f"""Du är en analytiker på xZero som skriver beslutsunderlag.

Data Lab-resultat:
- Mål: prediktera '{target_col}'
- Simuleringsregel: {rule}
- I-faktor: {elir['i_pct']}%
- Noggrannhetsförbättring vs baseline: {elir['accuracy_gain']}%
- Konfidens: {elir['confidence']}
- MAE baseline: {elir['mae_baseline']}, MAE modell: {elir['mae_model']}

Opportunity Scan: {scan_info.get('company_name','Okänt bolag')}
Hypotes: {scan_info.get('hypothesis','')}

Skriv ett beslutsunderlag på 4-6 meningar på svenska:
1. Vad bevisade Data Lab för denna specifika hypotes?
2. Vad är den uppskattade I-faktorn för denna hypotes och vad den innebär ekonomiskt?
3. Betona att I-faktorn på {elir['i_pct']}% gäller enbart denna hypotes — bolagets totala I-faktor (summan av alla validerade hypoteser) kan bli betydligt högre när övriga hypoteser också testas och valideras i Data Lab.
4. Rekommendation: fortsätt till implementation / behöver mer data / potentialen för liten?
Direkt, konkret ton. Inga rubriker."""

    try:
        client, model = _get_claude()
        resp = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logging.warning(f"[datalab] claude elir narrative failed: {e}")
        return "Kunde inte generera beslutsunderlag."


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/datalab/page", response_class=HTMLResponse)
async def datalab_page():
    from fastapi.responses import HTMLResponse as _HR
    content = (Path(__file__).parent / "datalab.html").read_text(encoding="utf-8")
    return _HR(content, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@router.get("/api/datalab")
async def list_sessions():
    con = _db()
    rows = con.execute(
        """SELECT d.id, d.scan_id, d.hypothesis, d.step, d.status, d.created_at, d.updated_at,
                  s.company_name
           FROM datalab_sessions d
           LEFT JOIN scans s ON s.id = d.scan_id
           ORDER BY d.created_at DESC"""
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


class CreateSession(BaseModel):
    scan_id:        int | None = None
    hypothesis:     str = ""
    hypothesis_json: str = ""   # full hypothesis object JSON from workshop


@router.post("/api/datalab", status_code=201)
async def create_session(req: CreateSession):
    sid = str(uuid.uuid4())[:12].upper()
    now = _now()
    con = _db()
    company = ""
    if req.scan_id:
        row = con.execute("SELECT company_name FROM scans WHERE id=?", (req.scan_id,)).fetchone()
        if row:
            company = row[0] or ""
    con.execute(
        """INSERT INTO datalab_sessions
           (id, scan_id, hypothesis, hypothesis_json, step, status, company_name, created_at, updated_at)
           VALUES (?,?,?,?,1,'active',?,?,?)""",
        (sid, req.scan_id, req.hypothesis, req.hypothesis_json or None, company, now, now)
    )
    con.commit()
    con.close()
    return {"id": sid}


@router.get("/api/datalab/{sid}")
async def get_session(sid: str):
    return _get_session(sid)


@router.post("/api/datalab/{sid}/upload")
async def upload_dataset(sid: str, files: List[UploadFile] = File(...)):
    _get_session(sid)  # verify exists
    import pandas as pd

    dfs = []
    for f in files:
        content = await f.read()
        try:
            dfs.append(parse_file(content, f.filename))
        except Exception as e:
            raise HTTPException(400, f"Kunde inte läsa '{f.filename}': {e}")

    if not dfs:
        raise HTTPException(400, "Inga filer mottagna")

    try:
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    except Exception as e:
        raise HTTPException(400, f"Kunde inte slå ihop filer (kontrollera att kolumnerna stämmer): {e}")

    # save as CSV
    path = _session_path(sid) / "data.csv"
    df.to_csv(str(path), index=False)

    filename = files[0].filename if len(files) == 1 else f"{len(files)} filer"
    meta = compute_dataset_meta(df, filename)

    sess = _get_session(sid)
    hyp_obj = {}
    raw_hyp = sess.get("hypothesis_json")
    if raw_hyp:
        try:
            hyp_obj = json.loads(raw_hyp) if isinstance(raw_hyp, str) else raw_hyp
        except Exception:
            pass

    generic_labels, mapping = await _claude_suggest_mapping(
        meta["columns"], meta["sample_rows"],
        hypothesis=sess.get("hypothesis", ""),
        company_name=sess.get("company_name", ""),
        data_requirements=hyp_obj.get("data_requirements") or [],
        prediction_target=hyp_obj.get("prediction_target", ""),
    )

    _update_session(sid, step=2, dataset_meta=meta, generic_labels=generic_labels, mapping=mapping)
    return {"meta": meta, "mapping": mapping, "generic_labels": generic_labels}


class EnrichRequest(BaseModel):
    weather:  bool  = False
    lat:      float = 59.33
    lon:      float = 18.07
    location: str   = "Stockholm"
    calendar: bool  = False

@router.post("/api/datalab/{sid}/enrich")
async def enrich_dataset(sid: str, body: EnrichRequest):
    import pandas as pd
    import httpx
    from datalab_engine import enrich_with_external, find_date_col

    sess = _get_session(sid)
    path = _session_path(sid) / "data.csv"
    if not path.exists():
        raise HTTPException(400, "Dataset saknas — ladda upp filen först")

    df = pd.read_csv(str(path))
    date_col = find_date_col(df)
    if not date_col:
        raise HTTPException(400, "Ingen datumkolumn hittades i datasetet")

    weather_df = None
    if body.weather:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        start = dates.min().strftime("%Y-%m-%d")
        end   = dates.max().strftime("%Y-%m-%d")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={body.lat}&longitude={body.lon}"
            f"&start_date={start}&end_date={end}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
            f"&timezone=Europe%2FStockholm"
        )
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(502, f"Open-Meteo svarade {resp.status_code}")
        raw = resp.json().get("daily", {})
        weather_df = pd.DataFrame({
            "date":        raw.get("time", []),
            "temp_max":    raw.get("temperature_2m_max", []),
            "temp_min":    raw.get("temperature_2m_min", []),
            "precip_mm":   raw.get("precipitation_sum", []),
            "wind_max":    raw.get("wind_speed_10m_max", []),
        })
        weather_df["date"] = pd.to_datetime(weather_df["date"])

    df_enriched, added_cols = enrich_with_external(
        df, date_col, weather_df=weather_df, calendar=body.calendar
    )
    df_enriched.to_csv(str(path), index=False)

    # Rebuild meta with new columns
    from datalab_engine import compute_dataset_meta
    meta = compute_dataset_meta(df_enriched, sess.get("dataset_meta", {}).get("filename", "data.csv"))
    _update_session(sid, dataset_meta=meta)

    return {"added_columns": added_cols, "total_columns": len(df_enriched.columns)}


@router.post("/api/datalab/{sid}/mapping")
async def save_mapping(sid: str, body: dict):
    _get_session(sid)
    mapping = body.get("mapping", {})
    _update_session(sid, step=3, mapping=mapping)
    return {"ok": True}


@router.post("/api/datalab/{sid}/assess")
async def assess_dataset(sid: str):
    sess = _get_session(sid)
    path = _session_path(sid) / "data.csv"
    if not path.exists():
        raise HTTPException(400, "Dataset saknas")

    import pandas as pd
    df = pd.read_csv(str(path))
    mapping = sess.get("mapping") or {}

    assessment = run_assessment(df, mapping)
    interpretation = await _claude_interpret_assessment(assessment, mapping)
    assessment["interpretation"] = interpretation

    _update_session(sid, step=4, assessment=assessment)
    return assessment


@router.post("/api/datalab/{sid}/target")
async def set_target(sid: str, body: dict):
    _get_session(sid)
    target = body.get("target_col", "")
    if not target:
        raise HTTPException(400, "target_col saknas")
    _update_session(sid, step=5, target_col=target)
    return {"ok": True}


@router.post("/api/datalab/{sid}/train")
async def train_models(sid: str):
    sess     = _get_session(sid)
    target   = sess.get("target_col", "")
    mapping  = sess.get("mapping") or {}
    path     = _session_path(sid) / "data.csv"

    if not target:
        raise HTTPException(400, "target_col saknas")
    if not path.exists():
        raise HTTPException(400, "Dataset saknas")

    import pandas as pd
    df = pd.read_csv(str(path))

    def _run():
        X, y, dates = engineer_features(df, mapping, target)
        adapters    = get_adapters()
        return run_replay(X, y, dates, adapters)

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(500, f"Träning misslyckades: {e}")

    _update_session(sid, step=7, benchmark=result, replay=result)
    return result


@router.post("/api/datalab/{sid}/simulate")
async def simulate(sid: str, body: dict):
    sess     = _get_session(sid)
    replay   = sess.get("replay") or sess.get("benchmark")
    rule_key = body.get("rule", "exact")

    if not replay or "models" not in replay:
        raise HTTPException(400, "Kör träning först")
    if rule_key not in SIMULATION_RULES:
        raise HTTPException(400, "Ogiltig regel")

    best = replay.get("best_model")
    if not best or best not in replay["models"]:
        raise HTTPException(400, "Ingen modell att simulera")

    model_data = replay["models"][best]
    if "error" in model_data:
        raise HTTPException(400, model_data["error"])

    import numpy as np
    predicted  = np.array(model_data["predicted"])
    actual     = np.array(model_data["actual"])
    simulated  = apply_rule(predicted, rule_key)

    # Use full history if available, else fall back to test-only
    all_dates  = model_data.get("all_dates")  or model_data["dates"]
    all_actual = model_data.get("all_actual") or [round(float(v), 3) for v in actual]
    train_n    = len(all_dates) - len(model_data["dates"])
    sim_padded = [None] * train_n + [round(float(v), 3) for v in simulated]
    pred_padded = [None] * train_n + [round(float(v), 3) for v in predicted]

    sim = {
        "rule_key":   rule_key,
        "rule_label": SIMULATION_RULES[rule_key],
        "dates":      all_dates,
        "actual":     all_actual,
        "predicted":  pred_padded,
        "simulated":  sim_padded,
    }
    _update_session(sid, step=8, simulation=sim)
    return sim


@router.post("/api/datalab/{sid}/elir")
async def elir(sid: str):
    sess = _get_session(sid)
    sim  = sess.get("simulation")
    if not sim:
        raise HTTPException(400, "Kör simulering först")

    elir_result = calculate_elir(sim["actual"], sim["simulated"])

    # Claude narrative
    scan_info = {"company_name": sess.get("company_name",""), "hypothesis": sess.get("hypothesis","")}
    narrative = await _claude_elir_narrative(
        elir_result, scan_info,
        sess.get("target_col","?"), sim.get("rule_label","")
    )
    elir_result["narrative"] = narrative

    # Optionally save I-factor back to linked scan
    scan_id = sess.get("scan_id")
    if scan_id:
        try:
            con = _db()
            con.execute(
                "UPDATE scans SET i_pct=? WHERE id=?",
                (round(elir_result["i_pct"], 1), scan_id)
            )
            con.commit()
            con.close()
            elir_result["saved_to_scan"] = True
        except Exception:
            elir_result["saved_to_scan"] = False

    _update_session(sid, step=9, status="completed", elir=elir_result)
    return elir_result


@router.post("/api/datalab/{sid}/forecast")
async def forecast(sid: str, body: dict = None):
    import pandas as pd
    body = body or {}
    n_periods = max(1, min(365, int(body.get("n_periods", 30))))
    sess = _get_session(sid)
    mapping = sess.get("mapping") or {}
    target  = sess.get("target_col")
    if not target:
        raise HTTPException(400, "Välj målvariabel först (steg 5)")
    path = _session_path(sid) / "data.csv"
    if not path.exists():
        raise HTTPException(400, "Datafil saknas")
    df = pd.read_csv(str(path))
    result = run_forecast(df, mapping, target, n_periods)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.delete("/api/datalab/{sid}")
async def delete_session(sid: str):
    _get_session(sid)  # verify exists
    con = _db()
    con.execute("DELETE FROM datalab_sessions WHERE id=?", (sid,))
    con.commit()
    con.close()
    import shutil
    p = _session_path(sid)
    try:
        shutil.rmtree(str(p))
    except Exception:
        pass
    return {"ok": True}


@router.get("/api/datalab/{sid}/report")
async def get_report(sid: str):
    sess = _get_session(sid)
    return {
        "session":    {k: sess[k] for k in ("id","scan_id","hypothesis","company_name","created_at") if k in sess},
        "dataset":    sess.get("dataset_meta"),
        "mapping":    sess.get("mapping"),
        "assessment": sess.get("assessment"),
        "target":     sess.get("target_col"),
        "benchmark":  sess.get("benchmark"),
        "simulation": sess.get("simulation"),
        "elir":       sess.get("elir"),
    }


def _e(s):
    if not s:
        return ""
    return (str(s).replace("&","&amp;").replace("<","&lt;")
            .replace(">","&gt;").replace('"',"&quot;"))


def _fmt(v, d=1):
    """Format a numeric value to d decimals; return '–' for non-numeric."""
    if not isinstance(v, (int, float)):
        return str(v) if v else "–"
    return f"{v:.{d}f}"


def _simulation_svg(actual: list, simulated: list, max_pts: int = 300) -> str:
    """Generate an inline SVG line chart of actual vs simulated."""
    import math
    n = len(actual)
    if n == 0:
        return ""
    step = max(1, math.ceil(n / max_pts))
    idx  = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    act = [actual[i]    for i in idx]
    sim = [simulated[i] for i in idx]
    m   = len(idx)

    W, H, PL, PR, PT, PB = 800, 220, 60, 20, 20, 40
    cw, ch = W - PL - PR, H - PT - PB
    all_v  = act + sim
    lo, hi = min(all_v), max(all_v)
    rng    = hi - lo or 1

    def px(i): return PL + i / (m - 1) * cw if m > 1 else PL
    def py(v): return PT + ch - (v - lo) / rng * ch

    grid = ""
    for gi in range(5):
        yy  = PT + gi / 4 * ch
        val = hi - gi / 4 * rng
        grid += (f'<line x1="{PL}" y1="{yy:.1f}" x2="{PL+cw}" y2="{yy:.1f}" '
                 f'stroke="#e2e8f0" stroke-width="1"/>'
                 f'<text x="{PL-6}" y="{yy+4:.1f}" text-anchor="end" '
                 f'font-size="10" fill="#94a3b8">{val:.1f}</text>')

    # X labels every ~8 points
    xlabels = ""
    ls = max(1, m // 8)
    for i in range(0, m, ls):
        xlabels += (f'<text x="{px(i):.1f}" y="{H-6}" text-anchor="middle" '
                    f'font-size="9" fill="#94a3b8">{idx[i]}</text>')

    def polyline(vals, color, dash=""):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        da  = f'stroke-dasharray="{dash}"' if dash else ""
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="1.8" {da} stroke-linejoin="round"/>')

    note = f'<text x="{W-PR}" y="{PT-6}" text-anchor="end" font-size="9" fill="#94a3b8">Visar {m} av {n} punkter</text>' if n > max_pts else ""

    return f"""<svg viewBox="0 0 {W} {H+10}" style="width:100%;display:block">
  <rect x="{PL}" y="{PT-14}" width="10" height="3" fill="#94a3b8" rx="1"/>
  <text x="{PL+14}" y="{PT-9}" font-size="10" fill="#64748b">Faktiskt utfall</text>
  <rect x="{PL+110}" y="{PT-14}" width="10" height="3" fill="#2563eb" rx="1"/>
  <text x="{PL+124}" y="{PT-9}" font-size="10" fill="#64748b">Simulerat</text>
  {note}{grid}
  <line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PT+ch}" stroke="#e2e8f0"/>
  {polyline(act, "#94a3b8", "4 3")}
  {polyline(sim, "#2563eb")}
</svg>"""


def _datalab_report_html(sess: dict) -> str:
    from datetime import date as _date

    company   = _e(sess.get("company_name") or "Okänt bolag")
    hyp_title = _e(sess.get("hypothesis") or "")
    generated = _date.today().strftime("%d %B %Y").lstrip("0")

    meta   = sess.get("dataset_meta") or {}
    bench  = sess.get("benchmark")    or {}
    sim    = sess.get("simulation")   or {}
    elir   = sess.get("elir")         or {}
    hyp_j  = sess.get("hypothesis_json")
    if isinstance(hyp_j, str):
        try: hyp_j = json.loads(hyp_j)
        except Exception: hyp_j = {}
    hyp_j = hyp_j or {}
    target = _e(sess.get("target_col") or "")

    # ── ELIR numbers ──────────────────────────────────────────────────────────
    i_pct     = elir.get("i_pct", 0)
    acc_gain  = elir.get("accuracy_gain", 0)
    mae_base  = elir.get("mae_baseline", 0)
    mae_model = elir.get("mae_model",    0)
    n_samples = elir.get("n_samples",    0)
    conf      = _e(elir.get("confidence", ""))
    narrative = _e(elir.get("narrative", ""))
    vol_diff  = elir.get("volume_diff_pct", 0)

    # ── Benchmark table ───────────────────────────────────────────────────────
    models     = bench.get("models", {})
    best_model = bench.get("best_model", "")
    bench_rows = ""
    dash = "–"
    for model_name, m in (models.items() if isinstance(models, dict) else []):
        is_b     = "★ " if model_name == best_model else ""
        tr_class = ' class="best-row"' if is_b else ""
        err      = m.get("error")
        if err:
            bench_rows += f'<tr><td>{is_b}{_e(model_name)}</td><td colspan="5" style="color:#ef4444">{_e(str(err)[:80])}</td></tr>'
        else:
            me = m.get("metrics", {})
            bench_rows += (f'<tr{tr_class}>'
                           f'<td><strong>{is_b}</strong>{_e(model_name)}</td>'
                           f'<td>{_fmt(me.get("mae"))}</td>'
                           f'<td>{_fmt(me.get("rmse"))}</td>'
                           f'<td>{_fmt(me.get("r2"))}</td>'
                           f'<td>{_fmt(me.get("mape")) if me.get("mape") is not None else dash}</td>'
                           f'<td>{_fmt(me.get("bias"))}</td></tr>')

    # ── Simulation chart ──────────────────────────────────────────────────────
    chart_svg = ""
    if sim.get("actual") and sim.get("simulated"):
        chart_svg = _simulation_svg(sim["actual"], sim["simulated"])

    # ── Hypothesis details (from workshop JSON) ───────────────────────────────
    vs_label = {
        "confirmed":            "Bekräftad",
        "confirmed_adjusted":   "Bekräftad med justering",
        "partially_confirmed":  "Delvis bekräftad",
        "rejected":             "Avvisad",
        "not_discussed":        "Ej diskuterad",
        "not_validated":        "Ej validerad",
    }
    vs      = hyp_j.get("validation_status", "")
    vs_text = vs_label.get(vs, _e(vs))
    vs_color = {"confirmed":"#10b981","confirmed_adjusted":"#10b981",
                "partially_confirmed":"#f59e0b","rejected":"#ef4444"}.get(vs,"#64748b")

    conf_summary = _e(hyp_j.get("confirmation_summary",""))
    adj_notes    = _e(hyp_j.get("adjustment_notes",""))
    new_findings = hyp_j.get("new_findings") or []
    evidence     = hyp_j.get("evidence_collected") or []
    use_cases    = hyp_j.get("recommended_use_cases") or []
    quant_est    = hyp_j.get("quantification_estimates") or []

    # Hypothesis section HTML
    hyp_section = ""
    if conf_summary or vs:
        status_badge = f'<span style="background:{vs_color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600">{_e(vs_text)}</span>' if vs_text else ""
        findings_html = "".join(f"<li>{_e(f)}</li>" for f in new_findings) if new_findings else ""
        evidence_html = "".join(f"<li>{_e(e)}</li>" for e in evidence)     if evidence     else ""
        adj_html      = f'<div class="detail-block"><h4>Justering</h4><p>{adj_notes}</p></div>' if adj_notes else ""

        quant_rows = ""
        for q in quant_est:
            bv = q.get("base_value")
            quant_rows += (f'<tr><td>{_e(q.get("metric",""))}</td>'
                           f'<td>{_e(str(bv)) if bv is not None else "–"} {_e(q.get("unit",""))}</td>'
                           f'<td>{_e(q.get("confidence",""))}</td>'
                           f'<td>{_e(q.get("notes",""))}</td></tr>')

        hyp_section = f"""
        <section>
          <h2>Hypotesvalidering</h2>
          <div class="detail-block">
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem">
              <h3 style="margin:0">{hyp_title}</h3>{status_badge}
            </div>
            {f'<p>{conf_summary}</p>' if conf_summary else ''}
          </div>
          {adj_html}
          {f'<div class="detail-block"><h4>Nya insikter</h4><ul>{findings_html}</ul></div>' if findings_html else ''}
          {f'<div class="detail-block"><h4>Insamlade bevis</h4><ul>{evidence_html}</ul></div>' if evidence_html else ''}
          {f'''<div class="detail-block"><h4>Kvantifieringar</h4>
          <table class="data-table"><thead><tr><th>Mått</th><th>Värde</th><th>Konfidens</th><th>Notering</th></tr></thead>
          <tbody>{quant_rows}</tbody></table></div>''' if quant_rows else ''}
        </section>"""

    # Use cases section
    uc_html = ""
    for uc in use_cases:
        dr_rows = ""
        for dr in (uc.get("data_requirements") or []):
            avail_color = {"high":"#10b981","medium":"#f59e0b","low":"#ef4444"}.get(dr.get("availability",""),"#64748b")
            dr_rows += (f'<tr><td>{_e(dr.get("data_name",""))}</td>'
                        f'<td><span style="color:{avail_color};font-weight:600">{_e(dr.get("availability",""))}</span></td>'
                        f'<td>{_e(dr.get("owner",""))}</td>'
                        f'<td>{_e(dr.get("prep_effort",""))}</td>'
                        f'<td>{_e(dr.get("format_notes",""))}</td></tr>')
        dr_table = (f'<table class="data-table" style="margin-top:0.75rem"><thead>'
                    f'<tr><th>Dataset</th><th>Tillgänglighet</th><th>Ägare</th><th>Prep-insats</th><th>Format</th></tr></thead>'
                    f'<tbody>{dr_rows}</tbody></table>') if dr_rows else ""
        uc_html += f"""<div class="uc-card">
          <div class="uc-head">
            <span class="uc-module">{_e(uc.get("xzero_module",""))}</span>
            <strong>{_e(uc.get("name",""))}</strong>
            <span style="color:#64748b;font-size:0.85rem">{_e(uc.get("model_type",""))}</span>
          </div>
          <p>{_e(uc.get("description",""))}</p>
          {f'<p><strong>Förväntat värde:</strong> {_e(uc.get("expected_value",""))}</p>' if uc.get("expected_value") else ''}
          {f'<p><strong>Motivering:</strong> {_e(uc.get("motivation",""))}</p>' if uc.get("motivation") else ''}
          {dr_table}
        </div>"""

    uc_section = f"<section><h2>Rekommenderade use cases</h2>{uc_html}</section>" if uc_html else ""

    # Dataset stats
    rows_n  = meta.get("rows", "–")
    cols_n  = meta.get("cols", "–")
    dt_from = meta.get("date_from") or meta.get("date_min","–")
    dt_to   = meta.get("date_to")   or meta.get("date_max","–")
    train_r = bench.get("train_rows","–")
    test_r  = bench.get("test_rows","–")

    vol_str   = (_fmt(vol_diff) + " %") if isinstance(vol_diff, (int, float)) else "–"
    i_str     = (_fmt(i_pct)   + " %") if isinstance(i_pct,   (int, float)) else "–"
    acc_str   = (_fmt(acc_gain)+ " %") if isinstance(acc_gain,(int, float)) else "–"

    return f"""<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Lab – {company}</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --navy:#1a3a5c;--blue:#2563eb;--green:#10b981;--amber:#f59e0b;
  --red:#ef4444;--text:#1e293b;--muted:#64748b;--border:#e2e8f0;
  --bg:#f8fafc;--card:#fff;--accent-light:#e8eef4;
}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  color:var(--text);background:var(--bg);font-size:15px;line-height:1.6}}
.page{{max-width:920px;margin:0 auto;padding:2rem 1.5rem 4rem}}
header{{background:var(--navy);color:#fff;padding:2rem 2.5rem;border-radius:10px;margin-bottom:2rem}}
header .eyebrow{{font-size:0.72rem;letter-spacing:.1em;text-transform:uppercase;
  color:rgba(255,255,255,.5);margin-bottom:.35rem}}
header h1{{font-size:1.65rem;font-weight:700;line-height:1.2}}
header .sub{{margin-top:.4rem;color:rgba(255,255,255,.6);font-size:.88rem}}
section{{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:1.5rem 2rem;margin-bottom:1.5rem}}
section h2{{font-size:0.78rem;font-weight:700;color:var(--navy);text-transform:uppercase;
  letter-spacing:.1em;margin-bottom:1.2rem;padding-bottom:.55rem;border-bottom:2px solid var(--border)}}
section h3{{font-size:1rem;font-weight:600;color:var(--navy);margin-bottom:.5rem}}
section h4{{font-size:.82rem;font-weight:600;color:var(--muted);text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:.5rem}}
p{{margin-bottom:.65rem;color:var(--text);font-size:.95rem}}
ul{{padding-left:1.25rem;margin-bottom:.6rem}}
ul li{{margin-bottom:.25rem;font-size:.93rem}}
.elir-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.875rem;margin-bottom:1.25rem}}
.elir-card{{background:var(--accent-light);border-radius:6px;padding:.9rem 1rem;text-align:center}}
.elir-card .val{{font-size:1.8rem;font-weight:800;line-height:1;letter-spacing:-.02em}}
.elir-card .lbl{{font-size:.7rem;color:var(--muted);margin-top:.3rem;text-transform:uppercase;letter-spacing:.06em}}
.elir-card.accent .val{{color:var(--blue)}}
.elir-card.good   .val{{color:var(--green)}}
.narrative{{background:var(--accent-light);border-left:3px solid var(--navy);padding:.7rem 1rem;
  border-radius:0 4px 4px 0;font-size:.9rem;color:var(--text);white-space:pre-wrap;margin-top:1rem}}
.callout{{background:#f0f4f8;border-left:3px solid var(--navy);padding:.65rem 1rem;
  border-radius:0 4px 4px 0;margin-top:1.1rem}}
.callout strong{{display:block;font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;
  color:var(--navy);margin-bottom:.5rem}}
.callout table{{border-collapse:collapse;width:100%}}
.callout td{{padding:.25rem .4rem;font-size:.83rem;color:#3a3a3a;vertical-align:top;border:none}}
.callout td:first-child{{white-space:nowrap;font-weight:600;padding-right:.75rem;width:11rem}}
.stat-row{{display:flex;flex-wrap:wrap;gap:1.5rem;margin-bottom:1.1rem}}
.stat{{display:flex;flex-direction:column}}
.stat .v{{font-size:1.05rem;font-weight:700}}
.stat .k{{font-size:.75rem;color:var(--muted)}}
.data-table{{width:100%;border-collapse:collapse;font-size:.85rem}}
.data-table th{{background:var(--accent-light);text-align:left;padding:.5rem .75rem;
  font-size:.73rem;color:var(--navy);text-transform:uppercase;letter-spacing:.05em;
  border-bottom:2px solid var(--border)}}
.data-table td{{padding:.5rem .75rem;border-bottom:1px solid var(--border);vertical-align:top}}
.data-table .best-row td{{background:#eff6ff}}
.chart-wrap{{overflow-x:auto;margin-top:.75rem}}
.detail-block{{background:var(--accent-light);border-radius:6px;padding:.875rem 1rem;margin-bottom:.75rem}}
.uc-card{{border:1px solid var(--border);border-radius:6px;padding:1rem 1.25rem;margin-bottom:1rem}}
.uc-head{{display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem;flex-wrap:wrap}}
.uc-module{{background:var(--navy);color:#fff;font-size:.7rem;font-weight:700;
  padding:2px 8px;border-radius:20px;letter-spacing:.04em}}
.print-btn{{position:fixed;bottom:1.5rem;right:1.5rem;background:var(--navy);color:#fff;
  border:none;padding:.65rem 1.25rem;border-radius:6px;cursor:pointer;font-size:.88rem;
  font-weight:600;box-shadow:0 4px 14px rgba(26,58,92,.3)}}
@media print{{
  .print-btn{{display:none}}
  body{{background:#fff}}
  .page{{padding:0}}
  header{{border-radius:0}}
  section{{break-inside:avoid}}
}}
</style>
</head>
<body>
<div class="page">

<header>
  <div class="eyebrow">xZero Data Lab &mdash; Analysrapport</div>
  <h1>{company}</h1>
  <div class="sub">{hyp_title}{"&ensp;&middot;&ensp;" + generated if generated else ""}</div>
</header>

<section>
  <h2>ELIR-resultat</h2>
  <div class="elir-grid">
    <div class="elir-card good"><div class="val">{i_str}</div><div class="lbl">I-faktor</div></div>
    <div class="elir-card accent"><div class="val">{acc_str}</div><div class="lbl">Noggrannhetsvinst</div></div>
    <div class="elir-card"><div class="val">{_fmt(mae_base)}</div><div class="lbl">MAE Baseline</div></div>
    <div class="elir-card"><div class="val">{_fmt(mae_model)}</div><div class="lbl">MAE Modell (sim)</div></div>
    <div class="elir-card"><div class="val">{vol_str}</div><div class="lbl">Volymavvikelse</div></div>
    <div class="elir-card"><div class="val">{n_samples if isinstance(n_samples, int) else "–"}</div><div class="lbl">Testpunkter (n)</div></div>
  </div>
  {f'<div class="narrative">{narrative}</div>' if narrative else ''}
  <div class="callout">
    <strong>Om måtten</strong>
    <table><tbody>
      <tr><td>I-faktor</td><td>Andel av förbättringspotentialen som modellen realiserar. Beräknas som noggrannhetsvinst × R-faktor och är det centrala måttet på affärsvärde.</td></tr>
      <tr><td>Noggrannhetsvinst</td><td>Relativ förbättring i prediktionsnoggrannhet jämfört med en naiv basmodell (medelvärdesförutsägelse): (MAE<sub>baseline</sub> − MAE<sub>modell</sub>) / MAE<sub>baseline</sub>.</td></tr>
      <tr><td>MAE baseline</td><td>Mean Absolute Error för en naiv modell som alltid förutsäger medelvärdet. Referenspunkt för hur bra man gör utan AI.</td></tr>
      <tr><td>MAE modell (sim)</td><td>Mean Absolute Error för den bästa tränade modellen på testdata. Lägre är bättre; jämför alltid mot baseline.</td></tr>
      <tr><td>Volymavvikelse</td><td>Procentuell skillnad i total volym mellan simulerat och faktiskt utfall. Visar om modellen systematiskt över- eller underestimerar.</td></tr>
      <tr><td>Testpunkter (n)</td><td>Antal datapunkter i testmängden (30 % av dataset). Fler testpunkter ger mer tillförlitliga mätvärden.</td></tr>
    </tbody></table>
  </div>
</section>

{hyp_section}

<section>
  <h2>Dataset &amp; modellering</h2>
  <div class="stat-row">
    <div class="stat"><span class="v">{f"{rows_n:,}" if isinstance(rows_n, int) else rows_n}</span><span class="k">Rader</span></div>
    <div class="stat"><span class="v">{cols_n}</span><span class="k">Kolumner</span></div>
    <div class="stat"><span class="v">{_e(str(dt_from)[:10])}</span><span class="k">Från</span></div>
    <div class="stat"><span class="v">{_e(str(dt_to)[:10])}</span><span class="k">Till</span></div>
    <div class="stat"><span class="v">{_e(target)}</span><span class="k">Målvariabel</span></div>
    <div class="stat"><span class="v">{train_r}</span><span class="k">Träningsrader</span></div>
    <div class="stat"><span class="v">{test_r}</span><span class="k">Testrader</span></div>
  </div>
  <table class="data-table">
    <thead><tr><th>Modell</th><th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE %</th><th>Bias</th></tr></thead>
    <tbody>{bench_rows}</tbody>
  </table>
</section>

{"<section><h2>Simulerat utfall vs verklighet</h2><div class='chart-wrap'>" + chart_svg + "</div></section>" if chart_svg else ""}

{uc_section}

</div>
<button class="print-btn" onclick="window.print()">Skriv ut / Spara PDF</button>
</body>
</html>"""


@router.get("/api/datalab/{sid}/report.html", response_class=HTMLResponse)
async def get_report_html(sid: str):
    sess = _get_session(sid)
    if not sess.get("elir"):
        raise HTTPException(400, "ELIR-beräkning saknas — slutför steg 9 först")
    try:
        html = _datalab_report_html(sess)
        return HTMLResponse(html)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return HTMLResponse(f"<pre style='color:red;padding:2rem'>{tb}</pre>", status_code=500)
