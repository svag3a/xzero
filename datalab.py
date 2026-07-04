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
    apply_rule, calculate_elir,
)

router = APIRouter()

DATA_DIR    = Path(os.environ.get("DATA_DIR", Path(__file__).parent))
DB_PATH     = DATA_DIR / "scans.db"
LAB_DIR     = DATA_DIR / "datalab"
LAB_DIR.mkdir(exist_ok=True)

_claude = anthropic.Anthropic()
_MODEL  = "claude-opus-4-8"


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
        resp = _claude.messages.create(
            model=_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        return result.get("generic_labels", GENERIC_LABELS), result.get("mapping", {})
    except Exception as e:
        logging.warning(f"[datalab] claude mapping failed: {e}")
        heuristic = suggest_mapping_heuristic(columns)
        return GENERIC_LABELS, heuristic


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
        resp = _claude.messages.create(
            model=_MODEL,
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
1. Vad bevisade Data Lab?
2. Vad är den uppskattade I-faktorn och vad den innebär ekonomiskt?
3. Rekommendation: fortsätt till implementation / behöver mer data / potentialen för liten?
Direkt, konkret ton. Inga rubriker."""

    try:
        resp = _claude.messages.create(
            model=_MODEL,
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
        "SELECT id, scan_id, hypothesis, step, status, created_at, updated_at FROM datalab_sessions ORDER BY created_at DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


class CreateSession(BaseModel):
    scan_id:        int | None = None
    hypothesis:     str = ""
    hypothesis_json: str = ""   # full hypothesis object JSON from workshop


@router.get("/api/scans/{scan_id}/hypotheses")
async def get_scan_hypotheses(scan_id: int):
    con = _db()
    scan_row = con.execute(
        "SELECT workshop_hypotheses FROM scans WHERE id=?", (scan_id,)
    ).fetchone()
    if not scan_row:
        con.close()
        raise HTTPException(404, "Scan hittades inte")

    # Prefer live workshop session (has user-validated hypotheses)
    ws_row = con.execute(
        "SELECT session_json FROM workshop_sessions WHERE scan_id=? ORDER BY created_at DESC LIMIT 1",
        (scan_id,)
    ).fetchone()
    con.close()

    hypotheses = []
    if ws_row:
        try:
            hypotheses = json.loads(ws_row["session_json"]).get("hypotheses", [])
        except Exception:
            pass
    if not hypotheses:
        raw = scan_row["workshop_hypotheses"]
        if raw:
            try:
                hypotheses = json.loads(raw)
            except Exception:
                pass

    return [
        {
            "id":                h.get("hypothesis_id", ""),
            "title":             h.get("title", ""),
            "data_requirements": h.get("data_requirements", []),
            "prediction_target": (h.get("model_archetype") or {}).get("primary_prediction_target", ""),
            "use_case":          (h.get("candidate_use_case") or {}).get("name", ""),
        }
        for h in hypotheses if isinstance(h, dict)
    ]


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

    sim = {
        "rule_key":   rule_key,
        "rule_label": SIMULATION_RULES[rule_key],
        "dates":      model_data["dates"],
        "actual":     [round(float(v), 3) for v in actual],
        "predicted":  [round(float(v), 3) for v in predicted],
        "simulated":  [round(float(v), 3) for v in simulated],
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
