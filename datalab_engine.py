import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Generic data model ────────────────────────────────────────────────────────

GENERIC_COLUMNS = [
    "date", "product_id", "customer_id",
    "qty_sold", "qty_produced", "qty_forecast",
    "qty_stock", "qty_waste",
    "price", "cost", "revenue",
]

GENERIC_LABELS = {
    "date":         "Datum",
    "product_id":   "Produkt / SKU",
    "customer_id":  "Kund",
    "qty_sold":     "Såld mängd",
    "qty_produced": "Producerad mängd",
    "qty_forecast": "Forecast",
    "qty_stock":    "Lager",
    "qty_waste":    "Svinn",
    "price":        "Pris",
    "cost":         "Kostnad",
    "revenue":      "Omsättning",
}

_HINTS = {
    "date":         ["date","datum","dag","day","week","vecka","month","månad","period","time","tid"],
    "product_id":   ["product","produkt","sku","item","artikel","vara","kategori","category","name","namn"],
    "customer_id":  ["customer","kund","client","butik","store","buyer"],
    "qty_sold":     ["sold","såld","försäljning","sales","sell","sälj","demand","efterfrågan","quantity","antal","qty","vol"],
    "qty_produced": ["produc","tillverkad","manufactured","output","kapacitet","tillverkning"],
    "qty_forecast": ["forecast","prognos","predicted","plan","budget","planned"],
    "qty_stock":    ["stock","lager","inventory","balance","saldo"],
    "qty_waste":    ["waste","svinn","scrap","kassation","förlust","loss","spill","defect"],
    "price":        ["price","pris","rate","tariff","unit_price","listpris"],
    "cost":         ["cost","kostnad","expense","utgift","cogs","unit_cost"],
    "revenue":      ["revenue","omsättning","income","intäkt","turnover","sales_value","försäljningsvärde"],
}


def suggest_mapping_heuristic(columns: list[str]) -> dict:
    mapping = {g: None for g in GENERIC_COLUMNS}
    used = set()
    for generic, hints in _HINTS.items():
        for hint in hints:
            for col in columns:
                if col in used:
                    continue
                if hint in col.lower() and mapping[generic] is None:
                    mapping[generic] = col
                    used.add(col)
                    break
            if mapping[generic]:
                break
    return mapping


# ── Dataset parsing ───────────────────────────────────────────────────────────

def parse_file(content: bytes, filename: str) -> pd.DataFrame:
    import io
    if filename.lower().endswith(".csv"):
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, low_memory=False)
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
        return pd.read_csv(io.BytesIO(content), low_memory=False)
    else:
        return pd.read_excel(io.BytesIO(content))


def find_date_col(df: pd.DataFrame) -> str | None:
    """Return the most likely date column name, or None."""
    _DATE_HINTS = ["date","datum","dag","day","week","vecka","month","månad",
                   "period","time","tid","year","år","quarter","kvartal"]
    for col in df.columns:
        if any(h in col.lower() for h in _DATE_HINTS):
            try:
                if pd.to_datetime(df[col], errors="coerce").notna().mean() > 0.8:
                    return col
            except Exception:
                pass
    for col in df.columns:
        if df[col].dtype.kind in ("i", "u", "f"):
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8 and parsed.notna().sum() > 10:
                yr = parsed.dropna().dt.year
                if yr.min() >= 1990 and yr.max() <= 2035:
                    return col
        except Exception:
            pass
    return None


def _easter(year: int):
    """Return Easter Sunday as date."""
    from datetime import date as _date
    a = year % 19; b = year // 100; c = year % 100
    d = b // 4;    e = b % 4;      f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4;    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day   = ((h + l - 7*m + 114) % 31) + 1
    return _date(year, month, day)


def _swedish_holidays(year: int) -> set:
    from datetime import date as _date, timedelta
    e = _easter(year)
    holidays = {
        _date(year, 1, 1),   # Nyårsdagen
        _date(year, 1, 6),   # Trettondedag jul
        e - timedelta(2),    # Långfredag
        e,                   # Påskdagen
        e + timedelta(1),    # Annandag påsk
        _date(year, 5, 1),   # Första maj
        e + timedelta(39),   # Kristi himmelsfärdsdag
        e + timedelta(49),   # Pingstdagen
        _date(year, 6, 6),   # Nationaldagen
        _date(year, 12, 25), # Juldagen
        _date(year, 12, 26), # Annandag jul
    }
    # Midsommardagen: first Saturday >= Jun 20
    d = _date(year, 6, 20)
    while d.weekday() != 5:
        d = d + timedelta(1)
    holidays.add(d)
    # Alla helgons dag: first Saturday >= Oct 31
    d = _date(year, 10, 31)
    while d.weekday() != 5:
        d = d + timedelta(1)
    holidays.add(d)
    return holidays


def enrich_with_external(
    df: pd.DataFrame, date_col: str,
    weather_df: pd.DataFrame | None = None,
    calendar: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge weather and/or calendar features into df. Returns (enriched_df, added_col_names)."""
    from datetime import timedelta
    df = df.copy()
    added: list[str] = []

    dates_parsed = pd.to_datetime(df[date_col], errors="coerce")
    date_only = dates_parsed.dt.date  # for holiday lookup

    if calendar:
        years = set(dates_parsed.dt.year.dropna().astype(int))
        all_holidays: set = set()
        for y in years:
            try:
                all_holidays |= _swedish_holidays(y)
            except Exception:
                pass

        df["is_holiday"]      = date_only.apply(lambda d: int(d in all_holidays) if pd.notna(d) else 0).astype(int)
        df["is_weekend"]      = dates_parsed.dt.dayofweek.apply(lambda x: int(x >= 5) if pd.notna(x) else 0).astype(int)
        df["season"]          = dates_parsed.dt.month.apply(
            lambda m: 1 if m in (12,1,2) else 2 if m in (3,4,5) else 3 if m in (6,7,8) else 4
            if pd.notna(m) else 0
        ).astype(int)
        hol_list = sorted(all_holidays)
        def days_to_nearest(d):
            if not pd.notna(d): return 0
            if not hol_list: return 0
            return min(abs((d - h).days) for h in hol_list)
        df["days_to_holiday"] = date_only.apply(days_to_nearest).astype(int)
        added += ["is_holiday", "is_weekend", "season", "days_to_holiday"]

    if weather_df is not None and not weather_df.empty:
        weather_df = weather_df.copy()
        weather_df["_merge_date"] = pd.to_datetime(weather_df["date"]).dt.normalize()
        df["_merge_date"]         = dates_parsed.dt.normalize()
        weather_cols = [c for c in weather_df.columns if c not in ("date", "_merge_date")]
        df = df.merge(weather_df[["_merge_date"] + weather_cols], on="_merge_date", how="left")
        df = df.drop(columns=["_merge_date"])
        added += weather_cols

    return df, added


def compute_dataset_meta(df: pd.DataFrame, filename: str) -> dict:
    meta = {
        "filename": filename,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_pct": {c: round(float(df[c].isna().mean() * 100), 1) for c in df.columns},
        "sample_rows": df.head(5).fillna("").astype(str).values.tolist(),
    }
    _DATE_HINTS = ["date","datum","dag","day","week","vecka","month","månad",
                   "period","time","tid","year","år","quarter","kvartal"]
    # First pass: name-based detection
    date_col_found = False
    for col in df.columns:
        if any(h in col.lower() for h in _DATE_HINTS):
            try:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(dates) > 10:
                    meta["period_start"] = str(dates.min().date())
                    meta["period_end"]   = str(dates.max().date())
                    meta["period_days"]  = int((dates.max() - dates.min()).days)
                    date_col_found = True
                    break
            except Exception:
                pass
    # Second pass: value-based detection — skip numeric columns (they parse as epoch ns)
    if not date_col_found:
        for col in df.columns:
            if df[col].dtype.kind in ("i", "u", "f"):  # int / uint / float → skip
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                hit_rate = parsed.notna().mean()
                if hit_rate > 0.8 and parsed.notna().sum() > 10:
                    dates = parsed.dropna()
                    yr_min = int(dates.min().year)
                    yr_max = int(dates.max().year)
                    if yr_min < 1990 or yr_max > 2035:  # sanity-check year range
                        continue
                    meta["period_start"] = str(dates.min().date())
                    meta["period_end"]   = str(dates.max().date())
                    meta["period_days"]  = int((dates.max() - dates.min()).days)
                    break
            except Exception:
                pass
    return meta


# ── Dataset assessment ────────────────────────────────────────────────────────

def run_assessment(df: pd.DataFrame, mapping: dict) -> dict:
    issues, stats = [], {}
    deductions = 0.0

    for col in df.columns:
        pct = float(df[col].isna().mean())
        if pct > 0.3:
            issues.append(f"'{col}' saknar {pct*100:.0f}% av värdena — allvarligt")
            deductions += 0.25
        elif pct > 0.1:
            issues.append(f"'{col}' saknar {pct*100:.0f}% av värdena")
            deductions += 0.10

    date_col = mapping.get("date")
    if date_col and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").sort_values().dropna()
            if len(dates) > 2:
                diffs = dates.diff().dropna().dt.days
                regularity = float(1 - diffs.std() / (diffs.mean() + 1e-9))
                stats["date_regularity"]       = round(regularity, 2)
                stats["median_interval_days"]  = int(diffs.median())
                stats["unique_dates"]          = int(dates.nunique())
                if regularity < 0.5:
                    issues.append("Ojämna tidsintervall — kontrollera att datum är konsistenta")
                    deductions += 0.15
                if dates.nunique() < 30:
                    issues.append(f"Bara {dates.nunique()} unika datum — fler datapunkter ger bättre modell")
                    deductions += 0.15
        except Exception:
            pass

    for generic, col in mapping.items():
        if not col or col not in df.columns:
            continue
        s_raw = df[col]
        if not pd.api.types.is_numeric_dtype(s_raw):
            cleaned = s_raw.astype(str).str.replace(r"\s", "", regex=True).str.replace(",", ".")
            s_raw = pd.to_numeric(cleaned, errors="coerce")
            if s_raw.notna().sum() == 0:
                continue
        s = s_raw.dropna()
        if len(s) == 0:
            continue
        try:
            cv = float(s.std() / (abs(s.mean()) + 1e-9))
            stats[generic] = {
                "col": col,
                "min":       round(float(s.min()), 2),
                "max":       round(float(s.max()), 2),
                "mean":      round(float(s.mean()), 2),
                "std":       round(float(s.std()), 2),
                "cv":        round(cv, 2),
                "zeros_pct": round(float((s == 0).mean() * 100), 1),
            }
            if cv > 3:
                issues.append(f"Hög variation i '{col}' (CV={cv:.1f}) — kontrollera outliers")
                deductions += 0.10
        except Exception:
            pass

    prod_col = mapping.get("product_id")
    if prod_col and prod_col in df.columns:
        stats["n_products"] = int(df[prod_col].nunique())

    return {
        "quality_score": round(max(0.1, 1.0 - deductions), 2),
        "issues": issues,
        "stats": stats,
    }


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, mapping: dict, target_col: str) -> tuple:
    df = df.copy()
    def _to_numeric(s):
        if pd.api.types.is_numeric_dtype(s):
            return s
        cleaned = s.astype(str).str.replace(r"\s", "", regex=True).str.replace(",", ".")
        return pd.to_numeric(cleaned, errors="coerce")
    df[target_col] = _to_numeric(df[target_col]).fillna(0)
    # Find date/product columns by key name OR by detecting date-parseable columns
    date_col = mapping.get("date") or next(
        (v for k, v in mapping.items()
         if v and v in df.columns and "dat" in k.lower()
         and pd.to_datetime(df[v], errors="coerce").notna().mean() > 0.8),
        None
    )
    prod_col = mapping.get("product_id")

    # Identify external numeric columns (not target, not date/id cols from mapping)
    mapped_cols = set(v for v in mapping.values() if v)
    skip_cols   = {target_col, date_col} | (mapped_cols - {target_col})
    _KNOWN_EXTERNAL = {"is_holiday","is_weekend","season","days_to_holiday",
                       "temp_max","temp_min","precip_mm","wind_max"}
    external_cols = [
        c for c in df.columns
        if c not in skip_cols
        and pd.api.types.is_numeric_dtype(df[c])
        and (c in _KNOWN_EXTERNAL or c not in mapped_cols)
    ]

    if date_col and date_col in df.columns:
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)
        # Aggregate: sum target, mean for external features (they're already per-day)
        agg: dict = {target_col: "sum"}
        for c in external_cols:
            agg[c] = "mean"
        agg_df = df.groupby("_date").agg(agg).reset_index()
        df = agg_df.rename(columns={"_date": "__date"})
    else:
        df = df.reset_index(drop=True)
        df["__date"] = pd.RangeIndex(len(df))

    y     = df[target_col].ffill().fillna(0)
    dates = df["__date"]
    feats = pd.DataFrame(index=df.index)

    n = len(df)
    for lag in [1, 2, 3, 7, 14, 28]:
        if lag < n // 5:
            feats[f"lag_{lag}"] = y.shift(lag)
    for w in [7, 14, 28]:
        if w < n // 5:
            feats[f"roll_mean_{w}"] = y.shift(1).rolling(w).mean()
            feats[f"roll_std_{w}"]  = y.shift(1).rolling(w).std()

    if date_col and date_col in df.columns or "__date" in df.columns:
        try:
            d = pd.to_datetime(dates, errors="coerce")
            feats["dayofweek"] = d.dt.dayofweek
            feats["month"]     = d.dt.month
            feats["quarter"]   = d.dt.quarter
            feats["dayofyear"] = d.dt.dayofyear
            feats["year"]      = d.dt.year - int(d.dt.year.min())
        except Exception:
            pass

    # Add external columns directly as features
    for c in external_cols:
        if c in df.columns:
            feats[c] = df[c].values

    # Drop rows where lag/rolling features (not external) are NaN — need at least lag_1
    lag_cols = [c for c in feats.columns if c.startswith("lag_") or c.startswith("roll_")]
    if lag_cols:
        valid = feats[lag_cols].notna().all(axis=1)
    else:
        valid = pd.Series(True, index=feats.index)

    feats  = feats[valid].reset_index(drop=True)
    y      = y[valid].reset_index(drop=True)
    dates  = dates[valid].reset_index(drop=True)

    # Fill any remaining NaN in external/date features with column median then 0
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0)
    return feats, y, dates


# ── ML adapters ───────────────────────────────────────────────────────────────

class BaseAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> float: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def metrics(self, y_true, y_pred) -> dict:
        yt = np.array(y_true, dtype=float)
        yp = np.array(y_pred, dtype=float)
        mae  = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2   = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0
        bias = float(np.mean(yp - yt))
        mask = yt != 0
        mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.sum() > 0 else None
        return {"mae": round(mae,3), "rmse": round(rmse,3),
                "mape": round(mape,1) if mape is not None else None,
                "r2": round(r2,3), "bias": round(bias,3)}

    def feature_importance(self) -> dict:
        return {}


class NaiveAdapter(BaseAdapter):
    name = "Naive Forecast"
    _val: float = 0.0

    def fit(self, X, y):
        t = time.time()
        self._val = float(y.tail(min(7, len(y))).mean())
        return time.time() - t

    def predict(self, X):
        return np.full(len(X), self._val)


class GBMAdapter(BaseAdapter):
    name = "Gradient Boosting"

    def __init__(self):
        self._model = None
        self._feats: list[str] = []

    def fit(self, X, y):
        from sklearn.ensemble import GradientBoostingRegressor
        t = time.time()
        self._feats = list(X.columns)
        m = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, min_samples_leaf=5,
            subsample=0.8, random_state=42,
        )
        m.fit(X, y)
        self._model = m
        return time.time() - t

    def predict(self, X):
        return self._model.predict(X)

    def feature_importance(self):
        if not self._model:
            return {}
        imp   = self._model.feature_importances_
        total = sum(imp) or 1
        return {f: round(float(v / total * 100), 1) for f, v in zip(self._feats, imp)}


class RFAdapter(BaseAdapter):
    name = "Random Forest"

    def __init__(self):
        self._model = None
        self._feats: list[str] = []

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        t = time.time()
        self._feats = list(X.columns)
        m = RandomForestRegressor(
            n_estimators=100, max_depth=6,
            min_samples_leaf=5, n_jobs=1, random_state=42,
        )
        m.fit(X, y)
        self._model = m
        return time.time() - t

    def predict(self, X):
        return self._model.predict(X)

    def feature_importance(self):
        if not self._model:
            return {}
        imp   = self._model.feature_importances_
        total = sum(imp) or 1
        return {f: round(float(v / total * 100), 1) for f, v in zip(self._feats, imp)}


def get_adapters() -> list[BaseAdapter]:
    adapters: list[BaseAdapter] = [NaiveAdapter()]
    for cls in (GBMAdapter, RFAdapter):
        try:
            adapters.append(cls())
        except Exception:
            pass
    return adapters


# ── Walk-forward replay ───────────────────────────────────────────────────────

def run_replay(X: pd.DataFrame, y: pd.Series, dates, adapters: list[BaseAdapter]) -> dict:
    n       = len(X)
    split   = max(int(n * 0.70), 10)
    results = {}

    for adapter in adapters:
        try:
            t_train = adapter.fit(X.iloc[:split], y.iloc[:split])
            y_pred  = adapter.predict(X.iloc[split:])
            y_test  = y.iloc[split:].values
            m       = adapter.metrics(y_test, y_pred)
            m["training_time"] = round(t_train, 2)
            m["feature_importance"] = adapter.feature_importance()
            results[adapter.name] = {
                "metrics": m,
                "actual":    [round(float(v), 3) for v in y_test],
                "predicted": [round(float(v), 3) for v in y_pred],
                "dates":     [str(d)[:10] for d in dates.iloc[split:]],
                "all_dates":  [str(d)[:10] for d in dates],
                "all_actual": [round(float(v), 3) for v in y.values],
            }
        except Exception as e:
            results[adapter.name] = {"error": str(e)}

    best = min(
        (k for k in results if "metrics" in results[k]),
        key=lambda k: results[k]["metrics"]["mae"],
        default=None,
    )
    return {"models": results, "best_model": best, "train_rows": split, "test_rows": n - split}


# ── Business simulation ───────────────────────────────────────────────────────

SIMULATION_RULES = {
    "exact":        "Producera exakt forecast",
    "margin_10":    "Forecast + 10% säkerhetsmarginal",
    "margin_20":    "Forecast + 20% säkerhetsmarginal",
    "min_waste":    "Minimera svinn (0.9× forecast)",
    "max_fill":     "Maximera servicegrad (1.25× forecast)",
}


def apply_rule(predicted: np.ndarray, rule: str) -> np.ndarray:
    p = np.array(predicted, dtype=float)
    if rule == "exact":      return p
    if rule == "margin_10":  return p * 1.10
    if rule == "margin_20":  return p * 1.20
    if rule == "min_waste":  return p * 0.90
    if rule == "max_fill":   return p * 1.25
    return p


# ── ELIR calculation ──────────────────────────────────────────────────────────

def calculate_elir(actual: list, simulated: list) -> dict:
    a = np.array(actual, dtype=float)
    s = np.array(simulated, dtype=float)

    mae_baseline = float(np.abs(a - a.mean()).mean())
    mae_model    = float(np.abs(a - s).mean())

    accuracy_gain = (mae_baseline - mae_model) / (mae_baseline + 1e-9)
    i_factor      = round(min(0.30, max(0.0, accuracy_gain * 0.30)), 3)

    total_actual    = float(a.sum())
    total_simulated = float(s.sum())
    volume_diff_pct = round((total_simulated - total_actual) / (total_actual + 1e-9) * 100, 1)

    confidence = "hög" if i_factor > 0.10 else ("medium" if i_factor > 0.04 else "låg")

    return {
        "i_factor":       i_factor,
        "i_pct":          round(i_factor * 100, 1),
        "accuracy_gain":  round(accuracy_gain * 100, 1),
        "volume_diff_pct":volume_diff_pct,
        "total_actual":   round(total_actual, 1),
        "total_simulated":round(total_simulated, 1),
        "mae_baseline":   round(mae_baseline, 3),
        "mae_model":      round(mae_model, 3),
        "confidence":     confidence,
        "n_samples":      len(a),
    }


# ── Forward forecast ──────────────────────────────────────────────────────────

def run_forecast(df: pd.DataFrame, mapping: dict, target_col: str, n_periods: int) -> dict:
    """
    Fit best model on full historical data, then recursively predict n_periods ahead.
    Returns tail of history for chart context + forecast dates/values.
    """
    X, y, dates = engineer_features(df, mapping, target_col)
    if len(X) < 20:
        return {"error": "För lite data för prognos (minst 20 datapunkter krävs)"}

    # Determine date frequency from the aggregated series
    try:
        date_series = pd.to_datetime(dates, errors="coerce")
        deltas = date_series.diff().dropna().dt.days
        freq_days = int(round(float(deltas.median()))) if len(deltas) else 1
        freq_days = max(1, freq_days)
        last_date = date_series.iloc[-1]
    except Exception:
        freq_days = 1
        last_date = None

    # Fit best available adapter on full dataset
    for AdapterCls in (GBMAdapter, RFAdapter, NaiveAdapter):
        try:
            adapter = AdapterCls()
            adapter.fit(X, y)
            break
        except Exception:
            continue

    feature_cols = list(X.columns)
    lag_cols  = sorted([c for c in feature_cols if c.startswith("lag_")],
                       key=lambda c: int(c.split("_")[1]))
    roll_cols = [c for c in feature_cols if c.startswith("roll_")]

    # Historical buffer for recursive lag/rolling computation
    history = list(y.values)

    # Seasonal means for external weather features (use global average per month)
    _EXT_WEATHER = {"temp_max", "temp_min", "precip_mm", "wind_max"}
    ext_monthly: dict = {}
    for col in _EXT_WEATHER:
        if col in feature_cols:
            try:
                vals = X[col].values
                months = date_series.dt.month.values[-len(vals):]
                monthly = {}
                for m, v in zip(months, vals):
                    monthly.setdefault(m, []).append(v)
                ext_monthly[col] = {m: float(np.mean(vs)) for m, vs in monthly.items()}
            except Exception:
                ext_monthly[col] = {}

    forecast_dates: list  = []
    forecast_values: list = []

    for i in range(n_periods):
        if last_date is not None:
            next_date = last_date + pd.Timedelta(days=freq_days * (i + 1))
        else:
            next_date = None

        row: dict = {}

        # Lag features
        for lc in lag_cols:
            lag_n = int(lc.split("_")[1])
            row[lc] = history[-lag_n] if lag_n <= len(history) else float(np.mean(history))

        # Rolling features
        for rc in roll_cols:
            parts = rc.split("_")
            win = int(parts[-1])
            buf = history[-win:] if len(history) >= win else history
            if "std" in rc:
                row[rc] = float(np.std(buf)) if len(buf) > 1 else 0.0
            else:
                row[rc] = float(np.mean(buf))

        # Calendar features
        if next_date is not None:
            if "dayofweek" in feature_cols: row["dayofweek"] = next_date.dayofweek
            if "month"     in feature_cols: row["month"]     = next_date.month
            if "quarter"   in feature_cols: row["quarter"]   = (next_date.month - 1) // 3 + 1
            if "dayofyear" in feature_cols: row["dayofyear"] = next_date.dayofyear
            if "year"      in feature_cols: row["year"]      = max(0, next_date.year - int(date_series.dt.year.min()))
            if "is_weekend"    in feature_cols: row["is_weekend"]    = 1 if next_date.dayofweek >= 5 else 0
            if "is_holiday"    in feature_cols: row["is_holiday"]    = 0
            if "season"        in feature_cols: row["season"]        = (next_date.month % 12) // 3
            if "days_to_holiday" in feature_cols: row["days_to_holiday"] = 7

            # Weather: use seasonal average for this month
            for col in _EXT_WEATHER:
                if col in feature_cols:
                    m = next_date.month
                    row[col] = ext_monthly.get(col, {}).get(m, float(np.mean(list(ext_monthly.get(col, {1: 0}).values()) or [0])))

        # Fill any remaining features with column median from training data
        for fc in feature_cols:
            if fc not in row:
                row[fc] = float(X[fc].median()) if fc in X.columns else 0.0

        feat_df = pd.DataFrame([row])[feature_cols]
        pred = float(adapter.predict(feat_df)[0])
        pred = max(0.0, pred)

        forecast_dates.append(str(next_date)[:10] if next_date is not None else str(len(history) + i))
        forecast_values.append(round(pred, 3))
        history.append(pred)

    # Return last 90 history points as context for the chart
    tail = min(90, len(y))
    hist_dates  = [str(d)[:10] for d in date_series.iloc[-tail:]]
    hist_actual = [round(float(v), 3) for v in y.iloc[-tail:].values]

    return {
        "history_dates":   hist_dates,
        "history_actual":  hist_actual,
        "forecast_dates":  forecast_dates,
        "forecast_values": forecast_values,
        "freq_days":       freq_days,
        "model":           adapter.name,
    }
