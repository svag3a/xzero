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
    # Second pass: value-based detection (try every column if no name match)
    if not date_col_found:
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                hit_rate = parsed.notna().mean()
                if hit_rate > 0.8 and parsed.notna().sum() > 10:
                    dates = parsed.dropna()
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
        if generic in ("date", "product_id", "customer_id"):
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
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
    date_col    = mapping.get("date")
    prod_col    = mapping.get("product_id")

    if date_col and date_col in df.columns:
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)
        agg_df = df.groupby("_date")[target_col].sum().reset_index()
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

    valid = feats.notna().all(axis=1)
    return feats[valid].reset_index(drop=True), y[valid].reset_index(drop=True), dates[valid].reset_index(drop=True)


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


class LGBMAdapter(BaseAdapter):
    name = "LightGBM"

    def __init__(self):
        self._model = None
        self._feats: list[str] = []

    def fit(self, X, y):
        import lightgbm as lgb
        t = time.time()
        self._feats = list(X.columns)
        m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=5,
            n_jobs=2, verbose=-1,
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
    try:
        adapters.append(LGBMAdapter())
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
    }
