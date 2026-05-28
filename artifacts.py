
from __future__ import annotations

import importlib
import importlib.util
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

try:
    from feature_schema import clean_model_feature_columns
except Exception:  # pragma: no cover - fallback for standalone reuse
    def clean_model_feature_columns(cols, expected_n=None):
        return list(cols) if cols is not None else []

import joblib
import numpy as np
import pandas as pd

# SHAP is lazy-loaded. Importing shap on every Streamlit rerun can dominate
# latency for small batches even when explanations are not requested.
shap = None  # type: ignore
_SHAP_OK = None

def _ensure_shap():
    global shap, _SHAP_OK
    if _SHAP_OK is True:
        return shap
    if _SHAP_OK is False:
        raise RuntimeError("shap package is unavailable")
    try:
        import shap as _shap  # type: ignore
        shap = _shap
        _SHAP_OK = True
        return shap
    except Exception as e:
        _SHAP_OK = False
        raise RuntimeError(f"shap package is unavailable: {e}")


MISS_STRS: Set[str] = {"", "unknown", "unknown-domain", "unknown_domain", "nan", "none", "null", "-", "--", "na", "n/a", "notprovided", "not_provided", "missing_token"}
BAD_STR_RE = re.compile(r"^(?:unknown|unknown-domain|unknown_domain|null|none|nan|na|n/a|-|--|notprovided|not_provided|missing_token)\s*$", re.I)
_IPV4_RE = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")

_MODEL_CANDIDATES = {
    "Decision Tree": ["Decision Tree_model.pkl", "DecisionTree_model.pkl", "decision_tree_model.pkl"],
    "Random Forest": ["Random Forest_model.pkl", "RandomForest_model.pkl", "random_forest_model.pkl"],
    "Logistic Regression": ["Logistic Regression_model.pkl", "LogisticRegression_model.pkl", "logistic_regression_model.pkl"],
    "XGBoost": ["XGBoost_model.pkl", "xgboost_model.pkl", "xgb_model.pkl"],
    "LightGBM": ["LightGBM_model.pkl", "lightgbm_model.pkl", "lgbm_model.pkl"],
    "CatBoost": ["CatBoost_model.pkl", "catboost_model.pkl"],
    "MoE Hybrid": ["MoE_meta_model.pkl"],
}
_CAL_CANDIDATES = {
    "Decision Tree": ["Decision Tree_calibrator.pkl", "DecisionTree_calibrator.pkl", "decision_tree_calibrator.pkl"],
    "Random Forest": ["Random Forest_calibrator.pkl", "RandomForest_calibrator.pkl", "random_forest_calibrator.pkl"],
    "Logistic Regression": ["Logistic Regression_calibrator.pkl", "LogisticRegression_calibrator.pkl", "logistic_regression_calibrator.pkl"],
    "XGBoost": ["XGBoost_calibrator.pkl", "xgboost_calibrator.pkl", "xgb_calibrator.pkl"],
    "LightGBM": ["LightGBM_calibrator.pkl", "lightgbm_calibrator.pkl", "lgbm_calibrator.pkl"],
    "CatBoost": ["CatBoost_calibrator.pkl", "catboost_calibrator.pkl"],
}
_THR_CANDIDATES = {
    "Decision Tree": ["optimal_threshold_Decision Tree.pkl", "optimal_threshold_decision_tree.pkl", "threshold_Decision Tree.pkl"],
    "Random Forest": ["optimal_threshold_Random Forest.pkl", "optimal_threshold_random_forest.pkl", "threshold_Random Forest.pkl"],
    "Logistic Regression": ["optimal_threshold_Logistic Regression.pkl", "optimal_threshold_logistic_regression.pkl", "threshold_Logistic Regression.pkl"],
    "XGBoost": ["optimal_threshold_XGBoost.pkl", "optimal_threshold_xgboost.pkl", "threshold_XGBoost.pkl"],
    "LightGBM": ["optimal_threshold_LightGBM.pkl", "optimal_threshold_lightgbm.pkl", "threshold_LightGBM.pkl"],
    "CatBoost": ["optimal_threshold_CatBoost.pkl", "optimal_threshold_catboost.pkl", "threshold_CatBoost.pkl"],
    "MoE Hybrid": ["optimal_threshold_MoE.pkl", "threshold_MoE.pkl"],
}

_CANON_RENAMES: Dict[str, str] = {
    "src_ip": "client_ip", "source_ip": "client_ip", "sourceip": "client_ip", "srcclient": "client_ip",
    "src_client": "client_ip", "clientip": "client_ip", "ip": "client_ip", "ip_address": "client_ip",
    "ip_addr": "client_ip", "source_address": "client_ip", "src": "client_ip", "client": "client_ip",
    "orig": "client_ip", "caller_ip": "client_ip", "requester_ip": "client_ip",
    "dst_ip": "dest_ip", "destination_ip": "dest_ip", "destinationip": "dest_ip", "destip": "dest_ip",
    "dest": "dest_ip", "dst": "dest_ip", "remote_ip": "dest_ip", "server_ip": "dest_ip", "resp": "dest_ip",
    "sport": "src_port", "source_port": "src_port", "client_port": "src_port",
    "dport": "dst_port", "destination_port": "dst_port", "server_port": "dst_port",
    "protocol": "proto", "service": "proto",
    "host": "domain", "hostname": "domain", "qname": "domain", "sni": "domain", "destination_domain": "domain",
    "dest_domain": "domain", "dst_domain": "domain", "destination_host": "domain",
    "url": "full_url", "request_url": "full_url", "requesturi": "full_url", "request_uri": "full_url",
    "uri": "url_path", "path": "url_path", "request_path": "url_path", "endpoint": "url_path", "resource": "url_path",
    "referer": "referrer", "ref": "referrer",
    "ua": "user_agent", "useragent": "user_agent", "agent": "user_agent",
    "http_method": "method", "verb": "method", "action_method": "method", "op": "method", "event": "method",
    "status_code": "status", "http_status": "status", "code": "status",
    "bytes_sent": "bytes_out", "out_bytes": "bytes_out", "bytesout": "bytes_out", "bytes_outbound": "bytes_out", "sent": "bytes_out", "orig_bytes": "bytes_out",
    "bytes_received": "bytes_in", "in_bytes": "bytes_in", "bytesin": "bytes_in", "bytes_inbound": "bytes_in", "recv": "bytes_in", "resp_bytes": "bytes_in",
    "user": "username", "account": "username", "principal": "username", "login": "username", "user_name": "username", "actor": "username", "requester": "username", "caller": "username",
    "host_name": "workstation", "computer": "workstation", "machine": "workstation", "device": "workstation", "node": "workstation",
    "proc": "process", "process_name": "process", "image": "process", "exe": "process", "application": "process",
    "cmd": "command", "cmdline": "command", "commandline": "command", "command_line": "command", "message": "command", "msg": "command", "description": "command", "reason": "command",
    "time": "timestamp", "datetime": "timestamp", "event_time": "timestamp", "log_time": "timestamp", "logged_at": "timestamp",
    "eventid": "event_id", "eid": "event_id", "logonid": "logon_id",
    "result": "outcome", "decision": "outcome", "action": "outcome",
    "score": "severity", "prio": "severity", "priority": "severity",
    "type": "log_type", "logtype": "log_type", "row_id": "source_row_id", "id": "source_row_id",
    "raw": "raw_log", "raw_line": "raw_log", "rawline": "raw_log", "line": "raw_log", "text": "raw_log",
}


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x)


def _norm_col(name: str) -> str:
    s = _safe_str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")




def _feature_key(name: Any) -> str:
    """Normalize feature names so `Rare Domain`, `Rare_Domain`, and `rare-domain` align."""
    return re.sub(r"[^a-z0-9]+", "_", _safe_str(name).strip().lower()).strip("_")


def _copy_feature_if_present(Xdf: pd.DataFrame, wanted: str) -> None:
    """Populate Xdf[wanted] from an equivalent existing column when exact name is absent."""
    if wanted in Xdf.columns:
        return
    exact_variants = [
        wanted.replace(" ", "_"),
        wanted.replace("_", " "),
        wanted.replace("-", "_"),
        wanted.replace("_", "-"),
    ]
    for alt in exact_variants:
        if alt in Xdf.columns:
            Xdf[wanted] = Xdf[alt]
            return
    key = _feature_key(wanted)
    norm_map: Dict[str, str] = {}
    for col in Xdf.columns:
        norm_map.setdefault(_feature_key(col), str(col))
    src = norm_map.get(key)
    if src is not None and src in Xdf.columns:
        Xdf[wanted] = Xdf[src]
        return
    Xdf[wanted] = 0


def _load_any(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _load_first(model_dir: Path, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        p = model_dir / name
        if p.exists():
            try:
                return _load_any(p)
            except Exception:
                continue
    return default


def _load_set(model_dir: Path, names: Sequence[str]) -> Set[str]:
    for name in names:
        p = model_dir / name
        if not p.exists():
            continue
        try:
            obj = _load_any(p)
            if isinstance(obj, set):
                return {str(x).strip() for x in obj if str(x).strip()}
            if isinstance(obj, (list, tuple)):
                return {str(x).strip() for x in obj if str(x).strip()}
            if isinstance(obj, dict):
                # count maps and boolean maps are both accepted.
                return {str(k).strip() for k, v in obj.items() if str(k).strip() and (bool(v) or isinstance(v, (int, float)))}
            if isinstance(obj, str):
                return {ln.strip() for ln in obj.splitlines() if ln.strip()}
        except Exception:
            # CSV/text fallback below
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                vals = []
                for line in txt.splitlines():
                    parts = [x.strip() for x in line.split(",") if x.strip()]
                    if not parts:
                        continue
                    cand = parts[-1] if len(parts) > 1 and parts[0].isdigit() else parts[0]
                    cand = cand.lower().replace("http://", "").replace("https://", "").split("/")[0].split(":")[0]
                    if cand and cand not in {"domain", "domains", "host", "hostname"}:
                        vals.append(cand)
                if vals:
                    return set(vals)
            except Exception:
                continue
    return set()


def _load_domain_union(model_dir: Path, names: Sequence[str]) -> Set[str]:
    """Load domain-like sets from pkl/joblib/csv/txt across model/data dirs."""
    out: Set[str] = set()
    roots = [model_dir, model_dir / "data", model_dir.parent, model_dir.parent / "data"]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            s = _load_set(root, [name])
            for x in s:
                y = str(x).strip().lower().replace("http://", "").replace("https://", "").split("/")[0].split(":")[0].strip(".")
                if y and "." in y and y not in {"domain", "domains", "host", "hostname"}:
                    out.add(y)
    return out


def _sanitize_feature_columns(cols: Any) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    if cols is None:
        return out
    if not isinstance(cols, (list, tuple)):
        try:
            cols = list(cols)
        except Exception:
            return out
    for c in cols:
        s = _safe_str(c).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _extract_model_feature_names(model: Any) -> List[str]:
    names: List[str] = []
    if model is None:
        return names

    def _take(obj: Any):
        nonlocal names
        if obj is None or names:
            return
        try:
            vals = list(obj)
            vals = [str(v).strip() for v in vals if _safe_str(v).strip()]
            if vals:
                names = _sanitize_feature_columns(vals)
        except Exception:
            pass

    for attr in ("feature_names_in_", "feature_name_", "feature_names"):
        _take(getattr(model, attr, None))
        if names:
            return names

    booster = getattr(model, "booster_", None)
    if booster is None:
        booster = getattr(model, "_Booster", None)
    if booster is None and hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
        except Exception:
            booster = None
    if booster is not None and hasattr(booster, "feature_name"):
        try:
            vals = booster.feature_name()
            _take(vals)
        except Exception:
            pass

    if names:
        return names

    nested = getattr(model, "estimator", None)
    if nested is not None and nested is not model:
        names = _extract_model_feature_names(nested)
        if names:
            return names

    for cc in getattr(model, "calibrated_classifiers_", []) or []:
        nested = getattr(cc, "estimator", None)
        names = _extract_model_feature_names(nested)
        if names:
            return names

    return names


def _extract_scaler_feature_names(scaler: Any) -> List[str]:
    if scaler is None:
        return []
    for attr in ("feature_names_in_", "feature_names"):
        vals = getattr(scaler, attr, None)
        cols = _sanitize_feature_columns(vals)
        if cols:
            return cols
    return []


def _infer_expected_n_features(obj: Any) -> Optional[int]:
    seen: Set[int] = set()
    queue: List[Any] = [obj]
    while queue:
        cur = queue.pop(0)
        if cur is None:
            continue
        cid = id(cur)
        if cid in seen:
            continue
        seen.add(cid)

        n = getattr(cur, "n_features_in_", None)
        try:
            if n is not None:
                n = int(n)
                if n > 0:
                    return n
        except Exception:
            pass

        for attr in ("estimator", "base_estimator", "model"):
            nxt = getattr(cur, attr, None)
            if nxt is not None:
                queue.append(nxt)

        for cc in getattr(cur, "calibrated_classifiers_", []) or []:
            queue.append(cc)
            nxt = getattr(cc, "estimator", None)
            if nxt is not None:
                queue.append(nxt)
    return None


def _align_feature_width(X: Any, expected_n: Optional[int]):
    if expected_n is None:
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    cur_n = int(arr.shape[1])
    if cur_n == int(expected_n):
        return arr
    if cur_n > int(expected_n):
        return arr[:, : int(expected_n)]
    pad = np.zeros((arr.shape[0], int(expected_n) - cur_n), dtype=arr.dtype if hasattr(arr, 'dtype') else float)
    return np.hstack([arr, pad])


def _resolve_feature_columns(mdir: Path, existing: Any, scaler: Any) -> List[str]:
    """Resolve model feature columns without eagerly loading heavy models.

    The earlier resolver opened LightGBM/XGBoost/RF/CatBoost files just to read
    feature names.  On Streamlit reruns this made tiny batches feel slow.  The
    training notebooks already save extended_feature_columns.pkl / scaler.pkl;
    prefer those artifacts and only inspect model pickles as a last resort.
    """
    existing_cols = _sanitize_feature_columns(existing)
    scaler_cols = _extract_scaler_feature_names(scaler)
    scaler_n = getattr(scaler, "n_features_in_", None) if scaler is not None else None
    try:
        scaler_n = int(scaler_n) if scaler_n is not None else None
    except Exception:
        scaler_n = None

    if existing_cols:
        chosen = existing_cols
    elif scaler_cols:
        chosen = scaler_cols
    else:
        inferred: List[str] = []
        for ui_name in ("LightGBM", "XGBoost", "CatBoost", "Random Forest", "Decision Tree", "Logistic Regression"):
            for fname in _MODEL_CANDIDATES.get(ui_name, []):
                p = mdir / fname
                if not p.exists():
                    continue
                try:
                    model = _load_any(p)
                except Exception:
                    continue
                inferred = _extract_model_feature_names(model)
                if inferred:
                    break
            if inferred:
                break
        chosen = inferred

    if scaler_n is not None:
        if chosen:
            if len(chosen) > scaler_n:
                chosen = chosen[:scaler_n]
            elif len(chosen) < scaler_n:
                chosen = chosen + [f"__pad_feature_{i}" for i in range(len(chosen), scaler_n)]
        else:
            chosen = [f"f{i}" for i in range(scaler_n)]

    chosen = _sanitize_feature_columns(chosen)
    try:
        chosen = clean_model_feature_columns(chosen, expected_n=scaler_n)
    except Exception:
        pass
    return _sanitize_feature_columns(chosen)

def _merge_bytes_priors(base: Dict[str, Any], dyn: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    if dyn:
        out["dynamic"] = dyn
    return out


@dataclass
class Bundle:
    model_dir: str
    priors: Dict[str, Any] = field(default_factory=dict)
    resolver_state: Dict[str, Any] = field(default_factory=dict)
    bytes_priors: Dict[str, Any] = field(default_factory=dict)
    bad_ips: Set[str] = field(default_factory=set)
    bad_domains: Set[str] = field(default_factory=set)
    top_domains: Set[str] = field(default_factory=set)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    scaler: Any = None
    _supervised_cache: Dict[str, tuple] = field(default_factory=dict, repr=False)

    def load_supervised(self, ui_name: str):
        if ui_name in self._supervised_cache:
            return self._supervised_cache[ui_name]
        mdir = Path(self.model_dir)
        model = _load_first(mdir, _MODEL_CANDIDATES.get(ui_name, []), None)
        calibrator = _load_first(mdir, _CAL_CANDIDATES.get(ui_name, []), None)
        threshold = _load_first(mdir, _THR_CANDIDATES.get(ui_name, []), None)
        try:
            threshold = None if threshold is None else float(threshold)
        except Exception:
            threshold = None
        out = (model, calibrator, threshold)
        self._supervised_cache[ui_name] = out
        return out


def load_bundle(model_dir: str) -> Bundle:
    mdir = Path(model_dir)
    if not mdir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    priors = _load_first(mdir, ["priors.pkl", "priors.joblib"], {}) or {}
    resolver_state = _load_first(mdir, ["resolver_state.pkl", "resolver.pkl"], {}) or {}

    # Advanced imputation notebooks save high-confidence repair maps as
    # forensic_artifacts/repair_maps.pkl. Merge them into resolver_state so the
    # runtime imputer can repair domain/workstation/username without generic
    # unresolved placeholders.
    repair_maps = {}
    for root in [mdir, mdir / "forensic_artifacts", mdir / "data", mdir.parent, mdir.parent / "forensic_artifacts", mdir.parent / "data"]:
        if not root.exists():
            continue
        repair_maps = _load_first(root, [
            "repair_maps.pkl", "forensic_repair_maps.pkl", "advanced_repair_maps.pkl",
            "imputation_repair_maps.pkl", "resolver_repair_maps.pkl",
        ], {}) or {}
        if isinstance(repair_maps, dict) and repair_maps:
            break
    if isinstance(repair_maps, dict) and repair_maps:
        resolver_state = dict(resolver_state) if isinstance(resolver_state, dict) else {}
        resolver_state.setdefault("repair_maps", {}).update(repair_maps)
        # Also expose maps at top level for direct lookups.
        for k, v in repair_maps.items():
            if isinstance(v, dict):
                resolver_state.setdefault(k, v)

    bytes_priors = _load_first(mdir, [
        # Names used across the uploaded imputation notebooks.
        "bytes_priors.pkl", "byte_priors.pkl", "byte_priors_hier_baseline.pkl",
        "bytes_priors_hier_baseline.pkl", "baseline_bytes_priors.pkl", "baseline_byte_priors.pkl",
    ], {}) or {}
    bytes_dyn = _load_first(mdir, ["bytes_dynamic_priors.pkl", "byte_priors_dynamic.pkl", "byte_priors_hier_dynamic.pkl"], {}) or {}
    bytes_priors = _merge_bytes_priors(bytes_priors, bytes_dyn)

    bad_ips = _load_set(mdir, ["bad_ips.pkl", "known_bad_ips.pkl", "malicious_ips.pkl", "ioc_bad_ips.pkl", "blacklisted_ips.pkl"])
    bad_domains = _load_set(mdir, ["bad_domains.pkl", "known_bad_domains.pkl", "malicious_domains.pkl", "ioc_bad_domains.pkl", "blacklisted_domains.pkl"])
    top_domains = _load_domain_union(mdir, [
        "top_domains.pkl", "top_domain_set.pkl", "top_1m_domains.pkl", "top-1m.csv", "top_1m.csv",
        "whitelist_domains.pkl", "trusted_domains.pkl", "benign_domains.pkl", "common_domains.pkl", "whitelist.csv",
    ])
    feature_weights = _load_first(mdir, ["feature_weights.pkl"], {}) or {}
    imputation_defaults = _load_first(
        mdir,
        [
            "imputation_defaults.pkl",
            "advanced_imputation_defaults.pkl",
            "column_defaults.pkl",
            "missing_value_artifacts.pkl",
            "imputer_defaults.pkl",
        ],
        {},
    ) or {}
    if isinstance(imputation_defaults, dict):
        priors = dict(priors) if isinstance(priors, dict) else {}
        # Preserve nested artifacts while also exposing a uniform column_defaults map.
        priors.setdefault("imputation_defaults", imputation_defaults)
        if any(k in imputation_defaults for k in ("column_defaults", "text_defaults", "categorical_defaults", "fill_values")):
            for k in ("column_defaults", "text_defaults", "categorical_defaults", "fill_values"):
                if isinstance(imputation_defaults.get(k), dict):
                    priors.setdefault(k, {}).update(imputation_defaults[k])
        else:
            priors.setdefault("column_defaults", {}).update(imputation_defaults)
    raw_feature_columns = _load_first(mdir, [
        # Supervised training saved this exact model schema. Prefer it over
        # feature_columns.pkl, which the adaptive-labeling notebook may use for
        # expert/rule features or full FE outputs.
        "extended_feature_columns.pkl", "model_feature_columns.pkl", "model_features.pkl", "feature_names.pkl",
        "feature_columns.pkl",
    ], []) or []
    scaler = _load_first(mdir, ["scaler.pkl"], None)
    feature_columns = _resolve_feature_columns(mdir, raw_feature_columns, scaler)

    return Bundle(
        model_dir=str(mdir),
        priors=priors if isinstance(priors, dict) else {},
        resolver_state=resolver_state if isinstance(resolver_state, dict) else {},
        bytes_priors=bytes_priors if isinstance(bytes_priors, dict) else {},
        bad_ips={x for x in bad_ips if _safe_str(x).strip()},
        bad_domains={x.lower() for x in bad_domains if _safe_str(x).strip()},
        top_domains={x.lower() for x in top_domains if _safe_str(x).strip()},
        feature_weights=feature_weights if isinstance(feature_weights, dict) else {},
        feature_columns=feature_columns,
        scaler=scaler,
    )


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    ren: Dict[str, str] = {}
    for col in out.columns:
        norm = _norm_col(col)
        target = _CANON_RENAMES.get(norm)
        if target and target not in out.columns:
            ren[col] = target
    if ren:
        out = out.rename(columns=ren)
    return out


def _import_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_feature_engineering(bundle: Bundle) -> Callable[[pd.DataFrame], pd.DataFrame]:
    candidates: List[Path] = []
    here = Path(__file__).resolve().parent
    mdir = Path(bundle.model_dir).resolve()
    for base in [here, here / "utils", mdir, mdir.parent, Path.cwd(), Path.cwd() / "utils"]:
        for name in ("feature_engineering.py", "custom_feature_engineering.py", "feature_engineering_module.py"):
            p = base / name
            if p.exists():
                candidates.append(p)
    for p in candidates:
        try:
            mod = _import_module_from_path(p)
            for fn_name in ("our_custom_feature_engineering_function", "custom_feature_engineering_function", "feature_engineering_function", "build_features", "featurize"):
                fn = getattr(mod, fn_name, None)
                if callable(fn):
                    return fn
        except Exception:
            continue
    for mod_name in ("feature_engineering", "custom_feature_engineering"):
        try:
            mod = importlib.import_module(mod_name)
            for fn_name in ("our_custom_feature_engineering_function", "custom_feature_engineering_function", "feature_engineering_function", "build_features", "featurize"):
                fn = getattr(mod, fn_name, None)
                if callable(fn):
                    return fn
        except Exception:
            continue
    raise ImportError("Could not load feature engineering function")


_HIGH_CONF_RE = re.compile(
    r"(?i)(?:"
    r"\bverdict(?:=|\s)(?:malicious|phish|malware|spoof)\b|"
    r"\bET\s+(?:MALWARE|TROJAN)\b|\bPossible\s+C2\b|\bNmap\s+Scripting\s+Engine\b|"
    r"\brule=(?:SQLI|RCE|PATH_TRAVERSAL)\b|"
    r"\blsass_dump\.exe\b|\bcat\s+/etc/shadow\b|"
    r"\bcertutil(?:\.exe)?\b[^\n\r]{0,140}-urlcache\s+-split\s+-f\s+https?://|"
    r"\bmsxsl(?:\.exe)?\b[^\n\r]{0,180}https?://|"
    r"\bdesktopimgdownldr(?:\.exe)?\b[^\n\r]{0,180}/lockscreenurl:\s*https?://|"
    r"\bms-appinstaller://\?source=https?://|"
    r"\bcurl\b[^\n\r]{0,120}\|\s*(?:sh|bash)\b|"
    r"\bnc\b\s+\d{1,3}(?:\.\d{1,3}){3}\s+4444\b|"
    r"\b(?:StopLogging|DisableMailboxAudit|ExportWorkspaceData|bypass_dlp)\b|"
    r"\bcluster-admin\b|\bclusterrolebindings\b|privileged=true|Created\s+privileged\s+pod"
    r")"
)
_BENIGN_RE = re.compile(r"(?i)(?:\bnon[-_ ]?malicious\b|\bnot[-_ ]?malicious\b|\bverdict(?:=|\s)(?:clean|benign|legit)\b|\bfalse\s*positive\b)")


def _series_text(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    parts = []
    for c in cols:
        if c in df.columns:
            parts.append(df[c].fillna("").astype(str))
    if not parts:
        return pd.Series("", index=df.index, dtype="object")
    out = parts[0]
    for s in parts[1:]:
        out = out + " " + s
    return out


def compute_primary_flags(df: pd.DataFrame, bundle: Bundle) -> np.ndarray:
    if df is None or len(df) == 0:
        return np.zeros(0, dtype=np.int8)
    # Prefer the family-aware override engine when available. It suppresses
    # benign DHCP/proxy/firewall/clean-EDR cases while preserving hard security verdicts and IOCs.
    try:
        from override_rules import compute_override_signals
        flags, _codes, _debug = compute_override_signals(
            df,
            X=None,
            bad_ips=getattr(bundle, "bad_ips", set()),
            bad_domains=getattr(bundle, "bad_domains", set()),
            top_domains=getattr(bundle, "top_domains", set()),
            whitelist_domains=getattr(bundle, "top_domains", set()),
        )
        return flags.astype(np.int8)
    except Exception:
        pass
    cip = df.get("client_ip", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    dip = df.get("dest_ip", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    dom = df.get("domain", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().str.lower()
    blob = _series_text(df, ["raw_log", "raw", "command", "process", "user_agent", "referrer", "full_url", "domain", "outcome", "severity", "event_id", "proto", "method", "status"]).str.lower()

    bad_ip_hit = cip.isin(bundle.bad_ips) | dip.isin(bundle.bad_ips)
    bad_dom_hit = dom.isin(bundle.bad_domains)
    text_hit = blob.str.contains(_HIGH_CONF_RE, na=False) & (~blob.str.contains(_BENIGN_RE, na=False))
    edr_hit = blob.str.contains(r"\bedr\b", na=False) & blob.str.contains(r"\bverdict(?:=|\s)(?:malicious|malware)\b", na=False)
    email_hit = blob.str.contains(r"\bemail_sec\b", na=False) & blob.str.contains(r"\bverdict(?:=|\s)(?:phish|malware|spoof)\b", na=False)
    ids_hit = blob.str.contains(r"\bids\b", na=False) & blob.str.contains(r"(?:possible c2|et malware|et trojan|nmap scripting engine|suspicious tls sni)", na=False)
    waf_hit = blob.str.contains(r"\bwaf\b", na=False) & blob.str.contains(r"\baction(?:=|\s)(?:block|deny|drop)\b", na=False) & blob.str.contains(r"(?:sqli|rce|path_traversal|\.env)", na=False)

    return (bad_ip_hit | bad_dom_hit | text_hit | edr_hit | email_hit | ids_hit | waf_hit).astype(np.int8).to_numpy(dtype=np.int8, copy=False)


def prepare_model_matrix(X: pd.DataFrame, scaler: Any, feature_cols: Sequence[str]):
    """Return a numeric model matrix with lossless feature-name alignment.

    The training notebook used a few names with spaces (for example
    ``domain_cat_Rare Domain``), while the Streamlit app historically sanitized
    spaces to underscores.  This function aligns exact names first, then
    underscore/space/dash-normalized aliases, and only then pads with zero.
    """
    Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    Xdf.columns = [str(c) for c in Xdf.columns]

    expected_n0 = getattr(scaler, "n_features_in_", None) if scaler is not None else None
    try:
        expected_n0 = int(expected_n0) if expected_n0 is not None else None
    except Exception:
        expected_n0 = None
    cols = _sanitize_feature_columns(feature_cols)
    if not cols:
        cols = _extract_scaler_feature_names(scaler)
    try:
        cols = clean_model_feature_columns(cols, expected_n=expected_n0)
    except Exception:
        pass

    if cols:
        for c in cols:
            _copy_feature_if_present(Xdf, c)
        Xdf = Xdf.loc[:, cols].copy()

    for c in Xdf.columns:
        if pd.api.types.is_bool_dtype(Xdf[c]):
            Xdf[c] = Xdf[c].astype(np.int8)
        elif not pd.api.types.is_numeric_dtype(Xdf[c]):
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = Xdf.to_numpy(dtype=float, copy=False)

    expected_n = getattr(scaler, "n_features_in_", None) if scaler is not None else None
    try:
        expected_n = int(expected_n) if expected_n is not None else None
    except Exception:
        expected_n = None

    if scaler is not None and hasattr(scaler, "transform"):
        try:
            arr = np.asarray(scaler.transform(arr), dtype=float)
        except Exception:
            arr = _align_feature_width(arr, expected_n)
            try:
                arr = np.asarray(scaler.transform(arr), dtype=float)
            except Exception:
                # Keep the validated numeric matrix rather than failing inference.
                pass
    else:
        arr = _align_feature_width(arr, expected_n)

    return arr, list(Xdf.columns), Xdf


def predict_proba_with_optional_calibrator(model: Any, calibrator: Any, X_scaled, use_calibrator: bool = True) -> np.ndarray:
    target = calibrator if (use_calibrator and calibrator is not None) else model
    if target is None:
        return np.zeros(len(X_scaled), dtype=float)

    expected_n = _infer_expected_n_features(target)
    X_use = _align_feature_width(X_scaled, expected_n)

    if hasattr(target, "predict_proba"):
        try:
            p = np.asarray(target.predict_proba(X_use))
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1].astype(float, copy=False)
            return p.reshape(-1).astype(float, copy=False)
        except Exception:
            try:
                X_use2 = _align_feature_width(X_scaled, _infer_expected_n_features(model))
                p = np.asarray(target.predict_proba(X_use2))
                if p.ndim == 2 and p.shape[1] >= 2:
                    return p[:, 1].astype(float, copy=False)
                return p.reshape(-1).astype(float, copy=False)
            except Exception:
                pass
    if hasattr(target, "decision_function"):
        try:
            s = np.asarray(target.decision_function(X_use), dtype=float).reshape(-1)
            return 1.0 / (1.0 + np.exp(-np.clip(s, -50, 50)))
        except Exception:
            pass
    try:
        return np.asarray(target.predict(X_use), dtype=float).reshape(-1)
    except Exception:
        return np.zeros(len(X_use), dtype=float)


class ShapEngine:
    def __init__(self, bundle: Bundle, background_rows: int = 256):
        self.bundle = bundle
        self.background_rows = int(background_rows)
        self._cache: Dict[int, Any] = {}

    def _tree_like(self, model: Any) -> bool:
        name = type(model).__name__.lower()
        return any(k in name for k in ("lgbm", "xgb", "randomforest", "decisiontree", "catboost", "forest", "tree", "boost"))

    def _make_explainer(self, model: Any, X_df: pd.DataFrame):
        shap_mod = _ensure_shap()
        key = id(model)
        if key in self._cache:
            return self._cache[key]
        if self._tree_like(model):
            explainer = shap_mod.TreeExplainer(model)
        else:
            bg = X_df if len(X_df) <= self.background_rows else X_df.sample(self.background_rows, random_state=42)
            explainer = shap_mod.Explainer(model, bg)
        self._cache[key] = explainer
        return explainer

    @staticmethod
    def _coerce(vals: Any) -> np.ndarray:
        if isinstance(vals, list):
            vals = vals[-1]
        if hasattr(vals, "values"):
            vals = vals.values
        arr = np.asarray(vals, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 3 and arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        return arr

    @staticmethod
    def _breakdown(vec: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(vec, dtype=float).reshape(-1)
        mag = np.abs(arr)
        total = float(mag.sum())
        if total <= 1e-12:
            return {"pos_pct": 0.0, "neg_pct": 0.0, "neutral_pct": 100.0, "net_pct": 0.0}
        pos = float(mag[arr > 1e-12].sum())
        neg = float(mag[arr < -1e-12].sum())
        neu = float(mag[(arr >= -1e-12) & (arr <= 1e-12)].sum())
        return {"pos_pct": 100.0 * pos / total, "neg_pct": 100.0 * neg / total, "neutral_pct": 100.0 * neu / total, "net_pct": 100.0 * (pos - neg) / total}

    def compute_for_rows(self, model: Any, X_df: pd.DataFrame, row_indices: Sequence[int], ip_bad_truth=None, ip_private_truth=None, topk: int = 80, semantic_ip: bool = True) -> Dict[str, Any]:
        if X_df is None or len(X_df) == 0:
            raise ValueError("X_df is empty")
        idx = list(row_indices)
        if not idx:
            raise ValueError("row_indices is empty")
        X_sel = X_df.iloc[idx].copy()
        explainer = self._make_explainer(model, X_df)
        sv = self._coerce(explainer.shap_values(X_sel))
        feat_names = list(X_sel.columns)
        rows = [self._breakdown(v) for v in sv]
        mean = self._breakdown(np.mean(sv, axis=0))
        return {"sv_rows": sv, "feat_names": feat_names, "breakdown_rows": rows, "breakdown_mean": mean, "row_indices": idx, "topk": int(topk), "semantic_ip": bool(semantic_ip)}


__all__ = ["Bundle", "load_bundle", "canonicalize_columns", "load_feature_engineering", "compute_primary_flags", "prepare_model_matrix", "predict_proba_with_optional_calibrator", "ShapEngine"]
