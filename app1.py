# utils/app.py
# ============================================================
# CyberSecurity Log Classifier — FINAL HARDENED APP (NO-LOSS)
# ------------------------------------------------------------
# Core guarantees (best-effort, production-hardened):
# ✅ Primary override conditions:
#    - explicit/critical keywords => ALWAYS malicious
#    - moderate keywords => malicious when combined with strong signals
#    - ipv6 tunnel combos + high-risk signals => malicious
# ✅ Whitelist CSV usage:
#    - dampens probability ONLY if whitelist_hit AND NOT suspicious_context AND info_complete
#    - modes: Off/Soft/Medium/Hard/Custom
# ✅ SHAP:
#    - dashboard: row / subset / overall (avg) [computed on preview/sample rows for speed]
#    - PDF: includes SHAP charts + contribution tables when enabled
# ✅ Performance metrics:
#    - dashboard: metrics + ROC/PR curves (if labels provided)
#    - PDF: metrics + CM + ROC/PR curves (if labels provided)
# ✅ PDF (professional, readable):
#    - summary counts & %
#    - exfil tables by domain and dest_ip (bytes_out)
#    - per-entry blocks with standardized key-values + full raw log
# ✅ Hybrid/multi-type logs:
#    - parser + imputer + FE handle mixed types in one batch
# ✅ ONNX acceleration:
#    - optional onnxruntime fast path with parity guard + Python fallback
#    - supports existing .onnx artifacts and safe sklearn auto-conversion cache
# ✅ 1GB uploads:
#    - app writes Streamlit config and streams raw uploads line-by-line
# ============================================================

from __future__ import annotations

import os
import sys
import re
import io
import csv
import json
import smtplib
import warnings
import importlib
import importlib.util
import tempfile
import inspect
import ipaddress
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from typing import Any, Dict, Optional, Tuple, List, Iterable

# ============================================================
# Streamlit upload bootstrap (1GB)
# ------------------------------------------------------------
# Streamlit's file_uploader limit is controlled by server config.
# The environment variables below are read by Streamlit when available;
# the config file is written before the first Streamlit command so future
# launches from this app directory inherit a 1GB cap automatically.
# ============================================================
STREAMLIT_UPLOAD_MAX_MB = int(os.getenv("CYBER_MAX_UPLOAD_MB", "1024"))
os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", str(STREAMLIT_UPLOAD_MAX_MB))
os.environ.setdefault("STREAMLIT_SERVER_MAX_MESSAGE_SIZE", str(STREAMLIT_UPLOAD_MAX_MB))
_STREAMLIT_UPLOAD_CONFIG_NOTE = ""

def _bootstrap_streamlit_upload_config(max_mb: int = STREAMLIT_UPLOAD_MAX_MB) -> str:
    try:
        app_dir_boot = os.path.dirname(os.path.abspath(__file__))
        cfg_dir = os.path.join(app_dir_boot, ".streamlit")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, "config.toml")
        existing = ""
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    existing = f.read()
            except Exception:
                existing = ""
        def _merge_server_upload_limits(text: str) -> str:
            lines = text.splitlines()
            target = [f"maxUploadSize = {int(max_mb)}", f"maxMessageSize = {int(max_mb)}"]
            server_idx = None
            for i, line in enumerate(lines):
                if line.strip() == "[server]":
                    server_idx = i
                    break
            if server_idx is None:
                prefix = text.rstrip() + ("\n\n" if text.strip() else "")
                return prefix + "[server]\n" + "\n".join(target) + "\nenableXsrfProtection = true\n"
            end_idx = len(lines)
            for j in range(server_idx + 1, len(lines)):
                if lines[j].strip().startswith("[") and lines[j].strip().endswith("]"):
                    end_idx = j
                    break
            cleaned = []
            for line in lines[server_idx + 1:end_idx]:
                key = line.split("=", 1)[0].strip()
                if key in {"maxUploadSize", "maxMessageSize"}:
                    continue
                cleaned.append(line)
            merged_lines = lines[:server_idx + 1] + target + cleaned + lines[end_idx:]
            return "\n".join(merged_lines).rstrip() + "\n"

        merged = _merge_server_upload_limits(existing)
        if merged != existing:
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(merged)
            return f"Wrote {cfg_path} with {int(max_mb)} MB upload/message limits. Restart Streamlit once if this was the first launch."
        return f"Upload config present at {cfg_path}."
    except Exception as e:
        return f"Could not write .streamlit/config.toml automatically: {e}"

_STREAMLIT_UPLOAD_CONFIG_NOTE = _bootstrap_streamlit_upload_config(STREAMLIT_UPLOAD_MAX_MB)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

try:
    from feature_schema import clean_model_feature_columns
except Exception:
    def clean_model_feature_columns(cols, expected_n=None):
        return list(cols) if cols is not None else []

from urllib.parse import unquote

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

# ============================================================
# ✅ MUST BE FIRST STREAMLIT COMMAND
# ============================================================
_PAGE_CONFIG_ERR = None
try:
    st.set_page_config(
        page_title="CyberSecurity Log Classifier",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception as e:
    _PAGE_CONFIG_ERR = str(e)

_UPLOAD_RUNTIME_ERR = ""
try:
    # Some Streamlit versions reject server.* changes after startup.  This is
    # best-effort; the config/env bootstrap above is the durable path.
    st.set_option("server.maxUploadSize", int(STREAMLIT_UPLOAD_MAX_MB))
    st.set_option("server.maxMessageSize", int(STREAMLIT_UPLOAD_MAX_MB))
except Exception as e:
    _UPLOAD_RUNTIME_ERR = str(e)

# ---- Stability knobs ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None
np.seterr(all="ignore")

# ============================================================
# GLOBAL SAFE REGEX PATCH (prevents "global flags not at the start")
# ============================================================
if not getattr(re, "_SAFE_COMPILE_PATCHED", False):
    _orig_compile = re.compile
    _FLAG_MAP = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE, "a": re.ASCII, "u": 0}
    _GLOBAL_INLINE_FLAGS_RE = _orig_compile(r"\(\?([aimsxau]+)\)")

    def _safe_compile(pattern, flags: int = 0):
        try:
            return _orig_compile(pattern, flags)
        except re.error as e:
            if "global flags not at the start" not in str(e):
                raise
            if not isinstance(pattern, str):
                raise
            add_flags = 0

            def _strip(m):
                nonlocal add_flags
                for ch in m.group(1):
                    add_flags |= _FLAG_MAP.get(ch, 0)
                return ""

            fixed = _GLOBAL_INLINE_FLAGS_RE.sub(_strip, pattern)
            return _orig_compile(fixed, flags | add_flags)

    re.compile = _safe_compile  # type: ignore
    re._SAFE_COMPILE_PATCHED = True  # type: ignore

# ============================================================
# Paths / sys.path
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ============================================================
# UI theme
# ============================================================
st.markdown(
    """
<style>
:root{
  --bg:#FFFFFF; --sidebar:#F6F7F9; --card:#FFFFFF; --border:#E5E7EB;
  --text:#111827; --muted:#6B7280; --ok:#16A34A; --bad:#DC2626;
}
html, body, [data-testid="stAppViewContainer"]{ background:var(--bg)!important; color:var(--text)!important; }
[data-testid="stSidebar"]{ background:var(--sidebar)!important; }
.hr{ border:0; height:1px; background:var(--border); margin:12px 0; }
.block{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 16px; }
.good{ color:var(--ok)!important; font-weight:700; }
.bad{ color:var(--bad)!important; font-weight:700; }
.small{ color:var(--muted)!important; font-size:0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("<h1>🛡️ CyberSecurity Log Classifier</h1>", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

if _PAGE_CONFIG_ERR:
    with st.sidebar.expander("ℹ️ Streamlit config", expanded=False):
        st.caption("set_page_config() warning (safe to ignore if multipage/app already set it):")
        st.code(_PAGE_CONFIG_ERR)

with st.sidebar.expander("📦 Upload limit", expanded=False):
    st.caption(f"Target upload/message limit: {STREAMLIT_UPLOAD_MAX_MB} MB (1GB mode).")
    if _STREAMLIT_UPLOAD_CONFIG_NOTE:
        st.caption(_STREAMLIT_UPLOAD_CONFIG_NOTE)
    if _UPLOAD_RUNTIME_ERR:
        st.caption("Runtime update was not accepted by this Streamlit server; config/env bootstrap is still in place.")
        st.code(_UPLOAD_RUNTIME_ERR)
    try:
        st.caption(f"Streamlit reports maxUploadSize={st.get_option('server.maxUploadSize')} MB, maxMessageSize={st.get_option('server.maxMessageSize')} MB.")
    except Exception:
        pass

# ============================================================
# Safe PDF text (unicode normalization)
# ============================================================
def safe_pdf_text(s: str) -> str:
    rep = {"🚨": "[MAL]", "✅": "[OK]", "↑": "^", "↓": "v", "→": "->", "—": "-", "–": "-", "…": "..."}
    s = "" if s is None else str(s)
    for k, v in rep.items():
        s = s.replace(k, v)
    return s

# ============================================================
# Robust import of local artifacts.py
# ============================================================
def _import_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _load_local_artifacts_module(root_dir: str, app_dir: str) -> Tuple[Any, str]:
    candidates = [
        os.path.join(root_dir, "artifacts.py"),
        os.path.join(root_dir, "utils", "artifacts.py"),
        os.path.join(app_dir, "artifacts.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                mod = _import_module_from_file("artifacts", path)
                sys.modules["artifacts"] = mod
                return mod, f"file:{path}"
            except Exception:
                pass
    mod = importlib.import_module("artifacts")
    return mod, f"import:{getattr(mod, '__file__', 'artifacts')}"

ART, ART_SRC = _load_local_artifacts_module(ROOT_DIR, APP_DIR)
with st.sidebar.expander("🧩 Artifacts import", expanded=False):
    st.caption(f"Using artifacts module from: {ART_SRC}")

def _try_get(name: str, default=None):
    return getattr(ART, name, default)

# ============================================================
# Fast artifact/model cache
# ============================================================
try:
    _cache_resource_no_spinner = st.cache_resource(show_spinner=False)
except Exception:
    def _cache_resource_no_spinner(func=None, **_kwargs):
        def deco(f):
            return f
        return deco(func) if func is not None else deco

def _artifact_signature(path: str) -> tuple[str, int, int]:
    p = os.path.abspath(str(path))
    try:
        stt = os.stat(p)
        return p, int(getattr(stt, "st_mtime_ns", int(stt.st_mtime * 1e9))), int(stt.st_size)
    except Exception:
        return p, 0, 0

@_cache_resource_no_spinner
def _fast_joblib_load_cached(path: str, mtime_ns: int, size_bytes: int):
    # mtime/size are part of the cache key. Updated artifacts reload automatically.
    return joblib.load(path)

def _fast_joblib_load(path: str):
    p, mtime_ns, size_bytes = _artifact_signature(path)
    try:
        return _fast_joblib_load_cached(p, mtime_ns, size_bytes)
    except Exception:
        return joblib.load(p)

# ============================================================
# Bundle fallback (if artifacts.load_bundle missing/broken)
# ============================================================
@dataclass
class BundleFallback:
    model_dir: str
    priors: dict
    resolver_state: dict
    bytes_priors: dict
    bad_ips: set
    bad_domains: set
    top_domains: set
    feature_weights: dict
    feature_columns: list
    scaler: Any

    def load_supervised(self, ui_name: str):
        mdl = None
        cal = None
        thr = None

        candidates_model = [f"{ui_name}_model.pkl", f"{ui_name}.pkl", f"{ui_name}_classifier.pkl"]
        candidates_cal = [f"{ui_name}_calibrator.pkl", f"{ui_name}_calibrated.pkl", f"{ui_name}_calibrated_model.pkl"]
        candidates_thr = [f"optimal_threshold_{ui_name}.pkl", f"threshold_{ui_name}.pkl"]

        for fn in candidates_model:
            p = os.path.join(self.model_dir, fn)
            if os.path.exists(p):
                try:
                    mdl = joblib.load(p)
                    break
                except Exception:
                    pass

        for fn in candidates_cal:
            p = os.path.join(self.model_dir, fn)
            if os.path.exists(p):
                try:
                    cal = joblib.load(p)
                    break
                except Exception:
                    pass

        for fn in candidates_thr:
            p = os.path.join(self.model_dir, fn)
            if os.path.exists(p):
                try:
                    thr = float(joblib.load(p))
                    break
                except Exception:
                    pass

        return mdl, cal, thr

def _fallback_load_bundle(model_dir: str) -> BundleFallback:
    def _load_one(names: List[str], default):
        for nm in names:
            p = os.path.join(model_dir, nm)
            if os.path.exists(p):
                try:
                    return _fast_joblib_load(p)
                except Exception:
                    try:
                        import pickle
                        with open(p, "rb") as f:
                            return pickle.load(f)
                    except Exception:
                        continue
        return default

    priors = _load_one(["priors.pkl", "priors.joblib", "priors.pkl.gz"], {})
    resolver_state = _load_one(["resolver_state.pkl", "resolver.pkl"], {})
    bytes_priors = _load_one([
        "bytes_priors.pkl", "byte_priors.pkl", "byte_priors_hier_baseline.pkl",
        "bytes_priors_hier_baseline.pkl", "baseline_bytes_priors.pkl", "bytes_stats.pkl"
    ], {})
    bad_ips = _load_one(["bad_ips.pkl", "bad_ips_set.pkl"], set())
    if not isinstance(bad_ips, set):
        try:
            bad_ips = set(bad_ips)
        except Exception:
            bad_ips = set()

    bad_domains = _load_one(["bad_domains.pkl", "known_bad_domains.pkl", "malicious_domains.pkl", "ioc_bad_domains.pkl"], set())
    if not isinstance(bad_domains, set):
        try:
            bad_domains = set(bad_domains)
        except Exception:
            bad_domains = set()

    top_domains = _load_one([
        "top_domains.pkl", "top_domain_set.pkl", "top_1m_domains.pkl",
        "whitelist_domains.pkl", "trusted_domains.pkl", "benign_domains.pkl", "common_domains.pkl"
    ], set())
    if not isinstance(top_domains, set):
        try:
            top_domains = set(top_domains)
        except Exception:
            top_domains = set()

    feature_weights = _load_one(["feature_weights.pkl"], {})
    feature_columns = _load_one([
        "extended_feature_columns.pkl", "model_feature_columns.pkl", "model_features.pkl", "feature_names.pkl", "feature_columns.pkl"
    ], [])
    if not isinstance(feature_columns, list):
        feature_columns = list(feature_columns) if feature_columns is not None else []

    scaler = _load_one(["scaler.pkl"], None)
    expected_n = getattr(scaler, "n_features_in_", None) if scaler is not None else None
    try:
        expected_n = int(expected_n) if expected_n is not None else None
    except Exception:
        expected_n = None
    try:
        feature_columns = clean_model_feature_columns(feature_columns, expected_n=expected_n)
    except Exception:
        pass

    return BundleFallback(
        model_dir=model_dir,
        priors=priors if isinstance(priors, dict) else {},
        resolver_state=resolver_state if isinstance(resolver_state, dict) else {},
        bytes_priors=bytes_priors if isinstance(bytes_priors, dict) else {},
        bad_ips=bad_ips,
        bad_domains=bad_domains,
        top_domains=top_domains,
        feature_weights=feature_weights if isinstance(feature_weights, dict) else {},
        feature_columns=feature_columns,
        scaler=scaler,
    )

# ============================================================
# Resolve artifacts API + safe fallbacks
# ============================================================
load_bundle = _try_get("load_bundle", None)
canonicalize_columns = _try_get("canonicalize_columns", None)
load_feature_engineering = _try_get("load_feature_engineering", None)
compute_primary_flags = _try_get("compute_primary_flags", None)
prepare_model_matrix = _try_get("prepare_model_matrix", None)
predict_proba_with_optional_calibrator = _try_get("predict_proba_with_optional_calibrator", None)
ShapEngine = _try_get("ShapEngine", None)

if canonicalize_columns is None:
    def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ren = {
            "src_ip": "client_ip", "source_ip": "client_ip", "srcclient": "client_ip", "src_client": "client_ip",
            "clientip": "client_ip", "c_ip": "client_ip",
            "dst_ip": "dest_ip", "destip": "dest_ip", "destination_ip": "dest_ip", "dst": "dest_ip",
            "ua": "user_agent", "useragent": "user_agent",
            "host": "domain", "hostname": "domain",
            "url": "full_url", "uri": "url_path", "path": "url_path",
            "referer": "referrer", "ref": "referrer",
            "proc": "process", "cmd": "command",
            "time": "timestamp", "datetime": "timestamp", "ts": "timestamp",
        }
        cols_lower = {c: str(c).strip().lower() for c in df.columns}
        mapped = {c: ren[low] for c, low in cols_lower.items() if low in ren}
        if mapped:
            df = df.rename(columns=mapped)
        return df

if prepare_model_matrix is None:
    def prepare_model_matrix(X: pd.DataFrame, scaler, feature_cols: list):
        Xdf = X.copy()
        expected_n = getattr(scaler, "n_features_in_", None) if scaler is not None else None
        try:
            expected_n = int(expected_n) if expected_n is not None else None
        except Exception:
            expected_n = None
        feature_cols = clean_model_feature_columns(feature_cols, expected_n=expected_n)
        if feature_cols:
            for c in feature_cols:
                if c not in Xdf.columns:
                    alt = c.replace(" ", "_")
                    if alt in Xdf.columns:
                        Xdf[c] = Xdf[alt]
                    else:
                        Xdf[c] = 0
            Xdf = Xdf[feature_cols].copy()
        for c in Xdf.columns:
            if not pd.api.types.is_numeric_dtype(Xdf[c]):
                Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
        Xdf = Xdf.fillna(0.0)
        if scaler is not None and hasattr(scaler, "transform"):
            try:
                Xs = scaler.transform(Xdf.to_numpy())
                return np.asarray(Xs), list(Xdf.columns), Xdf
            except Exception:
                pass
        return Xdf.to_numpy(dtype=float), list(Xdf.columns), Xdf

if predict_proba_with_optional_calibrator is None:
    def predict_proba_with_optional_calibrator(model, calibrator, X_scaled, use_calibrator: bool = True):
        m = calibrator if (use_calibrator and calibrator is not None) else model
        if m is None:
            return np.zeros(len(X_scaled), dtype=float)
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X_scaled)[:, 1]
        if hasattr(m, "decision_function"):
            s = m.decision_function(X_scaled)
            return 1.0 / (1.0 + np.exp(-np.clip(s, -50, 50)))
        return m.predict(X_scaled).astype(float)

# ============================================================
# Optional ONNX Runtime acceleration
# ------------------------------------------------------------
# This fast path is deliberately conservative:
#   1) Use an existing .onnx artifact when present.
#   2) Otherwise auto-convert supported sklearn estimators to a disk cache.
#   3) Run a parity guard against Python predict_proba on a sample.
#   4) Fall back to the original Python model if ONNX is missing/mismatched.
# ============================================================
ONNX_IMPORT_ERR = ""
try:
    import onnxruntime as ort  # type: ignore
except Exception as e:
    ort = None  # type: ignore
    ONNX_IMPORT_ERR = str(e)

SKL2ONNX_IMPORT_ERR = ""
try:
    from skl2onnx import convert_sklearn  # type: ignore
    from skl2onnx.common.data_types import FloatTensorType  # type: ignore
except Exception as e:
    convert_sklearn = None  # type: ignore
    FloatTensorType = None  # type: ignore
    SKL2ONNX_IMPORT_ERR = str(e)

ONNX_VALIDATION_CACHE: Dict[str, bool] = {}
ONNX_VALIDATION_NOTES: Dict[str, str] = {}
ONNX_CONVERSION_FAILURE_CACHE: set = set()
LAST_INFERENCE_SOURCE = "Python"

_MODEL_ALIAS = {
    "Decision Tree": ["Decision Tree", "DecisionTree", "decision_tree", "dt"],
    "Random Forest": ["Random Forest", "RandomForest", "random_forest", "rf"],
    "Logistic Regression": ["Logistic Regression", "LogisticRegression", "logistic_regression", "lr"],
    "XGBoost": ["XGBoost", "xgboost", "xgb"],
    "LightGBM": ["LightGBM", "lightgbm", "lgbm", "lgb"],
    "CatBoost": ["CatBoost", "catboost", "cb"],
    "MoE_meta": ["MoE_meta", "MoE_meta_model", "moe_meta", "moe_meta_model"],
}

def _onnx_safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()).strip("_")
    return s or "model"

def _onnx_name_candidates(ui_name: str, role: str = "model") -> List[str]:
    names: List[str] = []
    for base in _MODEL_ALIAS.get(str(ui_name), [str(ui_name)]):
        safe = _onnx_safe_name(base)
        role_suffixes = []
        if role in {"calibrator", "calibrated"}:
            role_suffixes = ["calibrator", "calibrated", "calibrated_model"]
        elif role in {"meta", "moe_meta"}:
            role_suffixes = ["meta", "meta_model", "model"]
        else:
            role_suffixes = ["model", "classifier", ""]
        for suf in role_suffixes:
            if suf:
                names.extend([f"{base}_{suf}.onnx", f"{safe}_{suf}.onnx"])
            else:
                names.extend([f"{base}.onnx", f"{safe}.onnx"])
    # MoE historical exact name
    if str(ui_name) == "MoE_meta":
        names = ["MoE_meta_model.onnx", "MoE_meta.onnx", "moe_meta_model.onnx"] + names
    # Stable de-dupe
    out: List[str] = []
    seen = set()
    for x in names:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

def _find_onnx_file(ui_name: str, role: str = "model") -> Optional[str]:
    try:
        md = os.path.abspath(str(globals().get("model_dir", "") or ""))
    except Exception:
        md = ""
    if not md:
        return None
    search_dirs = [md, os.path.join(md, "onnx"), os.path.join(md, "_onnx_cache")]
    for d in search_dirs:
        for fn in _onnx_name_candidates(ui_name, role=role):
            path = os.path.join(d, fn)
            if os.path.exists(path) and os.path.isfile(path):
                return path
    return None

@_cache_resource_no_spinner
def _load_onnx_session_cached(path: str, mtime_ns: int, size_bytes: int, providers_key: Tuple[str, ...]):
    if ort is None:
        return None
    sess_opts = ort.SessionOptions()
    try:
        sess_opts.intra_op_num_threads = int(os.getenv("ONNX_INTRA_OP_THREADS", "1"))
        sess_opts.inter_op_num_threads = int(os.getenv("ONNX_INTER_OP_THREADS", "1"))
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass
    providers = list(providers_key) if providers_key else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=sess_opts, providers=providers)

def _load_onnx_session(path: str):
    if ort is None or not path or not os.path.exists(path):
        return None
    p, mtime_ns, size_bytes = _artifact_signature(path)
    providers = tuple(os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider").split(","))
    providers = tuple([x.strip() for x in providers if x.strip()]) or ("CPUExecutionProvider",)
    try:
        return _load_onnx_session_cached(p, mtime_ns, size_bytes, providers)
    except Exception as e:
        ONNX_VALIDATION_NOTES[p] = f"ONNX session load failed: {e}"
        return None

def _onnx_cache_path(ui_name: str, role: str, n_features: int) -> str:
    md = os.path.abspath(str(globals().get("model_dir", ".") or "."))
    cache_dir = os.path.join(md, "_onnx_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{_onnx_safe_name(ui_name)}_{_onnx_safe_name(role)}_{int(n_features)}f.onnx")

def _maybe_convert_sklearn_to_onnx(ui_name: str, model_obj: Any, n_features: int, role: str = "model") -> Optional[str]:
    if not bool(globals().get("ENABLE_ONNX_AUTO_CONVERT", True)):
        return None
    if convert_sklearn is None or FloatTensorType is None or model_obj is None:
        return None
    # CalibratedClassifierCV and many third-party boosters require extra converters.
    # Try safely, never fail the app.
    path = _onnx_cache_path(ui_name, role, n_features)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    fail_key = (str(ui_name), str(role), int(n_features), type(model_obj).__module__, type(model_obj).__name__)
    if fail_key in ONNX_CONVERSION_FAILURE_CACHE:
        return None
    tmp_path = path + ".tmp"
    try:
        initial_type = [("float_input", FloatTensorType([None, int(n_features)]))]
        onx = convert_sklearn(model_obj, initial_types=initial_type, target_opset=int(os.getenv("ONNX_TARGET_OPSET", "17")))
        with open(tmp_path, "wb") as f:
            f.write(onx.SerializeToString())
        os.replace(tmp_path, path)
        return path
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        ONNX_CONVERSION_FAILURE_CACHE.add(fail_key)
        ONNX_VALIDATION_NOTES[path] = f"Auto-convert skipped/failed: {e}"
        return None

def _prob_from_onnx_outputs(outputs: List[Any], output_names: List[str], n_rows: int) -> Optional[np.ndarray]:
    candidates: List[Tuple[int, np.ndarray]] = []
    for i, out in enumerate(outputs):
        name = output_names[i].lower() if i < len(output_names) else ""
        if "label" in name and "prob" not in name and "score" not in name:
            continue
        arr = None
        # ZipMap probability output: list[dict[class_label -> prob]]
        if isinstance(out, list) and out and isinstance(out[0], dict):
            vals = []
            for d in out:
                if 1 in d:
                    vals.append(d.get(1, 0.0))
                elif "1" in d:
                    vals.append(d.get("1", 0.0))
                elif True in d:
                    vals.append(d.get(True, 0.0))
                else:
                    try:
                        # Prefer largest non-zero class label if labels are strings/ints.
                        keys = sorted(d.keys(), key=lambda x: str(x))
                        vals.append(d[keys[-1]] if keys else 0.0)
                    except Exception:
                        vals.append(0.0)
            arr = np.asarray(vals, dtype=float)
        else:
            try:
                arr = np.asarray(out, dtype=float)
            except Exception:
                arr = None
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[0] == n_rows and arr.shape[1] >= 2:
            score = 100 if ("prob" in name or "score" in name) else 50
            candidates.append((score, arr[:, 1].astype(float)))
        elif arr.ndim == 2 and arr.shape[0] == n_rows and arr.shape[1] == 1:
            v = arr[:, 0].astype(float)
            if np.nanmin(v) >= -1e-9 and np.nanmax(v) <= 1 + 1e-9:
                score = 80 if ("prob" in name or "score" in name) else 10
                candidates.append((score, v))
        elif arr.ndim == 1 and arr.shape[0] == n_rows:
            v = arr.astype(float)
            if np.nanmin(v) >= -1e-9 and np.nanmax(v) <= 1 + 1e-9:
                # Avoid treating a hard label output as probability unless the name says so.
                if ("prob" in name or "score" in name) or not np.all(np.isin(np.unique(v[~np.isnan(v)]), [0.0, 1.0])):
                    score = 70 if ("prob" in name or "score" in name) else 5
                    candidates.append((score, v))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    prob = np.asarray(candidates[0][1], dtype=float).reshape(-1)
    if len(prob) != n_rows:
        return None
    return np.clip(prob, 0.0, 1.0)

def _predict_proba_onnx_path(path: str, X_scaled: Any) -> Optional[np.ndarray]:
    sess = _load_onnx_session(path)
    if sess is None:
        return None
    try:
        input_meta = sess.get_inputs()[0]
        input_name = input_meta.name
        X_arr = np.asarray(X_scaled, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        outs_meta = sess.get_outputs()
        output_names = [o.name for o in outs_meta]
        outputs = sess.run(None, {input_name: X_arr})
        return _prob_from_onnx_outputs(outputs, output_names, X_arr.shape[0])
    except Exception as e:
        ONNX_VALIDATION_NOTES[os.path.abspath(path)] = f"ONNX inference failed: {e}"
        return None

def _python_predict_for_parity(model_obj: Any, X_sample: Any) -> Optional[np.ndarray]:
    try:
        p = predict_proba_with_optional_calibrator(model_obj, None, X_sample, use_calibrator=False)
        return np.asarray(p, dtype=float).reshape(-1)
    except Exception:
        return None

def predict_proba_with_onnx_acceleration(
    model: Any,
    calibrator: Any,
    X_scaled: Any,
    use_calibrator: bool = True,
    model_name: str = "model",
    role_hint: str = "model",
) -> np.ndarray:
    """Drop-in probability scorer with optional ONNX Runtime acceleration.

    It preserves model quality by using ONNX only when the probability output
    matches Python predict_proba within a strict sample tolerance.  If a
    calibrator is enabled, ONNX is used only for a calibrated ONNX artifact or a
    successfully converted calibrator; otherwise the Python calibrated path is
    retained.
    """
    global LAST_INFERENCE_SOURCE
    X_arr = np.asarray(X_scaled)
    n_rows = len(X_arr)
    if n_rows == 0:
        LAST_INFERENCE_SOURCE = "empty"
        return np.zeros(0, dtype=float)

    enable = bool(globals().get("ENABLE_ONNX_INFERENCE", False)) and (ort is not None)
    parity = bool(globals().get("ENABLE_ONNX_PARITY_GUARD", True))
    atol = float(globals().get("ONNX_PARITY_ATOL", 1e-5) or 1e-5)
    max_check = int(globals().get("ONNX_PARITY_SAMPLE_ROWS", 512) or 512)

    target_obj = calibrator if (use_calibrator and calibrator is not None) else model
    target_role = "calibrator" if (use_calibrator and calibrator is not None) else role_hint

    if enable and target_obj is not None:
        path = _find_onnx_file(model_name, role=target_role)
        if path is None:
            path = _maybe_convert_sklearn_to_onnx(model_name, target_obj, int(X_arr.shape[1]), role=target_role)
        if path:
            path_abs = os.path.abspath(path)
            is_valid = ONNX_VALIDATION_CACHE.get(path_abs, None)
            if is_valid is not False:
                p_onnx = _predict_proba_onnx_path(path_abs, X_arr)
                if p_onnx is not None:
                    if parity and is_valid is None:
                        k = min(max_check, n_rows)
                        p_py = _python_predict_for_parity(target_obj, X_arr[:k])
                        p_check = p_onnx[:k]
                        if p_py is not None and len(p_py) == len(p_check):
                            diff = float(np.nanmax(np.abs(np.asarray(p_py, dtype=float) - np.asarray(p_check, dtype=float)))) if k else 0.0
                            if diff <= atol:
                                ONNX_VALIDATION_CACHE[path_abs] = True
                                ONNX_VALIDATION_NOTES[path_abs] = f"parity_ok max_abs_diff={diff:.3g}"
                            else:
                                ONNX_VALIDATION_CACHE[path_abs] = False
                                ONNX_VALIDATION_NOTES[path_abs] = f"parity_failed max_abs_diff={diff:.3g} > {atol:g}; Python fallback"
                                p_onnx = None
                        else:
                            # If Python parity is unavailable for this object, keep ONNX only when user disabled parity.
                            ONNX_VALIDATION_CACHE[path_abs] = False
                            ONNX_VALIDATION_NOTES[path_abs] = "parity_unavailable; Python fallback"
                            p_onnx = None
                    if p_onnx is not None:
                        LAST_INFERENCE_SOURCE = f"ONNX Runtime ({os.path.basename(path_abs)})"
                        return np.asarray(p_onnx, dtype=float)

    LAST_INFERENCE_SOURCE = "Python predict_proba"
    return np.asarray(predict_proba_with_optional_calibrator(model, calibrator, X_arr, use_calibrator=use_calibrator), dtype=float)

def summarize_inference_sources(srcs: List[str]) -> str:
    if not srcs:
        return ""
    from collections import Counter
    c = Counter([str(x) for x in srcs if str(x).strip()])
    return "; ".join(f"{k} x{v}" for k, v in c.most_common(6))

# ============================================================
# Imports from modular files
# ============================================================
try:
    from log_parser import parse_log_line_universal as _parse_line  # type: ignore
except Exception:
    from log_parser import parse_log_universal as _parse_line  # type: ignore

import imputer as imputer_mod  # module for patching
from imputer import ForensicImputer, PRIMARY_COLS  # type: ignore
from ipv6_primary_conditions import add_ipv6_primary_conditions  # type: ignore
try:
    from override_rules import compute_override_signals, OVERRIDE_REASON_LEGEND  # type: ignore
except Exception:
    compute_override_signals = None  # type: ignore
    OVERRIDE_REASON_LEGEND = None  # type: ignore

# ============================================================
# Patch imputer.sanitize_domain_series to kill "~ float" crash
# ============================================================
_IPV4_RE_SAFE = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")

def _sanitize_domain_series_safe(host: pd.Series) -> pd.Series:
    if host is None:
        return pd.Series(dtype=object)

    s = host.copy().astype("string")
    s = s.str.strip().str.lower()
    s = s.str.strip("[](){}<>\"' ")
    s = s.str.rstrip(".,;)]}\"'")

    s = s.str.replace(r"^https?://", "", regex=True)
    s = s.str.split("/", n=1).str[0].str.split("?", n=1).str[0].str.split("#", n=1).str[0]
    s = s.str.split("@", n=1).str[-1]
    s = s.str.split(":", n=1).str[0]

    miss = {"", "-", "--", "none", "null", "nan", "na", "n/a", "unknown", "notprovided", "not_provided", "unknown_domain", "unknown-domain"}
    miss_mask = (s.isna() | s.isin(list(miss))).fillna(True)
    s = s.mask(miss_mask, pd.NA)

    s = s.mask(s.str.match(r"^\d+(?:\.\d+)*$", na=False), pd.NA)
    s = s.mask(s.str.match(_IPV4_RE_SAFE, na=False), pd.NA)

    internal_ok = (".local", ".lan", ".internal", ".intra", ".corp", ".corp.local")
    has_dot = s.str.contains(".", regex=False, na=False).astype(bool)
    is_internal = s.str.endswith(internal_ok, na=False).astype(bool)
    ok = (has_dot | is_internal).astype(bool)

    keep = (~pd.isna(s)).to_numpy(dtype=bool)
    bad = keep & np.logical_not(ok.to_numpy(dtype=bool))
    if bad.any():
        s = s.mask(pd.Series(bad, index=s.index), pd.NA)

    out = s.astype(object)
    out = out.where(pd.notna(out), np.nan)
    return out

imputer_mod.sanitize_domain_series = _sanitize_domain_series_safe

# ============================================================
# Primary override keywords (explicit/critical/moderate)
# ============================================================

# ------------------------------------------------------------
# 0) Helpers
# ------------------------------------------------------------
def _alt(items):
    """Safe alternation for literals."""
    return "|".join(re.escape(x) for x in items if x)

# ------------------------------------------------------------
# 1) Strict “explicitly benign” gate (avoid killing firewall ALLOW)
#    Only blocks when the text explicitly asserts non-maliciousness,
#    not when a device/action says ALLOW.
# ------------------------------------------------------------
_NONMAL_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:"
    r"non[-_ ]?malicious|not[-_ ]?malicious|"
    r"(?:verdict|label|classification|is_malicious|malicious|threat|detection)\s*[:=]\s*(?:"
    r"false|0|no|none|benign|clean|legit|legitimate|safe|undetected|not_detected"
    r")|"
    r"false\s*positive|\bfp\b\s*[:=]\s*true"
    r")(?:[^a-z0-9]|$)",
    flags=re.IGNORECASE,
)

# ------------------------------------------------------------
# 2) Primary override keywords (EXPLICIT)
#    These are *high-confidence* signals: if present => ALWAYS malicious.
#    Keep this list precise: avoid generic words like "bot" or "download" here.
# ------------------------------------------------------------
_EXPLICIT_TERMS = [
    # Vuln scanners / web attack tooling
    r"sqlmap", r"nikto", r"acunetix", r"netsparker", r"arachni", r"wpscan",
    r"masscan", r"nmap", r"zmap", r"zgrab", r"naabu", r"nuclei",
    r"dirbuster", r"dirsearch", r"gobuster", r"ffuf", r"feroxbuster", r"wfuzz", r"dirb\b",
    r"burp(?:suite)?", r"zap(?:roxy)?",
    r"nessus", r"openvas", r"qualys", r"rapid7",

    # Brute-force / credential / lateral tooling
    r"hydra", r"medusa", r"ncrack", r"patator", r"kerbrute", r"crowbar",
    r"mimikatz", r"rubeus", r"secretsdump", r"impacket",
    r"crackmapexec|\bcme\b", r"evil-winrm", r"wmiexec", r"smbexec", r"atexec", r"dcomexec", r"psexec",

    # Post-ex / C2 frameworks (names are strong signals in logs/UAs)
    r"metasploit", r"meterpreter", r"cobalt\s*strike", r"\bbeacon\b",
    r"sliver", r"empire", r"covenant", r"mythic",

    # LOLBAS / script engines (kept because these appear in your corp datasets as explicit attack strings)
    r"powershell", r"pwsh", r"cmd\.exe", r"wmic", r"mshta", r"cscript", r"wscript",
    r"regsvr32", r"rundll32", r"certutil", r"bitsadmin", r"installutil",
    r"msbuild", r"schtasks", r"wevtutil",

    # Malware families / loaders / botnets (commonly used as UA/labels)
    r"zeus", r"emotet", r"trickbot", r"dridex", r"qakbot|qbot",
    r"agenttesla", r"formbook", r"azorult",
    r"raccoon(?:\s*stealer)?", r"lumm?a\s*stealer", r"redline", r"vidar",
    r"njrat", r"asyncrat", r"remcos", r"nanocore", r"warzone(?:\s*rat)?",
    r"darkcomet", r"gh0st",
    r"mirai", r"mozi", r"xmrig", r"coinhive",

    # Generic but high-signal malware terms
    r"ransomware", r"rootkit", r"keylogger", r"botnet", r"cryptojack",

    # Piracy / mass-spam kits (kept for your threat model; move to moderate if too noisy)
    r"warez", r"keygen", r"\bcrack\b", r"carding", r"ccdump",
    r"massmailer", r"bulkmailer", r"emailharvest", r"emailcollector",
]

explicit_regex_pattern = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(_EXPLICIT_TERMS) +
    r")(?![a-z0-9])",
    flags=re.IGNORECASE,
)

# ------------------------------------------------------------
# 3) Observed high-confidence IOCs (from your 35-type corpus)
#    Kept small + explicit; disable if you want purely-generic detection.
# ------------------------------------------------------------
INCLUDE_OBSERVED_IOCS = True

_OBS_IOC_DOMAINS = [
    # From your incident bundle / hybrid patterns:
    "e-files.download",
    "hkdust.github.wiki",
    "cdn-update-check.net",
    "itamaraty-gov.com",
    "7xq2k9d1k3.biz",
]
_OBS_IOC_IPS = [
    # From your incident bundle / type9 attackers + C2 infra in examples:
    "185.112.83.116",
    "61.177.56.27",
    "94.190.43.52",
    "208.81.37.55",
    "134.122.188.249",
    "185.227.70.204",
    "43.131.69.98",
    "77.73.133.73",
]
_OBS_DOM_ALT = _alt(_OBS_IOC_DOMAINS)
_OBS_IP_ALT  = _alt(_OBS_IOC_IPS)

_OBS_IOC_DOM_RE = rf"(?:{_OBS_DOM_ALT})" if _OBS_DOM_ALT else r"(?!x)x"
_OBS_IOC_IP_RE  = rf"(?:{_OBS_IP_ALT})" if _OBS_IP_ALT else r"(?!x)x"

# ------------------------------------------------------------
# 4) High-risk / cheap attacker TLDs (+ your cases)
#    (Keep this list conservative; too broad => false positives.)
# ------------------------------------------------------------
_TLDS = [
    "link","top","xyz","icu","pw","tk","club",
    "live","click","buzz","gq","work","fit","lol","cyou",
    "tech","download","zip","mov","cam","cfd","sbs","rest",
    "online","site","space","monster","house","men",
    "ga","cf","ml",  # free/cheap TLDs frequently abused
]
_TLD_ALT = _alt(_TLDS)

# ------------------------------------------------------------
# 5) “malicious” token safe-guard
#    (won’t fire on nonmalicious / not malicious / notmalicious)
# ------------------------------------------------------------
_MAL_TOKEN_SAFE = (
    r"(?:(?<!not)(?<!not\ )(?<!not-)(?<!not_)(?<!not:)(?<!not\t)"
    r"(?<!non)(?<!non\ )(?<!non-)(?<!non_)(?<!non:)(?<!non\t)"
    r"malicious)"
)

# ------------------------------------------------------------
# 6) Phishing / Login Theft (SSO + login-verification + drive+brand lures)
# ------------------------------------------------------------
_SSO_CORE   = r"(?:sso|auth|login|signin|idp|oauth|saml|mfa|2fa|okta|azuread|microsoftonline|office365|o365|owa)"
_SSO_ACT    = r"(?:verify|verification|validate|confirm|update|security|support|account|session|password|reset|unlock|portal|continue)"
_SSO_BRAND  = r"(?:microsoft|office|outlook|onedrive|github|yahoo|google|gmail|paypal|amazon|aws|bank|irs|tax|gov)"
_SSO_EXPL   = rf"(?:my[-_.]?sso)\.(?:{_TLD_ALT})"
_LOGIN_VER  = rf"(?:login|signin|auth|sso)[-_.](?:verify|verification|validate|confirm)(?:[-_.][a-z0-9-]{{0,20}})*\.(?:{_TLD_ALT})"
_SSO_LURE   = rf"(?:{_SSO_CORE})(?:[-_.][a-z0-9-]{{0,30}})*[-_.](?:{_SSO_ACT})(?:[-_.][a-z0-9-]{{0,30}})*\.(?:{_TLD_ALT})"
_SSO_BR_LUR = rf"(?:{_SSO_BRAND})(?:[-_.][a-z0-9-]{{0,30}})*[-_.](?:{_SSO_CORE}|{_SSO_ACT})(?:[-_.][a-z0-9-]{{0,30}})*\.(?:{_TLD_ALT})"
_DRIVE_BR   = rf"(?:mydrive|drive|gdrive|onedrive|icloud|sharepoint)(?:[-_.][a-z0-9-]{{0,20}})*[-_.](?:outlook|microsoft|google|github|yahoo)\.(?:{_TLD_ALT})"

# ------------------------------------------------------------
# 7) Malware payload hosting (fake “download/installer” drive/brand domains)
# ------------------------------------------------------------
_PAYLOAD_W  = r"(?:installer|setup|update|patch|payload|dropper|stager|downloader|exe|msi|dmg|pkg|apk)"
_FAKE_PAYL  = rf"(?:drive|mydrive|gdrive|onedrive|icloud|google|gmail|outlook|microsoft)(?:[-_.][a-z0-9-]{{0,25}})*[-_.](?:download|{_PAYLOAD_W})(?:[-_.][a-z0-9-]{{0,25}})*\.(?:{_TLD_ALT})"

# ------------------------------------------------------------
# 8) Deceptive update / scareware / push scams
# ------------------------------------------------------------
_FAKE_UPDATE = (
    r"small[-_.]?(?:updt|update|upd)|browser[-_.]?(?:update|upd)|fake[-_.]?update|"
    r"chrome[-_.]?update|windows[-_.]?update|microsoft[-_.]?update|"
    r"security[-_.]?(?:update|check)|update[-_.]?(?:service|center)|quick[-_.]?update|"
    r"flash[-_.]?update|java[-_.]?update|plugin[-_.]?update"
)
_SCARE_HOST = rf"(?:my[-_.]?security)\.(?:{_TLD_ALT})"
_SCARE_TXT  = r"(?:your\s+pc\s+is\s+infected|virus\s+alert|call\s+now|tech\s+support|support\s+number|scareware|adware)"
_PUSH_TXT   = r"(?:push|notification|notify|allow\s+notifications|enable\s+notifications)"

# ------------------------------------------------------------
# 9) Typosquat / brand-gov lures on risky TLDs
# ------------------------------------------------------------
_BRAND_GOV  = r"(?:apple|icloud|itunes|microsoft|windows|office|onedrive|outlook|google|gmail|youtube|paypal|amazon|aws|gov|irs|tax|refund|bank)"
_BR_ACT     = r"(?:verify|verification|refund|update|security|login|support|billing|payment|check)"
_BRAND_LURE = rf"(?:{_BRAND_GOV})(?:[-_.][a-z0-9-]{{0,30}})*[-_.](?:{_BR_ACT})(?:[-_.][a-z0-9-]{{0,30}})*\.(?:{_TLD_ALT})"

# ------------------------------------------------------------
# 10) File hosting mimicry on sketchy TLDs
# ------------------------------------------------------------
_FILE_MIMIC = r"(?:files|download|cdn|cache|static|content|assets|update|updates)[-_][a-z0-9-]{2,63}"
_FILE_HOST  = rf"(?:{_FILE_MIMIC})\.(?:{_TLD_ALT})"
_FILELINK   = r"(?:files|usercontent)[a-z0-9-]*(?:\.[a-z0-9-]{1,63})*\.link"

# ------------------------------------------------------------
# 11) Disposable hosting abuse
# ------------------------------------------------------------
_DISPOSABLE = r"(?:[a-z0-9-]{1,63}\.)*(?:glitch\.me|github\.io|pages\.dev|workers\.dev|vercel\.app|netlify\.app|web\.app|firebaseapp\.com|replit\.app|onrender\.com|herokuapp\.com|surge\.sh|appspot\.com|azurewebsites\.net)"
_GLITCH_PH  = r"(?:[a-z0-9-]*?(?:push|notify|notification|login|verify|verification|update|security|account|signin|auth)[a-z0-9-]*\.)+glitch\.me"

# ------------------------------------------------------------
# 12) Tunneling / Shadow IT
# ------------------------------------------------------------
_TUNNELS = r"(?:[a-z0-9-]{1,63}\.)*(?:ngrok\.io|ngrok-free\.app|ngrok\.app|ngrok\.dev|localhost\.run|localtunnel\.me|serveo\.net|trycloudflare\.com|pagekite\.me|tunnel\.to)"

# ------------------------------------------------------------
# 13) C2 / Exfiltration / Heartbeat / Redirectors
# ------------------------------------------------------------
_LONGMUSIC  = r"(?:[a-z0-9-]{1,63}\.)*longmusic\.com"
_C2_WORDS   = r"(?:heartbeat|beacon(?:ing)?|check[-_ ]?in|keepalive|callback|c2|cobalt\s*strike|sliver|empire|covenant|mythic|meterpreter|redirector|exfil|ransomware|dns[-_ ]?exfil|dns[-_ ]?tunnel|dnscat2|iodine|dns2tcp|xmrig)"

# ------------------------------------------------------------
# 14) “Lucky / pwn / shell / cmd / vault” tokens
# ------------------------------------------------------------
_LUCKY = r"(?:l4cky|lucky|pwn|shell|cmd|vault|rootkit)"

# ------------------------------------------------------------
# 15) Recon / vuln mapping / mass scanning tooling (UA/text)
# ------------------------------------------------------------
_VULN_ENUM = r"(?:httpx|nuclei|naabu|amass|subfinder|assetfinder|gau|waybackurls|hakrawler|whatweb|wappalyzer|dirsearch|ffuf|feroxbuster|shodan|censys|jbrofuzz|pacu)"

# ------------------------------------------------------------
# 16) Offline downloader / mirroring tools (UA/text)
# ------------------------------------------------------------
_LEECH = r"(?:httrack|winhttrack|teleport|webcopy|offline\s+explorer|sitesucker|aria2c?|axel|lftp|leechget|smartdownload|realdownload|mass\s+downloader|webzip|websucker|webcopier|pavuk|extreme\s+picture\s+finder|cyotekwebcopy)"

# ------------------------------------------------------------
# 17) “spy-named spider” + odd named crawlers
# ------------------------------------------------------------
_SPY_SPIDER = r"(?:james[-_ ]?bond|007)[-_ ]?(?:spider|crawler|bot)|(?:snoopy|netspider)"

# ------------------------------------------------------------
# 18) Scanners/bots/automation UAs
# ------------------------------------------------------------
_UA_SCANNERS = r"(?:fyrebot|zmeu|morfeus|masscan|zgrab|zmap|nmap|nikto|sqlmap|gobuster|dirbuster|acunetix|netsparker|wpscan|openvas|nessus)"
_UA_HEADLESS = r"(?:headlesschrome|phantomjs|selenium|puppeteer|playwright)"
_UA_SCRIPT   = r"(?:python-requests/|go-http-client/|aws-sdk-go/|aiohttp/|okhttp/|java/|libwww-perl|libweb|curl/|wget/|getright|markmonitor|mj12bot|majestic(?:\s*seo)?|axios/|postmanruntime/)"
_UA_WEIRD    = r"(?:tinytestbot|testbot|tinybot|spider[_-]?bot|extractor|whacker|bruteforce|payload|clever\s+internet\s+suite|msiecrawler|femtosearchbot|urly\\?\s*warning)"

# ------------------------------------------------------------
# 19) DGA-ish heuristic
# ------------------------------------------------------------
_DGA_EXCL = (
    r"(?:www|mail|smtp|imap|pop|api|cdn|static|assets|img|images|"
    r"auth|login|sso|idp|oauth|saml|mfa|secure|portal|account|"
    r"dev|stage|staging|prod|test|uat|beta|demo|internal|corp|local)"
)
_DGA_LAB  = r"(?=[a-z0-9]{6,24}\.)(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]{6,24}"
_DGA_HOST = rf"(?!{_DGA_EXCL}\.)(?:{_DGA_LAB})\.(?:[a-z0-9-]{{2,63}}\.)+(?:[a-z]{{2,24}})"
_DGA_SLD_BIZ = rf"(?!{_DGA_EXCL}\.)(?:{_DGA_LAB})\.biz"

# ------------------------------------------------------------
# 20) Obfuscated URL + punycode
# ------------------------------------------------------------
_HXXP     = r"(?:hxxps?\s*(?:\[\s*:\s*\]|:)\s*//)"
_PUNYCODE = r"(?:xn--[a-z0-9-]{4,}(?:\.[a-z0-9-]{2,63})+)"

# ------------------------------------------------------------
# 21) High-risk TLD host (generic; supports leading *.)
# ------------------------------------------------------------
_RISK_TLD_HOST = rf"(?:\*\.)?(?:[a-z0-9-]{{1,63}}\.)+(?:{_TLD_ALT})"

# ============================================================
# 35-log-family explicit “high confidence” indicators
# ============================================================

# -- Web/API exploitation footprints (paths + method abuse + SQLi/RCE tokens)
_HTTP_ATTACK_PATHS  = r"(?:/\.(?:env|git|svn)\b|/wp-admin/?\b|/wp-login\.php\b|/xmlrpc\.php\b|/phpmyadmin/?\b|/cgi-bin/?\b|/server-status\b)"
_HTTP_METHOD_ABUSE  = r"(?:\b(?:PUT|DELETE|PROPFIND|MKCOL|MOVE|COPY|PATCH)\b\s+/(?:login|index\.html|wp-admin/|api/(?:upload|import)|shell)\b)"
_HTTP_BOT_METHOD_ABUSE = r"(?:\b(?:PUT|DELETE|PROPFIND|MKCOL|MOVE|COPY|PATCH)\b[^\n\r]{0,220}\b(?:bot|crawler(?:s)?|spider(?:s)?|robot(?:s)?)\b)"
_ATTACK_TOKENS      = r"(?:\b(?:sqli|sql\s*injection|rce|xss|ssti|lfi|rfi|cmdi|path[_-]?traversal)\b|'\s*or\s*1\s*=\s*1\s*--|\bunion\s+select\b)"

# -- LOLBAS / endpoint abuse (very specific forms in your samples + common high-signal additions)
_LOLBAS_ABUSE = (
    r"(?:"
    r"\bmsxsl\.exe\b.*https?://|"
    r"\bcertutil\.exe\b\s+-urlcache\s+-split\s+-f\s+https?://|"
    r"\bdesktopimgdownldr\.exe\b.*\b/lockscreenurl:\s*https?://|"
    r"\bms-appinstaller://\?source=https?://|"
    r"\bwevtutil(?:\.exe)?\b\s+cl\s+Security\b|"
    r"\blsass[_-]?dump\.exe\b|"
    r"\bprocdump(?:64)?\.exe\b.*\blsass\b|"
    r"\bpowershell(?:\.exe)?\b[^\n\r]{0,120}\s-(?:enc|encodedcommand)\b|"
    r"\b(?:curl|wget)\b[^\n\r]{0,120}\|\s*(?:sh|bash)\b|"
    r"\bcat\s+/etc/shadow\b"
    r")"
)

# -- Linux auth brute-force / intrusion-ish strings (high signal, common in attacks)
_LINUX_AUTH_ATTACK = r"(?:Failed\s+password\s+for\s+root|Too\s+many\s+authentication\s+failures|Failed\s+password\s+for\s+invalid\s+user|Disconnecting:\s+Too\s+many\s+authentication\s+failures)"

# -- K8s/container abuse signals (high-confidence)
_K8S_CONTAINER_ABUSE = r"(?:\bcluster-admin\b|\bclusterrolebindings\b|privileged(?:_container)?\b|privileged=true|Created\s+privileged\s+pod|curl\s+https?://[^\s]+\s*\|\s*(?:sh|bash)\b|\bnc\b\s+[0-9]{1,3}(?:\.[0-9]{1,3}){3}\s+4444\b)"

# -- Cloud/SaaS “cover tracks / bypass controls” (high-confidence)
_CLOUD_COVERTRACKS = r"(?:StopLogging|DisableMailboxAudit|bypass_dlp|ExportWorkspaceData)"

# -- Explicit security verdict fields (high-confidence)
_SECURITY_VERDICTS = r"(?:\bverdict=(?:malicious|phish|malware|spoof)\b|\bverdict=(?:PHISH|MALWARE|SPOOF)\b|\baction=BLOCK\b|\baction=DROP\b|\brule=(?:SQLI|RCE|PATH_TRAVERSAL)\b|\bET\s+(?:MALWARE|TROJAN)\b|\bNmap\s+Scripting\s+Engine\b|\bET\s+POLICY\s+Suspicious\s+TLS\s+SNI\b)"

# ---- Compact “critical words” ----
_CRIT_WORDS = (
    r"porn|xx|sex|erotic|escort|cocaine|heroin|weed|cannabis|marijuna|marijuana|narcotic|"
    r"terrorist|warning|harm|childporn|pedo|isis|malware|ransomware|paedo|"
    r"rape|molest|prostitut|traffick|explosive|weapon|alqaeda|neo-nazi|nazism|"
    r"phish|phishing|credential|creds|steal|stealer|"
    r"ad[-_ ]?inject|adware|injector|downloader"
)

# ============================================================
# FINAL: ONE compiled regex (boundary-aware)
# ============================================================
_CRIT_BODY = "|".join([
    _CRIT_WORDS,
    _MAL_TOKEN_SAFE,
    _SECURITY_VERDICTS,

    _SPY_SPIDER,
    _LONGMUSIC,
    _C2_WORDS,

    _SSO_EXPL,
    _LOGIN_VER,
    _SSO_LURE,
    _SSO_BR_LUR,
    _DRIVE_BR,

    _FAKE_PAYL,

    _FAKE_UPDATE,
    _SCARE_HOST,
    _SCARE_TXT,
    _PUSH_TXT,

    _BRAND_LURE,
    _FILE_HOST,
    _FILELINK,

    _DISPOSABLE,
    _GLITCH_PH,
    _TUNNELS,

    _LUCKY,

    _VULN_ENUM,
    _LEECH,

    _UA_SCANNERS,
    _UA_HEADLESS,
    _UA_SCRIPT,
    _UA_WEIRD,

    _DGA_HOST,
    _DGA_SLD_BIZ,

    _HXXP,
    _PUNYCODE,

    _RISK_TLD_HOST,

    _HTTP_ATTACK_PATHS,
    _HTTP_METHOD_ABUSE,
    _HTTP_BOT_METHOD_ABUSE,
    _ATTACK_TOKENS,
    _LOLBAS_ABUSE,
    _LINUX_AUTH_ATTACK,
    _K8S_CONTAINER_ABUSE,
    _CLOUD_COVERTRACKS,
])

# Add observed IOCs (optional)
if INCLUDE_OBSERVED_IOCS:
    _CRIT_BODY = "|".join([_CRIT_BODY, _OBS_IOC_DOM_RE, _OBS_IOC_IP_RE])

critical_pattern = re.compile(
    rf"(?<![a-z0-9])(?:{_CRIT_BODY})(?![a-z0-9])",
    flags=re.IGNORECASE,
)

def is_critical(text: str) -> bool:
    """Strict boolean gate: won't fire on explicit non-malicious assertions."""
    if not text:
        return False
    if _NONMAL_RE.search(text):
        return False
    return bool(critical_pattern.search(text))

# ------------------------------------------------------------
# OPTIONAL: quick evaluator over a pasted multi-type blob
# ------------------------------------------------------------
def eval_by_type(raw: str, max_samples: int = 5):
    """
    Parses blocks starting with 'type N:' or 'N) ...' headings and counts matches.
    Returns: dict[type_key] = {'lines':X,'critical':Y,'samples':[...]}

    NOTE: line-by-line only; correlation (e.g., 4624 + attacker IP elsewhere)
    should be handled in your feature engineering/model stage.
    """
    from collections import defaultdict  # local import => no global dependency

    cur = "unknown"
    out = defaultdict(lambda: {"lines": 0, "critical": 0, "samples": []})
    for ln in (raw or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        m = re.match(r"^(?:type\s+(\d+)\s*:|(\d+)\)\s+)", s, flags=re.IGNORECASE)
        if m:
            cur = m.group(1) or m.group(2)
            cur = f"type_{cur}"
            continue
        if s.startswith("#") or set(s) <= {"-", "=", "_"}:
            continue
        out[cur]["lines"] += 1
        if is_critical(s):
            out[cur]["critical"] += 1
            if len(out[cur]["samples"]) < max_samples:
                out[cur]["samples"].append(s[:260])
    return dict(out)

# When use it, apply the negation mask:
# hit_critical = blob.str.contains(critical_pattern, na=False) & (~blob.str.contains(_NONMAL_RE, na=False))


# ------------------------------------------------------------
# Moderate keywords (lower-confidence; used only with strong signals)
# ------------------------------------------------------------
moderate_keywords = [
    'crawler','bot','spam','phish','bruteforce','cracker','keygen','darkweb',
    'fuzz','grabber','hydra','bruter','libwww','curl','wget','python-requests','urllib','java/','perl','ruby',
    'mechanize','httpclient','http_request',
    'meth','weed',
    'adult',
    # LOLBins
    'powershell','pwsh','cmd.exe','wmic','mshta','cscript','wscript','regsvr32',
    'certutil','bitsadmin','installutil','msbuild','schtasks','rundll32','psexec',
    'wevtutil','dnscmd','esentutl','forfiles','makecab','expand'
]
moderate_pattern = re.compile(r"(?i)\b(" + "|".join(map(re.escape, moderate_keywords)) + r")\b")

# Context hints
SUSP_TZ_HARD_RE = re.compile(r"\b(vpn|tor|wireguard|openvpn|proxy|tunnel|teredo|6to4|isatap|ipsec|nat64|dns64)\b", re.IGNORECASE)
BOT_UA_RE = re.compile(r"\b(bot|crawler|spider|scrapy|selenium|headless)\b", re.IGNORECASE)
AUTO_CMD_RE = re.compile(r"\b(powershell|pwsh|cmd\.exe|curl|wget|certutil|bitsadmin|mshta|rundll32|regsvr32)\b", re.IGNORECASE)
# Time-basis forcing should be conservative: use UTC only when evidence suggests bot/automation/VPN/tunnel.
TZ_FORCE_UTC_RE = re.compile(r"\b(vpn|tor|wireguard|openvpn|ipsec|tunnel|teredo|6to4|isatap|nat64|dns64)\b", re.IGNORECASE)
PROXY_HINT_RE = re.compile(r"\bproxy\b", re.IGNORECASE)


# Extra override patterns
AUTH_EXEC_RE = re.compile(
    r"(?i)\b("
    r"auth|token|login|signin|session|cookie|oauth|saml|jwt|apikey|api[-_]?key|"
    r"admin|root|sudo|privilege|elevat|"
    r"shell|cmd|exec|payload|dropper|stager|"
    r"download|upload|exfil|leak|dump|"
    r"invoke-webrequest|iwr|downloadstring|frombase64string"
    r")\b"
)
TUNNEL_TEXT_RE = re.compile(
    r"(?i)\b("
    r"vpn|tor|wireguard|openvpn|ipsec|l2tp|pptp|sstp|"
    r"proxy|tunnel|tunneling|relay|"
    r"teredo|6to4|isatap|6in4|6rd|"
    r"nat64|dns64"
    r")\b"
)
IPV6_TUNNEL_ADDR_HINT_RE = re.compile(
    r"(?i)(::ffff:|64:ff9b::|2002:|2001:0000:)"
)
SUSP_URL_HINT_RE = re.compile(
    r"(?i)\b("
    r"\.\./|%2e%2e%2f|%2e%2e\\|"
    r"union\s+select|select\s+.+\s+from|or\s+1=1|"
    r"<script|javascript:|vbscript:|"
    r"/etc/passwd|\\windows\\system32|"
    r"cmd=|exec=|payload=|"
    r"base64|powershell|pwsh|certutil|bitsadmin|"
    r")\b"
)

CONF_REASON_LEGEND = dict(OVERRIDE_REASON_LEGEND) if OVERRIDE_REASON_LEGEND else {
    0: "none",
    1: "explicit_regex",
    2: "critical_keyword",
    3: "ops_tunnel_plus_auth_exec",
    4: "ipv6_tunnel_combo",
    5: "post_put_exfil_rule",
    6: "status404_scan_rule",
    7: "whitelist_suspicious_combo",
    8: "moderate_keyword_plus_strong_signals",
}

def _feat_arr(X: Optional[pd.DataFrame], name: str, n: int) -> np.ndarray:
    if X is None or name not in X.columns:
        return np.zeros(n, dtype=float)
    return pd.to_numeric(X[name], errors="coerce").fillna(0).to_numpy(dtype=float)

def compute_confidential_primary_debug(
    df: pd.DataFrame,
    X: Optional[pd.DataFrame],
    odd_used: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      conf_flag_bool (n,)
      conf_reason_code int8 (n,)
      debug_cols dict[str, bool-array]
    """
    n = len(df)

    # v4: family-aware override engine. This is conservative and typed by
    # log family (proxy/web/apache, IDS/EDR/email, DHCP, OS/auth, network/flow).
    if compute_override_signals is not None:
        try:
            return compute_override_signals(
                df,
                X=X,
                odd_used=odd_used,
                bad_ips=globals().get("BAD_IPS", set()),
                bad_domains=globals().get("BAD_DOMAINS", set()),
                top_domains=globals().get("TOP_DOMAINS", set()),
                whitelist_domains=globals().get("WHITELIST", set()),
            )
        except Exception:
            # Keep the legacy regex path as a safety fallback.
            pass

    # ---- pull text fields safely ----
    raw  = df.get("raw_log",    pd.Series("", index=df.index)).fillna("").astype(str)
    ua   = df.get("user_agent", pd.Series("", index=df.index)).fillna("").astype(str)
    dom  = df.get("domain",     pd.Series("", index=df.index)).fillna("").astype(str)
    fu   = df.get("full_url",   pd.Series("", index=df.index)).fillna("").astype(str)
    up   = df.get("url_path",   pd.Series("", index=df.index)).fillna("").astype(str)
    ref  = df.get("referrer",   pd.Series("", index=df.index)).fillna("").astype(str)
    cmd  = df.get("command",    pd.Series("", index=df.index)).fillna("").astype(str)
    proc = df.get("process",    pd.Series("", index=df.index)).fillna("").astype(str)

    blob = (raw + " " + ua + " " + dom + " " + fu + " " + up + " " + ref + " " + cmd + " " + proc)

    # ---- light decode only when needed (keeps speed) ----
    has_pct = blob.str.contains("%", na=False)
    if has_pct.any():
        blob.loc[has_pct] = blob.loc[has_pct].map(lambda x: unquote(x))

    # ---- benign assertions gate ----
    benign_asserted = blob.str.contains(_NONMAL_RE, na=False)

    # ---- core hits ----
    hit_explicit = blob.str.contains(explicit_regex_pattern, na=False) & (~benign_asserted)

    # critical_pattern is already token-safe; also guard explicit "benign" assertions
    hit_critical = blob.str.contains(critical_pattern, na=False) & (~benign_asserted)

    hit_tunnel_text = (
        blob.str.contains(TUNNEL_TEXT_RE, na=False)
        | blob.str.contains(IPV6_TUNNEL_ADDR_HINT_RE, na=False)
    )
    hit_auth_exec = blob.str.contains(AUTH_EXEC_RE, na=False)
    hit_ops = hit_tunnel_text & hit_auth_exec

    hit_moderate = blob.str.contains(moderate_pattern, na=False)

    # ---- cached classifier regexes (compiled once) ----
    # scanner-ish = scan tools / known scanners in UA
    scanner_re = globals().get("_SCANNER_CLASS_RE", None)
    if scanner_re is None:
        scanner_re = re.compile(
            r"(?<![a-z0-9])(?:"
            r"nmap|masscan|zmap|zgrab|naabu|nuclei|"
            r"nikto|sqlmap|wpscan|acunetix|netsparker|arachni|"
            r"nessus|openvas|qualys|rapid7|"
            r"dirbuster|dirsearch|gobuster|ffuf|feroxbuster|wfuzz|"
            r"burp(?:suite)?|zap(?:roxy)?|"
            r"censys|shodan"
            r")(?![a-z0-9])",
            flags=re.IGNORECASE,
        )
        globals()["_SCANNER_CLASS_RE"] = scanner_re

    # threat-ish = malware/c2/phish/etc (separate from scanners)
    threat_re = globals().get("_THREAT_CLASS_RE", None)
    if threat_re is None:
        threat_re = re.compile(
            r"(?<![a-z0-9])(?:"
            r"ransomware|malware|virus|trojan|worm|spyware|rootkit|backdoor|"
            r"stealer|keylogger|botnet|cryptojack|coinhive|xmrig|"
            r"phish|phishing|credential|creds|"
            r"c2|beacon|meterpreter|cobalt\s*strike|sliver|empire|covenant|mythic|"
            r"emotet|trickbot|dridex|qakbot|qbot|agenttesla|formbook|azorult|"
            r"raccoon(?:\s*stealer)?|lumm?a\s*stealer|redline|vidar|"
            r"njrat|asyncrat|remcos|warzone(?:\s*rat)?|"
            r"mirai|mozi"
            r")(?![a-z0-9])",
            flags=re.IGNORECASE,
        )
        globals()["_THREAT_CLASS_RE"] = threat_re

    tag_scanner = blob.str.contains(scanner_re, na=False) & (~benign_asserted)
    # treat "critical" as threat unless it's clearly scanner-only
    tag_threat = (hit_critical | blob.str.contains(threat_re, na=False)) & (~benign_asserted)
    tag_threat = tag_threat & (~tag_scanner)

    # ---- auxiliary signals ----
    odd = (odd_used.astype(int) > 0) if odd_used is not None else np.zeros(n, dtype=bool)

    ip_bad = (
        pd.to_numeric(df.get("ip_bad_truth", 0), errors="coerce")
          .fillna(0)
          .to_numpy(dtype=float) > 0
    )

    suspicious_geo = (_feat_arr(X, "suspicious_geo", n) > 0)

    suspicious_url = (
        (_feat_arr(X, "suspicious_url", n) > 0)
        | blob.str.contains(SUSP_URL_HINT_RE, na=False).to_numpy(dtype=bool)
    )

    lolbin_sig = (
        (_feat_arr(X, "malicious_lolbin_ua", n) > 0)
        | blob.str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
    )

    ipv6_tunnel_any = (
        (_feat_arr(X, "ipv6_tunnel_any", n) > 0)
        | hit_tunnel_text.to_numpy(dtype=bool)
    )

    # ---- whitelist suspicious combo ----
    wl_combo_feat = (_feat_arr(X, "whitelist_suspicious_combo", n) > 0)
    wl_hit = (
        pd.to_numeric(df.get("whitelist_hit", 0), errors="coerce")
          .fillna(0)
          .to_numpy(dtype=float) > 0
    )
    wl_combo = wl_combo_feat | (wl_hit & (ip_bad | suspicious_geo | suspicious_url | lolbin_sig | odd | ipv6_tunnel_any))

    # ---- POST/PUT exfil rule (feature-based) ----
    post_z = _feat_arr(X, "method_POST_ratio_z", n)
    put_z  = _feat_arr(X, "method_PUT_ratio_z", n)
    win_cnt = _feat_arr(X, "method_window_count", n)
    req_hr  = _feat_arr(X, "requests_per_ip_hour", n)
    high_bytes_out = (_feat_arr(X, "high_bytes_out", n) > 0)

    POST_Z = 2.0
    PUT_Z  = 1.5
    MIN_SUPPORT = 8

    hit_post_put = (
        ((post_z >= POST_Z) | (put_z >= PUT_Z))
        & high_bytes_out
        & (win_cnt >= MIN_SUPPORT)
        & (req_hr >= 3.0)
    )

    # ---- status 404 scan-ish rule ----
    status = pd.to_numeric(df.get("status", 200), errors="coerce").fillna(200).to_numpy(dtype=int)
    hit_404 = (
        (status == 404)
        & ((post_z >= 1.5) | (put_z >= 1.0))
        & (win_cnt >= MIN_SUPPORT)
        & (req_hr >= 3.0)
    )

    # ---- combos ----
    hit_ipv6_combo = ipv6_tunnel_any & (ip_bad | suspicious_geo | suspicious_url | lolbin_sig | odd | wl_combo)

    hit_moderate_combo = (
        hit_moderate.to_numpy(dtype=bool)
        & (ip_bad | suspicious_geo | suspicious_url | lolbin_sig | ipv6_tunnel_any | odd | wl_combo)
    )

    # ---- “suspicious” tag (for display) ----
    # NOTE: this is not the final ML label; it’s a semantic tag you can show.
    tag_suspicious = (
        tag_scanner.to_numpy(dtype=bool)
        | tag_threat.to_numpy(dtype=bool)
        | hit_ops.to_numpy(dtype=bool)
        | hit_post_put
        | hit_404
        | wl_combo
        | hit_moderate_combo
    )

    # ---- final confidential primary flag ----
    conf = (
        hit_explicit.to_numpy(dtype=bool)
        | hit_critical.to_numpy(dtype=bool)
        | hit_ops.to_numpy(dtype=bool)
        | hit_ipv6_combo
        | hit_post_put
        | hit_404
        | wl_combo
        | hit_moderate_combo
    )

    # ---- reason code (priority order) ----
    conds = [
        hit_explicit.to_numpy(dtype=bool),
        hit_critical.to_numpy(dtype=bool),
        hit_ops.to_numpy(dtype=bool),
        hit_ipv6_combo,
        hit_post_put,
        hit_404,
        wl_combo,
        hit_moderate_combo,
    ]
    codes = np.select(conds, [1, 2, 3, 4, 5, 6, 7, 8], default=0).astype(np.int8)

    debug = {
        # existing
        "conf_hit_explicit": hit_explicit.to_numpy(dtype=bool),
        "conf_hit_critical": hit_critical.to_numpy(dtype=bool),
        "conf_hit_ops": hit_ops.to_numpy(dtype=bool),
        "conf_hit_ipv6_combo": hit_ipv6_combo,
        "conf_hit_post_put": hit_post_put,
        "conf_hit_404": hit_404,
        "conf_hit_wl_combo": wl_combo,
        "conf_hit_moderate_combo": hit_moderate_combo,
        "conf_benign_asserted": benign_asserted.to_numpy(dtype=bool),

        # NEW: display tags
        "conf_tag_scanner": tag_scanner.to_numpy(dtype=bool),
        "conf_tag_threat": tag_threat.to_numpy(dtype=bool),
        "conf_tag_suspicious": tag_suspicious.astype(bool),
    }
    return conf, codes, debug

def compute_suspicious_context(df: pd.DataFrame, X: Optional[pd.DataFrame], combined_primary: np.ndarray, odd_used: np.ndarray) -> np.ndarray:
    raw = df.get("raw_log", pd.Series("", index=df.index)).fillna("").astype(str)
    ua  = df.get("user_agent", pd.Series("", index=df.index)).fillna("").astype(str)
    cmd = df.get("command", pd.Series("", index=df.index)).fillna("").astype(str)
    proc= df.get("process", pd.Series("", index=df.index)).fillna("").astype(str)

    s = np.zeros(len(df), dtype=bool)
    s |= (combined_primary > 0)
    s |= raw.str.contains(SUSP_TZ_HARD_RE, na=False).to_numpy(dtype=bool)
    s |= ua.str.contains(BOT_UA_RE, na=False).to_numpy(dtype=bool)
    s |= cmd.str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
    s |= proc.str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
    s |= (odd_used.astype(int) > 0) & ua.str.contains(BOT_UA_RE, na=False).to_numpy(dtype=bool)

    if X is not None:
        for c in ["ipv6_tunnel_any", "whitelist_suspicious_combo", "timestamp_suspicious_tz", "malicious_lolbin_ua", "suspicious_url"]:
            if c in X.columns:
                s |= (pd.to_numeric(X[c], errors="coerce").fillna(0).to_numpy() > 0)

    return s


# ============================================================
# Timezone + wall-time parsing (odd-hours local unless suspicious)
# ============================================================
TZ_ABBR_OFFSETS_MIN = {
    "UTC": 0, "GMT": 0,
    "IST": 330, "PKT": 300, "BDT": 480, "MYT": 480, "SGT": 480, "HKT": 480, "ICT": 420,
    "CET": 60, "CEST": 120, "EET": 120, "EEST": 180, "BST": 60, "MSK": 180,
    "PST": -480, "PDT": -420, "MST": -420, "MDT": -360, "CST": -360,
    "EST": -300, "EDT": -240,
    "JST": 540, "KST": 540,
    "AEST": 600, "AEDT": 660, "ACST": 570, "AWST": 480,
    "NZST": 720, "NZDT": 780,
}
TZ_ABBR_RE = re.compile(r"\b(" + "|".join(sorted(TZ_ABBR_OFFSETS_MIN.keys(), key=len, reverse=True)) + r")\b")

SITE_TZ_MIN = {
    "NYC": -300, "NY": -300, "LON": 0, "FRA": 60, "AMS": 60,
    "TOK": 540, "TKY": 540, "DEL": 330, "BLR": 330, "BOM": 330,
    "SIN": 480, "HKG": 480, "DXB": 240,
    "SF": -480, "SFO": -480, "LA": -480, "SEA": -480,
    "CHI": -360, "DAL": -360,
}


# Precompiled workstation site-code extractor (vectorized)
_SITE_CODE_RE = re.compile(
    r"(?:^|[-_/\.])("
    + "|".join(sorted(SITE_TZ_MIN.keys(), key=len, reverse=True))
    + r")(?:[-_/\.]|$)"
)

def infer_site_offset_min_series(workstation_s: pd.Series) -> pd.Series:
    """Vectorized workstation offset inference. Returns minutes offset (can be NaN)."""
    ws = workstation_s.fillna("").astype(str).str.upper()
    code = ws.str.extract(_SITE_CODE_RE, expand=False)
    off = code.map(SITE_TZ_MIN)
    return pd.to_numeric(off, errors="coerce")

def infer_site_offset_min(workstation: str) -> Optional[int]:
    w = (workstation or "").upper()
    if not w:
        return None
    parts = re.split(r"[-_/\.]", w)
    for c in parts:
        if 2 <= len(c) <= 4 and c in SITE_TZ_MIN:
            return int(SITE_TZ_MIN[c])
    return None

_TS_EXTRACTORS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[ ]?[+-]\d{2}:?\d{2})\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}\b"),
    re.compile(r"\b\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4}\b"),
    re.compile(r"\b(?:audit\()?\s*(\d{9,19})(?:\.\d+)?(?:\))?\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}(?:\.\d+)?(?:\s*(?:AM|PM))?\b", re.IGNORECASE),
]

def extract_ts_candidate(ts_str: str, raw_line: str) -> str:
    s = (ts_str or "").strip().strip("[]")
    if s:
        return s
    r = raw_line or ""
    for rx in _TS_EXTRACTORS:
        m = rx.search(r)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 1:
            return (m.group(1) or "").strip()
        return (m.group(0) or "").strip()
    return ""

def _parse_offset_token(tok: str) -> Optional[int]:
    tok = (tok or "").strip()
    if not tok:
        return None
    if tok == "Z":
        return 0
    m = re.match(r"^([+-])(\d{1,2}):?(\d{2})$", tok)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hh = int(m.group(2)); mm = int(m.group(3))
        return sign * (hh * 60 + mm)
    return None

def parse_wall_dt_and_offset(ts_str: str, raw_line: str = "", workstation: str = "") -> Tuple[datetime, int, int]:
    raw_line = raw_line or ""
    workstation = workstation or ""
    suspicious = bool(SUSP_TZ_HARD_RE.search(raw_line))

    s = extract_ts_candidate(ts_str, raw_line).strip().strip("[]")

    # epoch (10/13)
    if s.isdigit():
        try:
            n = int(s)
            if len(s) == 10:
                dt = datetime.fromtimestamp(n, tz=timezone.utc)
                return dt.replace(tzinfo=None), 0, 1
            if len(s) == 13:
                dt = datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
                return dt.replace(tzinfo=None), 0, 1
        except Exception:
            pass

    # apache
    try:
        if re.match(r"^\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4}$", s):
            dt = datetime.strptime(s, "%d/%b/%Y:%H:%M:%S %z")
            off = int(dt.utcoffset().total_seconds() // 60)
            return dt.replace(tzinfo=None), off, 1
    except Exception:
        pass

    # YYYY-MM-DD HH:MM:SS +0530
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}$", s):
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")
            off = int(dt.utcoffset().total_seconds() // 60)
            return dt.replace(tzinfo=None), off, 1
    except Exception:
        pass

    off_hint = None
    m_off = re.search(r"([+-]\d{2}:?\d{2})\b", s)
    if m_off:
        off_hint = _parse_offset_token(m_off.group(1))

    mz = TZ_ABBR_RE.search(raw_line)
    if mz:
        off_hint = TZ_ABBR_OFFSETS_MIN.get(mz.group(1), off_hint)

    if (off_hint is None) and (not suspicious):
        off_hint = infer_site_offset_min(workstation)

    # dateutil
    try:
        from dateutil import parser as dtparser  # type: ignore
        from dateutil.tz import tzoffset  # type: ignore
        tzinfos = {k: tzoffset(k, v * 60) for k, v in TZ_ABBR_OFFSETS_MIN.items()}
        dt = dtparser.parse(s, tzinfos=tzinfos, fuzzy=True)
        if getattr(dt, "tzinfo", None) is not None and dt.utcoffset() is not None:
            off = int(dt.utcoffset().total_seconds() // 60)
            return dt.replace(tzinfo=None), off, 1
        if off_hint is None or suspicious:
            off_hint = 0
        return dt.replace(tzinfo=None), int(off_hint), 1
    except Exception:
        pass

    # fallback
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p"):
        try:
            dt = datetime.strptime(s, fmt)
            if off_hint is None or suspicious:
                off_hint = 0
            return dt, int(off_hint), 1
        except Exception:
            continue

    return datetime.utcnow(), 0, 0

def compute_odd_hours_from_wall(wall_dt: datetime) -> Tuple[int, float]:
    hf = wall_dt.hour + wall_dt.minute / 60.0
    odd = 1 if (hf >= 23.0 or hf < 5.5) else 0
    return odd, hf

def compute_odd_hours_from_utc(ts_utc: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts_utc, errors="coerce", utc=True)
    hf = t.dt.hour.fillna(0).astype(float) + t.dt.minute.fillna(0).astype(float) / 60.0
    return ((hf >= 23.0) | (hf < 5.5)).astype(np.int8).to_numpy()

# ============================================================
# Streamlit helpers
# ============================================================
def _clean_display_col_name(col: Any, pos: int) -> str:
    """Return a stable, semantic display column name.

    Streamlit/PyArrow does not tolerate duplicate names, and uploaded headerless
    tables can arrive as integer/Unnamed/Column-N columns. This also normalizes
    generic preserved-source columns such as input_Column_1 so they never leak
    into the dashboard.
    """
    if isinstance(col, (int, np.integer)):
        return PRIMARY_COLS[int(col)] if int(col) < len(PRIMARY_COLS) else f"extra_field_{int(col) + 1}"
    s = str(col).strip()
    m = re.fullmatch(r"input[_ ]column[_ ]?(\d+)", s, flags=re.I)
    if m:
        j = max(0, int(m.group(1)) - 1)
        return PRIMARY_COLS[j] if j < len(PRIMARY_COLS) else f"extra_field_{j + 1}"
    m = re.fullmatch(r"(?:column|unnamed|extra[_ ]field)[_ ]?(\d+)", s, flags=re.I)
    if m:
        j = max(0, int(m.group(1)) - 1)
        return PRIMARY_COLS[j] if j < len(PRIMARY_COLS) else f"extra_field_{j + 1}"
    if not s or s.lower().startswith("unnamed") or re.fullmatch(r"column\s*\d+", s, flags=re.I):
        return PRIMARY_COLS[pos] if pos < len(PRIMARY_COLS) else f"extra_field_{pos + 1}"
    return s

def _dedupe_columns(cols: List[Any]) -> List[str]:
    """Preserve order while removing duplicate column names."""
    out: List[str] = []
    seen = set()
    for i, c in enumerate(cols):
        name = _clean_display_col_name(c, i)
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out

def _ensure_unique_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    new_cols: List[str] = []
    counts: Dict[str, int] = {}
    for i, c in enumerate(out.columns):
        base = _clean_display_col_name(c, i)
        n = counts.get(base, 0)
        counts[base] = n + 1
        new_cols.append(base if n == 0 else f"{base}__dup{n}")
    out.columns = new_cols
    return out

def ui_df(df: pd.DataFrame, **kwargs):
    df = _ensure_unique_df_columns(df)
    try:
        return st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        return st.dataframe(df, **kwargs)

def ui_btn(label: str, **kwargs) -> bool:
    try:
        return bool(st.button(label, use_container_width=True, **kwargs))
    except TypeError:
        return bool(st.button(label, **kwargs))

def ui_dl(label: str, data: bytes, file_name: str, mime: str, **kwargs):
    try:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, use_container_width=True, **kwargs)
    except TypeError:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, **kwargs)

def _fig_to_png(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def _arrow(val: float, eps: float = 1e-12) -> str:
    if val > eps: return "↑"
    if val < -eps: return "↓"
    return "→"

def parse_index_list(text: str, n: int) -> List[int]:
    out: List[int] = []
    t = (text or "").strip()
    if not t:
        return out
    parts = [p.strip() for p in t.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = a.strip(); b = b.strip()
            if a.isdigit() and b.isdigit():
                lo = int(a); hi = int(b)
                if lo > hi: lo, hi = hi, lo
                out.extend(list(range(lo, hi + 1)))
        else:
            if p.isdigit():
                out.append(int(p))
    return sorted(set([i for i in out if 0 <= i < n]))

# ============================================================
# Bundle + init components
# ============================================================
DEFAULT_MODEL_DIR_WIN = r"C:\Users\prash\Downloads\doc-files-cybersecurity\cyber_streamlit_complete\cyber_streamlit_complete\models_comb_saa"
DEFAULT_MODEL_DIR_REL = os.path.join(ROOT_DIR, "models_comb_saa")
ALT_MODEL_DIR_WIN = r"C:\Users\prash\Downloads\doc-files-cybersecurity\cyber_streamlit_complete\cyber_streamlit_complete\models_comb_saa"
ALT_MODEL_DIR_REL = os.path.join(ROOT_DIR, "models_comb_sa")

_model_dir_default = None
for cand in (DEFAULT_MODEL_DIR_WIN, DEFAULT_MODEL_DIR_REL, ALT_MODEL_DIR_WIN, ALT_MODEL_DIR_REL):
    if os.path.isdir(cand):
        _model_dir_default = cand
        break
if _model_dir_default is None:
    _model_dir_default = DEFAULT_MODEL_DIR_REL

model_dir = os.getenv("MODEL_DIR", _model_dir_default)

with st.sidebar.expander("⚙️ Advanced", expanded=False):
    model_dir = st.text_input("📦 Model directory", value=model_dir, key="adv_model_dir")
    chunk_size = st.number_input("Chunk size (processing)", 1_000, 200_000, 20_000, 1_000, key="chunk_size")
    max_preview_rows = st.number_input("Max preview rows in UI", 100, 500_000, 50_000, 100, key="max_preview_rows")
    enable_shap_dashboard = st.checkbox("Enable SHAP dashboard after classification", value=False, key="enable_shap_dashboard")
    # Default timezone for naive timestamps (non-suspicious rows only).
    # Use this to align inference with the timezone assumptions used during training.
    _tz_opts = [
        "Auto (infer from log; else UTC)",
        "UTC (0)",
        "IST (+05:30)",
        "CET (+01:00)",
        "CEST (+02:00)",
        "EET (+02:00)",
        "EEST (+03:00)",
        "EST (-05:00)",
        "EDT (-04:00)",
        "PST (-08:00)",
        "PDT (-07:00)",
        "Custom minutes",
    ]
    _tz_mode = st.selectbox("Default timezone for naive timestamps (non-suspicious)", _tz_opts, index=0, key="default_tz_mode")

    _tz_map = {
        "UTC (0)": 0,
        "IST (+05:30)": 330,
        "CET (+01:00)": 60,
        "CEST (+02:00)": 120,
        "EET (+02:00)": 120,
        "EEST (+03:00)": 180,
        "EST (-05:00)": -300,
        "EDT (-04:00)": -240,
        "PST (-08:00)": -480,
        "PDT (-07:00)": -420,
    }

    if _tz_mode == "Custom minutes":
        _tz_min = int(st.number_input("Default TZ offset minutes (e.g., IST=330, CET=60)", -840, 840, 0, 15, key="default_tz_min"))
    elif _tz_mode == "Auto (infer from log; else UTC)":
        _tz_min = 0
    else:
        _tz_min = int(_tz_map.get(_tz_mode, 0))

    # Used by add_wall_time_and_odd() when timestamp has no tz and no inference is available.
    DEFAULT_TZ_MIN_NON_SUSPICIOUS = _tz_min

    FAST_PRIMARY_ALL = st.checkbox(
        "Small-batch acceleration: skip ML scoring when primary overrides already decide all rows",
        value=True,
        key="fast_primary_all",
    )
    FAST_PRIMARY_MAX_ROWS = int(st.number_input(
        "Small-batch acceleration max rows",
        100,
        100000,
        5000,
        100,
        key="fast_primary_max_rows",
    ))

    st.caption("Large files are processed chunked for speed. PDF is best for summaries + selected rows.")

if not os.path.isdir(model_dir):
    st.error("Model directory not found. Fix it in Sidebar → Advanced.")
    st.stop()

try:
    _early_cache_resource_no_spinner = st.cache_resource(show_spinner=False)
except Exception:
    def _early_cache_resource_no_spinner(func=None, **_kwargs):
        def deco(f):
            return f
        return deco(func) if func is not None else deco

@_early_cache_resource_no_spinner
def _cached_bundle_load(model_dir_key: str, art_src_key: str):
    if callable(load_bundle):
        try:
            return load_bundle(model_dir_key), ""
        except Exception as e:
            return _fallback_load_bundle(model_dir_key), f"artifacts.load_bundle failed: {e} — using fallback loader."
    return _fallback_load_bundle(model_dir_key), "artifacts.load_bundle not found — using fallback loader."

BUNDLE, _bundle_note = _cached_bundle_load(str(model_dir), ART_SRC)
if _bundle_note:
    st.warning(_bundle_note)

BAD_IPS = set(getattr(BUNDLE, "bad_ips", set()) or set())
BAD_DOMAINS = set(getattr(BUNDLE, "bad_domains", set()) or set())
TOP_DOMAINS = set(getattr(BUNDLE, "top_domains", set()) or set())
SCALER = getattr(BUNDLE, "scaler", None)
BUNDLE_FEATURE_COLS = getattr(BUNDLE, "feature_columns", []) or []
BUNDLE_FEATURE_WEIGHTS = getattr(BUNDLE, "feature_weights", {}) or {}

# MoE discovery
MOE_META_PATH = os.path.join(model_dir, "MoE_meta_model.pkl")
MOE_FEAT_PATH = os.path.join(model_dir, "MoE_meta_features.pkl")
MOE_EXP_PATH  = os.path.join(model_dir, "MoE_experts.pkl")
ISO_PATH_A    = os.path.join(model_dir, "iso_forest.pkl")
ISO_PATH_B    = os.path.join(model_dir, "isolation_forest.pkl")
ISO_PATH      = ISO_PATH_A if os.path.exists(ISO_PATH_A) else ISO_PATH_B
MOE_THR_PATH  = os.path.join(model_dir, "optimal_threshold_MoE.pkl")
MOE_AVAILABLE = all(os.path.exists(p) for p in [MOE_META_PATH, MOE_FEAT_PATH, MOE_EXP_PATH, ISO_PATH, MOE_THR_PATH])

@lru_cache(maxsize=4)
def _load_moe_artifacts_cached(meta_path: str, feat_path: str, exp_path: str, iso_path: str, thr_path: str):
    """Cache MoE/ISO artifacts so small batches do not repeatedly pay joblib load cost."""
    moe_meta = joblib.load(meta_path)
    moe_cols = joblib.load(feat_path)
    moe_exps = joblib.load(exp_path)
    iso = joblib.load(iso_path)
    thr_saved = float(joblib.load(thr_path))
    return moe_meta, moe_cols, moe_exps, iso, thr_saved

IMPUTER = ForensicImputer(
    priors=getattr(BUNDLE, "priors", {}) if isinstance(getattr(BUNDLE, "priors", {}), dict) else {},
    resolver_state=getattr(BUNDLE, "resolver_state", {}) if isinstance(getattr(BUNDLE, "resolver_state", {}), dict) else {},
    bytes_priors=getattr(BUNDLE, "bytes_priors", {}) if isinstance(getattr(BUNDLE, "bytes_priors", {}), dict) else {},
    top_domains=TOP_DOMAINS,
)

try:
    if callable(load_feature_engineering):
        our_custom_feature_engineering_function = load_feature_engineering(BUNDLE)
    else:
        fe_mod = importlib.import_module("feature_engineering")
        our_custom_feature_engineering_function = getattr(fe_mod, "our_custom_feature_engineering_function")
except Exception as e:
    st.error(f"Failed to import feature_engineering. Error: {e}")
    st.stop()


try:
    _FE_PARAMS = set(inspect.signature(our_custom_feature_engineering_function).parameters)
except Exception:
    _FE_PARAMS = set()
_FE_KW_DEBUG = "debug" in _FE_PARAMS
_FE_KW_WHITELIST = "whitelist_domains" in _FE_PARAMS

def _run_feature_engineering(df_fe: pd.DataFrame) -> pd.DataFrame:
    """Run FE with optional whitelist support and keep lossless feature aliases.

    Several trained bundles preserve original feature names with spaces while
    the legacy app sanitized them.  We keep both variants here; artifacts.prepare_model_matrix
    performs the final exact alignment to the scaler/model schema.
    """
    kwargs: Dict[str, Any] = {}
    if _FE_KW_DEBUG:
        # Skip the FE function's explicit gc.collect() on interactive batches.
        # This does not alter feature values; it only removes avoidable small-batch latency.
        kwargs["debug"] = True
    if _FE_KW_WHITELIST:
        kwargs["whitelist_domains"] = globals().get("WHITELIST", set())

    X = our_custom_feature_engineering_function(df_fe, **kwargs)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.copy()
    X.columns = [str(c) for c in X.columns]
    for c in list(X.columns):
        alias = str(c).replace(" ", "_")
        if alias and alias not in X.columns:
            X[alias] = X[c]
    return X

# Heavy objects are loaded lazily and cached. This prevents 100-log batches
# from spending more time loading artifacts than doing inference.
@_cache_resource_no_spinner
def _cached_load_supervised_resource(model_dir_abs: str, ui_name: str, art_src: str):
    # Use the already cached Bundle. Do NOT call load_bundle() again here;
    # that can re-scan/inspect artifacts and is the main small-batch latency trap.
    if hasattr(BUNDLE, "load_supervised") and callable(getattr(BUNDLE, "load_supervised")):
        return BUNDLE.load_supervised(ui_name)
    return _fallback_load_bundle(model_dir_abs).load_supervised(ui_name)

def load_supervised(ui_name: str):
    try:
        return _cached_load_supervised_resource(os.path.abspath(str(model_dir)), str(ui_name), ART_SRC)
    except Exception:
        if hasattr(BUNDLE, "load_supervised") and callable(getattr(BUNDLE, "load_supervised")):
            return BUNDLE.load_supervised(ui_name)
        return _fallback_load_bundle(model_dir).load_supervised(ui_name)

@_cache_resource_no_spinner
def _get_shap_engine_cached(model_dir_abs: str, art_src: str):
    if ShapEngine is None:
        return None
    try:
        return ShapEngine(BUNDLE)
    except Exception:
        return None

def get_shap_engine():
    return _get_shap_engine_cached(os.path.abspath(str(model_dir)), ART_SRC)

SHAP_ENGINE = None  # placeholder; real engine loads only when SHAP is enabled/requested.

# ============================================================
# Whitelist utilities
# ============================================================
def load_whitelist_domains_csv(file_obj) -> set:
    if file_obj is None:
        return set()
    try:
        data = file_obj.getvalue()
        tmp = pd.read_csv(BytesIO(data))
        if tmp.shape[1] == 0:
            return set()
        col = tmp.columns[0]
        for c in tmp.columns:
            if str(c).strip().lower() in ("domain", "domains", "host", "hostname"):
                col = c
                break
        vals = tmp[col].dropna().astype(str).str.strip().str.lower()
        vals = vals.str.replace(r"^https?://", "", regex=True).str.split("/", n=1).str[0]
        vals = vals.str.split(":", n=1).str[0]
        return set([v for v in vals.tolist() if v and v not in ("nan", "none", "null", "-", "--")])
    except Exception:
        try:
            txt = data.decode("utf-8", errors="ignore").splitlines()
            out = set()
            for ln in txt:
                ln = (ln or "").strip()
                if not ln:
                    continue
                parts = ln.split(",")
                dom = parts[-1].strip().lower()
                dom = dom.replace("http://", "").replace("https://", "").split("/")[0].split(":")[0].strip()
                if dom and dom not in ("domain", "domains", "host", "hostname", "nan", "none", "null", "-", "--"):
                    out.add(dom)
            return out
        except Exception:
            return set()

def whitelist_hit_series(dom: pd.Series, wl: set) -> pd.Series:
    d = dom.fillna("").astype(str).str.lower().str.strip()
    if not wl:
        return pd.Series(np.zeros(len(d), dtype=np.int8), index=d.index)

    codes, uniq = pd.factorize(d, sort=False)

    def hit_one(x: str) -> int:
        if not x:
            return 0
        if x in wl:
            return 1
        parts = x.split(".")
        for k in (2, 3, 4):
            if len(parts) >= k:
                suf = ".".join(parts[-k:])
                if suf in wl:
                    return 1
        return 0

    hit_u = np.fromiter((hit_one(u) for u in uniq), dtype=np.int8, count=len(uniq))
    return pd.Series(hit_u[codes], index=d.index, dtype=np.int8)

def compute_info_complete(df_imp: pd.DataFrame) -> np.ndarray:
    def ok_text(col: str) -> np.ndarray:
        s = df_imp.get(col, "").fillna("").astype(str).str.strip()
        bad = s.eq("") | s.str.lower().isin({"notprovided", "unknown", "nan", "none", "null", "-", "--", "0.0.0.0", "::", "local.invalid", "missing", "missing_token"})
        return (~bad).to_numpy(dtype=bool)

    ok = np.ones(len(df_imp), dtype=bool)
    for c in ["domain", "user_agent", "method", "timestamp", "client_ip", "dest_ip"]:
        ok &= ok_text(c)

    if "timestamp_parse_ok" in df_imp.columns:
        ok &= (pd.to_numeric(df_imp["timestamp_parse_ok"], errors="coerce").fillna(0).to_numpy() > 0)

    return ok

def apply_whitelist(
    prob: np.ndarray,
    dom: pd.Series,
    suspicious_mask: np.ndarray,
    whitelist_set: set,
    mode: str,
    custom_factor: Optional[float],
    info_complete: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    prob = np.asarray(prob, dtype=float)
    wl_hit = whitelist_hit_series(dom, whitelist_set).to_numpy(dtype=np.int8)

    if mode == "Off" or not whitelist_set:
        return np.clip(prob, 0.0, 1.0), wl_hit, 1.0, np.zeros(len(prob), dtype=bool)

    if mode == "Soft":
        factor = 0.85
    elif mode == "Medium":
        factor = 0.70
    elif mode == "Hard":
        factor = 0.50
    elif mode == "Custom":
        try:
            factor = float(custom_factor) if custom_factor is not None else 0.70
        except Exception:
            factor = 0.70
        factor = float(np.clip(factor, 0.05, 1.0))
    else:
        factor = 1.0

    eligible = (~np.asarray(suspicious_mask, dtype=bool))
    if info_complete is not None:
        eligible &= np.asarray(info_complete, dtype=bool)

    mask = (wl_hit > 0) & eligible
    adj = prob.copy()
    if mask.any():
        adj[mask] = adj[mask] * factor

    return np.clip(adj, 0.0, 1.0), wl_hit, float(factor), mask

# ============================================================
# Exfil analysis (bytes_out per domain/ip, verdicts)
# ============================================================
def compute_exfil_tables(df_imp: pd.DataFrame, preds: np.ndarray, topk: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp = df_imp.copy()
    tmp["_pred"] = pd.Series(preds, index=tmp.index).astype(int)

    tmp["bytes_out"] = pd.to_numeric(tmp.get("bytes_out", 0), errors="coerce").fillna(0).clip(lower=0)
    tmp["bytes_in"] = pd.to_numeric(tmp.get("bytes_in", 0), errors="coerce").fillna(0).clip(lower=0)

    # Domain
    dom = tmp.get("domain", pd.Series("", index=tmp.index)).fillna("").astype(str).str.lower()
    tmp["_domain"] = dom
    g = tmp.groupby("_domain", sort=False)

    dom_tbl = g.agg(
        entries=("_pred", "size"),
        malicious_entries=("_pred", "sum"),
        bytes_out_total=("bytes_out", "sum"),
        bytes_out_avg=("bytes_out", "mean"),
        bytes_in_total=("bytes_in", "sum"),
    ).reset_index().rename(columns={"_domain": "domain"})
    dom_tbl["malicious_pct"] = np.where(dom_tbl["entries"] > 0, 100.0 * dom_tbl["malicious_entries"] / dom_tbl["entries"], 0.0)
    dom_tbl["verdict"] = np.where(dom_tbl["malicious_entries"] > 0, "Malicious", "Legit")
    dom_tbl = dom_tbl.sort_values(["bytes_out_total", "malicious_entries"], ascending=[False, False]).head(int(topk)).reset_index(drop=True)
    dom_tbl = dom_tbl[dom_tbl["domain"].astype(str).str.len() > 0].reset_index(drop=True)

    # Dest IP
    dip = tmp.get("dest_ip", pd.Series("", index=tmp.index)).fillna("").astype(str)
    tmp["_dest_ip"] = dip
    g2 = tmp.groupby("_dest_ip", sort=False)

    ip_tbl = g2.agg(
        entries=("_pred", "size"),
        malicious_entries=("_pred", "sum"),
        bytes_out_total=("bytes_out", "sum"),
        bytes_out_avg=("bytes_out", "mean"),
        bytes_in_total=("bytes_in", "sum"),
    ).reset_index().rename(columns={"_dest_ip": "dest_ip"})
    ip_tbl["malicious_pct"] = np.where(ip_tbl["entries"] > 0, 100.0 * ip_tbl["malicious_entries"] / ip_tbl["entries"], 0.0)
    ip_tbl["verdict"] = np.where(ip_tbl["malicious_entries"] > 0, "Malicious", "Legit")
    ip_tbl = ip_tbl.sort_values(["bytes_out_total", "malicious_entries"], ascending=[False, False]).head(int(topk)).reset_index(drop=True)
    ip_tbl = ip_tbl[ip_tbl["dest_ip"].astype(str).str.len() > 0].reset_index(drop=True)

    return dom_tbl, ip_tbl

# ============================================================
# PDF generation (ReportLab preferred; structured and readable)
# ============================================================
def build_pdf_bytes(
    df_entries: pd.DataFrame,
    meta: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    shap_payloads: Optional[Dict[str, Any]] = None,
    exfil_domain: Optional[pd.DataFrame] = None,
    exfil_ip: Optional[pd.DataFrame] = None,
    raw_log_max_chars: int = 0,
) -> Tuple[bytes, str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CyberSecurity_Report_{ts}.pdf"
    last_err = ""

    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image as RLImage, PageBreak, KeepTogether
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import mm

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=landscape(A4),
            leftMargin=10 * mm, rightMargin=10 * mm, topMargin=10 * mm, bottomMargin=10 * mm
        )
        styles = getSampleStyleSheet()
        normal = ParagraphStyle("normal", parent=styles["BodyText"], fontSize=9, leading=11)
        small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10)
        tiny = ParagraphStyle("tiny", parent=styles["BodyText"], fontSize=7, leading=9, wordWrap="CJK")

        def P(x: str, sty=normal):
            return Paragraph(safe_pdf_text(x), sty)

        story: List[Any] = []
        story.append(P("CyberSecurity Log Classification Report", styles["Title"]))
        story.append(Spacer(1, 6))
        story.append(P(meta.get("summary_line", ""), normal))
        story.append(Spacer(1, 8))

        if meta.get("summary_block"):
            story.append(P("Summary", styles["Heading2"]))
            story.append(Spacer(1, 4))
            story.append(P(meta.get("summary_block", ""), normal))
            story.append(Spacer(1, 8))

        if metrics:
            story.append(P("Performance Metrics (Labeled)", styles["Heading2"]))
            story.append(Spacer(1, 4))
            story.append(P(metrics.get("line", ""), normal))
            story.append(Spacer(1, 6))

            cm = metrics.get("cm")
            if cm is not None:
                try:
                    cm_arr = np.asarray(cm, dtype=int)
                    if cm_arr.shape == (2, 2):
                        story.append(P("Confusion Matrix", styles["Heading3"]))
                        cm_tbl_data = [
                            ["", "Pred 0", "Pred 1"],
                            ["Actual 0", str(int(cm_arr[0, 0])), str(int(cm_arr[0, 1]))],
                            ["Actual 1", str(int(cm_arr[1, 0])), str(int(cm_arr[1, 1]))],
                        ]
                        cm_tbl = Table(cm_tbl_data, repeatRows=1, colWidths=[30 * mm, 30 * mm, 30 * mm])
                        cm_tbl.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]))
                        story.append(cm_tbl)
                        story.append(Spacer(1, 8))
                except Exception:
                    pass

            for key, title, w, h in [
                ("roc_png", "ROC Curve", 160 * mm, 95 * mm),
                ("pr_png",  "Precision-Recall Curve", 160 * mm, 95 * mm),
            ]:
                png = metrics.get(key)
                if png:
                    try:
                        story.append(P(title, styles["Heading3"]))
                        story.append(Spacer(1, 3))
                        story.append(RLImage(BytesIO(png), width=w, height=h))
                        story.append(Spacer(1, 10))
                    except Exception:
                        pass

        if exfil_domain is not None or exfil_ip is not None:
            story.append(PageBreak())
            story.append(P("Data Exfiltration Summary (bytes_out)", styles["Heading2"]))
            story.append(Spacer(1, 6))

            if exfil_domain is not None and len(exfil_domain) > 0:
                story.append(P("Top Domains by bytes_out", styles["Heading3"]))
                show = exfil_domain.copy()
                cols = ["domain", "bytes_out_total", "entries", "malicious_entries", "malicious_pct", "verdict"]
                cols = [c for c in cols if c in show.columns]
                show = show[cols].copy().head(50)
                data = [cols] + show.astype(str).values.tolist()
                tbl = Table(data, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 10))

            if exfil_ip is not None and len(exfil_ip) > 0:
                story.append(P("Top Dest IPs by bytes_out", styles["Heading3"]))
                show = exfil_ip.copy()
                cols = ["dest_ip", "bytes_out_total", "entries", "malicious_entries", "malicious_pct", "verdict"]
                cols = [c for c in cols if c in show.columns]
                show = show[cols].copy().head(50)
                data = [cols] + show.astype(str).values.tolist()
                tbl = Table(data, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 10))

        if shap_payloads:
            story.append(PageBreak())
            story.append(P("SHAP / Explanations", styles["Heading2"]))
            story.append(Spacer(1, 6))
            for key, pld in shap_payloads.items():
                if not pld:
                    continue
                story.append(P(pld.get("title", key), styles["Heading3"]))
                story.append(P(pld.get("summary", ""), normal))
                story.append(Spacer(1, 4))

                png = pld.get("png")
                if png:
                    try:
                        story.append(RLImage(BytesIO(png), width=250 * mm, height=120 * mm))
                        story.append(Spacer(1, 6))
                    except Exception:
                        pass

                top_rows = pld.get("top_rows")
                if isinstance(top_rows, pd.DataFrame) and len(top_rows) > 0:
                    try:
                        show = top_rows.head(30).copy()
                        cols = list(show.columns)
                        data = [cols] + show.astype(str).values.tolist()
                        tbl = Table(data, repeatRows=1)
                        tbl.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("FONTSIZE", (0, 0), (-1, -1), 7),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]))
                        story.append(tbl)
                        story.append(Spacer(1, 8))
                    except Exception:
                        pass

        story.append(PageBreak())
        story.append(P("RAW + Standardized Entries", styles["Heading2"]))
        story.append(Spacer(1, 6))

        primary_cols = list(PRIMARY_COLS)

        for i, row in df_entries.reset_index(drop=True).iterrows():
            title = f"Entry #{i+1} — {row.get('prediction','')}"
            sub = (
                f"ProbAdj={row.get('probability','')}  |  ProbRaw={row.get('probability_raw','')}  |  "
                f"Backend={row.get('inference_backend','')}  |  "
                f"Override={row.get('override_applied','')} ({row.get('override_reason','')})  |  "
                f"OddUsed={row.get('odd_hours_used','')} (TZsrc={row.get('tz_source','')})  |  "
                f"WhitelistHit={row.get('whitelist_hit','')} Mode={row.get('whitelist_mode','')} "
                f"Factor={row.get('whitelist_factor','')} Applied={row.get('whitelist_applied','')}"
            )

            kv = []
            for c in primary_cols:
                kv.append((c, str(row.get(c, ""))))

            grid = []
            for j in range(0, len(kv), 2):
                a = kv[j] if j < len(kv) else ("", "")
                b = kv[j + 1] if j + 1 < len(kv) else ("", "")
                grid.append([P(a[0], tiny), P(a[1], tiny), P(b[0], tiny), P(b[1], tiny)])

            kv_tbl = Table(grid, colWidths=[22 * mm, 88 * mm, 22 * mm, 88 * mm])
            kv_tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
            ]))

            rawlog = str(row.get("raw_log", ""))
            if raw_log_max_chars and raw_log_max_chars > 0 and len(rawlog) > raw_log_max_chars:
                rawlog = rawlog[:raw_log_max_chars]

            block = [
                P(title, styles["Heading3"]),
                P(sub, small),
                Spacer(1, 4),
                kv_tbl,
                Spacer(1, 4),
                P("RAW LOG:", small),
                P(rawlog.replace("\n", "<br/>"), tiny),
                Spacer(1, 10),
            ]
            story.append(KeepTogether(block))

        doc.build(story)
        return buf.getvalue(), filename, ""

    except Exception as e:
        last_err = f"ReportLab failed: {e}"

    # fallback: minimal pdf
    try:
        from fpdf import FPDF  # type: ignore
        pdf = FPDF("P", "mm", "A4")
        pdf.set_auto_page_break(True, margin=12)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.multi_cell(0, 8, safe_pdf_text("CyberSecurity Log Classification Report"), border=1, align="C")
        pdf.ln(2)
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 6, safe_pdf_text(meta.get("summary_line", "")), border=1)
        pdf.ln(2)

        if metrics:
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(0, 6, "Metrics (Labeled)", border=0)
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 6, safe_pdf_text(metrics.get("line", "")), border=1)
            pdf.ln(2)

        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 6, "Entries (top rows)", border=0)
        pdf.set_font("Arial", "", 7)
        for _, row in df_entries.head(200).iterrows():
            line = (
                f"Pred={row.get('prediction','')} ProbAdj={row.get('probability','')} "
                f"Override={row.get('override_applied','')} Domain={row.get('domain','')} "
                f"DestIP={row.get('dest_ip','')} BytesOut={row.get('bytes_out','')}"
            )
            pdf.multi_cell(0, 4, safe_pdf_text(line), border=1)

        out = pdf.output(dest="S").encode("latin-1", errors="replace")
        return out, filename, f"{last_err} (FPDF used)."
    except Exception as e:
        return b"", filename, f"{last_err} | FPDF failed: {e}"

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("✅ Controls")

MODEL_UI = [
    "Decision Tree",
    "Random Forest",
    "Logistic Regression",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Ensemble (excl CatBoost)",
]
if MOE_AVAILABLE:
    MODEL_UI.append("MoE Hybrid (Supervised+ISO)")

model_choice = st.sidebar.selectbox("Select Model", MODEL_UI, key="model_choice")
use_calibrator = st.sidebar.checkbox("Use probability calibrator (if available)", value=True, key="use_calibrator")
carry_over = st.sidebar.checkbox("Carry-over last threshold when unlabeled", value=True, key="carry_over")

with st.sidebar.expander("⚡ ONNX acceleration", expanded=False):
    _ort_ok = ort is not None
    enable_onnx_inference = st.checkbox(
        "Use ONNX Runtime when available",
        value=_ort_ok,
        disabled=not _ort_ok,
        key="enable_onnx_inference",
    )
    enable_onnx_auto_convert = st.checkbox(
        "Auto-convert supported sklearn models to ONNX cache",
        value=bool(_ort_ok and convert_sklearn is not None),
        disabled=not (_ort_ok and convert_sklearn is not None),
        key="enable_onnx_auto_convert",
    )
    enable_onnx_parity_guard = st.checkbox(
        "Parity guard: fallback if ONNX differs from Python",
        value=True,
        disabled=not _ort_ok,
        key="enable_onnx_parity_guard",
    )
    onnx_parity_atol = st.number_input(
        "Max allowed ONNX/Python probability delta",
        min_value=0.0,
        max_value=0.10,
        value=0.00001,
        step=0.00001,
        format="%.5f",
        disabled=not _ort_ok,
        key="onnx_parity_atol",
    )
    onnx_parity_rows = int(st.number_input(
        "Parity sample rows per ONNX model",
        min_value=32,
        max_value=10000,
        value=512,
        step=32,
        disabled=not _ort_ok,
        key="onnx_parity_rows",
    ))
    if _ort_ok:
        st.caption("ONNX Runtime available. Existing .onnx files are preferred; supported sklearn models can be cached as ONNX.")
    else:
        st.caption("ONNX Runtime not installed; Python inference remains active.")
        if ONNX_IMPORT_ERR:
            st.code(ONNX_IMPORT_ERR)
    if convert_sklearn is None and SKL2ONNX_IMPORT_ERR:
        st.caption("Auto-conversion unavailable; existing .onnx files can still be used if onnxruntime is installed.")

ENABLE_ONNX_INFERENCE = bool(enable_onnx_inference and (ort is not None))
ENABLE_ONNX_AUTO_CONVERT = bool(enable_onnx_auto_convert and ENABLE_ONNX_INFERENCE and (convert_sklearn is not None))
ENABLE_ONNX_PARITY_GUARD = bool(enable_onnx_parity_guard and ENABLE_ONNX_INFERENCE)
ONNX_PARITY_ATOL = float(onnx_parity_atol)
ONNX_PARITY_SAMPLE_ROWS = int(onnx_parity_rows)

with st.sidebar.expander("🟩 Whitelist domains (optional)", expanded=False):
    wl_file = st.file_uploader("Upload whitelist CSV", type=["csv"], key="wl_file")
    wl_mode = st.selectbox("Whitelist mode", ["Off", "Soft", "Medium", "Hard", "Custom"], index=0, key="wl_mode")
    wl_custom = st.slider("Custom factor", 0.10, 1.00, 0.70, 0.05, key="wl_custom") if wl_mode == "Custom" else 0.70
WHITELIST = load_whitelist_domains_csv(wl_file)

with st.sidebar.expander("📊 PDF Settings", expanded=False):
    include_shap_pdf = st.checkbox("Include SHAP in PDF (if computed)", value=True, key="pdf_include_shap")
    pdf_max_rows = st.number_input("Max entries in PDF", 1, 2000000, 200, 10, key="pdf_max_rows")
    pdf_raw_max_chars = st.number_input("Max RAW chars per entry (0 = unlimited)", 0, 200000, 0, 1000, key="pdf_raw_max_chars")
    pdf_exfil_topk = st.number_input("Top-K exfil rows (domain/ip) in PDF", 50, 200, 50, 5, key="pdf_exfil_topk")

with st.sidebar.expander("🔔 Alerts", expanded=False):
    enable_alerts = st.checkbox("Enable alerts", value=True, key="enable_alerts")
    alert_threshold_mode = st.selectbox("Alert threshold mode", ["Use main threshold", "Manual"], index=0, key="alert_thr_mode")
    manual_alert_thr = st.number_input("Alert threshold (prob ≥)", 0.0, 1.0, 0.5, 0.01, key="manual_alert_thr") if alert_threshold_mode == "Manual" else None

    email_enable = st.checkbox("Email alerts (optional)", value=False, key="email_enable")
    email_to = st.text_input("To (recipient email)", value="", key="email_to")
    st.caption("SMTP can be supplied via env vars or manually below.")
    smtp_host = st.text_input("SMTP host", value=os.getenv("SMTP_HOST", ""), key="smtp_host")
    smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=int(os.getenv("SMTP_PORT", "587") or "587"), key="smtp_port")
    smtp_user = st.text_input("SMTP username", value=os.getenv("SMTP_USER", ""), key="smtp_user")
    smtp_pass = st.text_input("SMTP password", value=os.getenv("SMTP_PASS", ""), type="password", key="smtp_pass")
    smtp_from = st.text_input("From (email)", value=os.getenv("SMTP_FROM", smtp_user), key="smtp_from")

with st.sidebar.expander("📈 Curves", expanded=False):
    show_curves = st.checkbox("Show ROC/PR curves in dashboard (if labeled)", value=True, key="show_curves")

last_thr = st.session_state.get("last_batch_threshold", None)
force_enabled = st.sidebar.checkbox("Force threshold (unlabeled)", value=False, disabled=(last_thr is None), key="force_thr")
forced_thr = st.sidebar.number_input(
    "Forced threshold", 0.0, 1.0,
    float(last_thr if last_thr is not None else 0.5),
    0.0001,
    format="%.4f",
    disabled=(not force_enabled),
    key="forced_thr",
)

# ============================================================
# Input (streaming for upload)
# ============================================================
input_method = st.sidebar.radio("Input Method", ["Paste Logs", "Upload Log File"], key="input_method")
labels_provided = False
y_true: Optional[np.ndarray] = None

def iter_lines_from_paste(text: str) -> Iterable[str]:
    for ln in (text or "").splitlines():
        if ln.strip():
            yield ln.rstrip("\n")

_STRUCTURED_ROW_PREFIX = "__CYBER_STRUCTURED_ROW_JSON__="
_HTTP_METHOD_SET = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
_IP_LIKE_RE = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")
_HOST_LIKE_RE = re.compile(r"^(?:[a-z0-9-]{1,63}\.)+[a-z]{2,24}$", re.I)
_TS_LIKE_RE = re.compile(r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}/[A-Za-z]{3}/\d{4})")
_UA_LIKE_RE = re.compile(r"(?:mozilla|chrome|safari|edge|firefox|curl|wget|python-requests|bot|crawler|spider|headless|go-http-client)", re.I)
_KNOWN_LOG_FAMILIES = {"dns", "flow", "ids", "waf", "edr", "app", "lb", "apigw", "tls", "email_sec", "cloud_audit", "k8s_audit", "linux_auth", "auditd"}


def _first_nonempty_local(*vals: Any) -> str:
    for v in vals:
        if v is None:
            continue
        try:
            if isinstance(v, float) and np.isnan(v):
                continue
        except Exception:
            pass
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null", "", "notprovided", "unknown", "-", "--"}:
            return s
    return ""


def _nonempty_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _ratio(s: pd.Series, pred) -> float:
    vals = _nonempty_str_series(s)
    vals = vals[vals.ne("")]
    if len(vals) == 0:
        return 0.0
    try:
        return float(vals.map(pred).mean())
    except Exception:
        return 0.0


def _column_header_quality(cols: List[Any]) -> int:
    score = 0
    known = {c.lower() for c in PRIMARY_COLS} | {"timestamp", "time", "date", "datetime", "event_time", "log_time", "source", "destination", "source_ip", "destination_ip", "src", "dst", "src_ip", "dst_ip", "clientip", "destip", "client_ip", "dest_ip", "user_agent", "useragent", "ua", "host", "hostname", "domain", "url", "uri", "path", "url_path", "raw_log", "raw", "line", "text", "method", "verb", "status", "status_code", "bytes_out", "bytes_in", "bytes_sent", "bytes_received", "username", "user", "account", "workstation", "process", "proc", "command", "cmd", "log_type", "type"}
    for c in cols:
        cs = str(c).strip()
        low = re.sub(r"[^a-z0-9]+", "_", cs.lower()).strip("_")
        if not cs or cs.lower().startswith("unnamed") or re.fullmatch(r"column\s*\d+", cs, flags=re.I):
            continue
        if low in known:
            score += 3
        elif re.search(r"[A-Za-z]", cs) and not _IP_LIKE_RE.match(cs) and not _TS_LIKE_RE.search(cs):
            score += 1
    return score



def _header_cell_looks_like_data(c: Any) -> bool:
    cs = str(c).strip()
    if not cs:
        return True
    low_norm = re.sub(r"[^a-z0-9]+", "_", cs.lower()).strip("_")
    known = {c.lower() for c in PRIMARY_COLS} | {"id", "row_id", "source_row_id", "date", "time", "timestamp", "datetime", "source_ip", "destination_ip", "src_ip", "dst_ip", "domain", "host", "referrer", "referer", "user_agent", "ua", "status", "bytes", "bytes_out", "bytes_in", "method", "url", "uri", "path", "event_id", "outcome", "action", "process", "command", "workstation", "username", "log_type"}
    if low_norm in known:
        return False
    if cs.lower().startswith("unnamed") or re.fullmatch(r"column\s*\d+", cs, flags=re.I):
        return True
    if cs.upper() in _HTTP_METHOD_SET or _IP_LIKE_RE.match(cs) or _TS_LIKE_RE.search(cs):
        return True
    if _HOST_LIKE_RE.match(cs.split(":", 1)[0].lower()) or cs.lower().startswith(("http://", "https://")) or cs.startswith("/"):
        return True
    if _UA_LIKE_RE.search(cs) or re.fullmatch(r"\d+(?:\.\d+)?", cs) or cs.upper() in {"ALLOW", "DENY", "BLOCK", "ACCEPT", "REJECT", "RENEW", "ASSIGN", "N/A", "-"}:
        return True
    if re.search(r"[A-Z]:\\|/bin/|/usr/|\.exe\b|^[0-9A-Fa-f]{8,}$", cs):
        return True
    return False


def _headers_look_like_data(cols: List[Any]) -> bool:
    if not cols:
        return False
    data_votes = sum(1 for c in cols if _header_cell_looks_like_data(c))
    quality = _column_header_quality(cols)
    return data_votes >= max(2, int(0.60 * len(cols))) and quality <= 3


def _col_vals(df: pd.DataFrame, i: int) -> pd.Series:
    return _nonempty_str_series(df.iloc[:, i]) if i < df.shape[1] else pd.Series([], dtype=str)


def _col_ratio(df: pd.DataFrame, i: int, pred) -> float:
    if i >= df.shape[1]:
        return 0.0
    return _ratio(_col_vals(df, i), pred)


def _time_like(x: Any) -> bool:
    return bool(re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?", str(x).strip()))


def _date_like(x: Any) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", str(x).strip()))


def _event_id_like(x: Any) -> bool:
    try:
        v = int(float(str(x).strip()))
        return 100 <= v <= 99999
    except Exception:
        return False


def _infer_table_schema_from_values(df: pd.DataFrame) -> Optional[List[str]]:
    """Infer whole-row schemas for headerless cybersecurity CSV/XLSX inputs.

    Column-by-column inference is not enough because the first numeric column is
    often a row id, and DHCP/firewall rows split date/time. This function covers
    the main log families before falling back to per-column heuristics.
    """
    if df is None or df.empty or df.shape[1] <= 1:
        return None
    n = int(df.shape[1])
    schema: Optional[List[str]] = None
    ip = lambda i: _col_ratio(df, i, lambda x: bool(_IP_LIKE_RE.match(str(x))))
    ts = lambda i: _col_ratio(df, i, lambda x: bool(_TS_LIKE_RE.search(str(x))) or pd.notna(pd.to_datetime(x, errors="coerce")))
    date = lambda i: _col_ratio(df, i, _date_like)
    time = lambda i: _col_ratio(df, i, _time_like)
    method = lambda i: _col_ratio(df, i, lambda x: str(x).upper() in _HTTP_METHOD_SET)
    host = lambda i: _col_ratio(df, i, lambda x: bool(_HOST_LIKE_RE.match(str(x).split(":", 1)[0].lower())) and not str(x).lower().startswith(("http://", "https://")))
    url = lambda i: _col_ratio(df, i, lambda x: str(x).lower().startswith(("http://", "https://")))
    path = lambda i: _col_ratio(df, i, lambda x: str(x).startswith("/"))
    ua = lambda i: _col_ratio(df, i, lambda x: bool(_UA_LIKE_RE.search(str(x))) or (len(str(x)) > 35 and " " in str(x)))
    num = lambda i: float(pd.to_numeric(_col_vals(df, i), errors="coerce").notna().mean()) if i < n and len(_col_vals(df, i)) else 0.0
    action = lambda i, words: _col_ratio(df, i, lambda x: str(x).upper() in words)
    eventid = lambda i: _col_ratio(df, i, _event_id_like)

    # web CSV: timestamp, client_ip, dest_ip, method, path, referrer, status, bytes_out, bytes_in, ua, username
    if n >= 10 and ts(0) >= 0.55 and ip(1) >= 0.55 and ip(2) >= 0.45 and method(3) >= 0.55:
        schema = ["timestamp", "client_ip", "dest_ip", "method", "url_path", "referrer", "status", "bytes_out", "bytes_in", "user_agent"] + (["username"] if n >= 11 else [])
    # process CSV: timestamp, username, workstation, process, command, dest_ip
    elif n >= 6 and ts(0) >= 0.55 and ip(5) >= 0.40 and num(0) < 0.80:
        schema = ["timestamp", "username", "workstation", "process", "command", "dest_ip"]
    # proxy: id, timestamp, client_ip, domain, referrer, user_agent, bytes_out, bytes_in(optional)
    elif n >= 7 and ts(1) >= 0.55 and ip(2) >= 0.55 and host(3) >= 0.40 and (host(4) >= 0.30 or url(4) >= 0.30) and ua(5) >= 0.30:
        schema = ["source_row_id", "timestamp", "client_ip", "domain", "referrer", "user_agent", "bytes_out"] + (["bytes_in"] if n >= 8 else [])
    # windows event: id, timestamp, workstation, event_id, outcome, command
    elif n >= 6 and ts(1) >= 0.55 and eventid(3) >= 0.55:
        schema = ["source_row_id", "timestamp", "workstation", "event_id", "outcome", "command"]
    # dhcp/asset: id, date, time, Renew/Assign, ip, workstation, mac
    elif n >= 7 and date(1) >= 0.55 and time(2) >= 0.55 and action(3, {"RENEW", "ASSIGN", "RELEASE", "DHCPACK", "DHCPREQUEST", "DHCPOFFER", "DHCPDISCOVER"}) >= 0.45 and ip(4) >= 0.55:
        schema = ["source_row_id", "date", "time", "outcome", "client_ip", "workstation", "mac_address"]
    # firewall: id, date, time, src, dst, ALLOW/DENY/BLOCK, bytes_out
    elif n >= 7 and date(1) >= 0.55 and time(2) >= 0.55 and ip(3) >= 0.55 and ip(4) >= 0.55 and action(5, {"ALLOW", "DENY", "BLOCK", "ACCEPT", "REJECT", "DROP"}) >= 0.45:
        schema = ["source_row_id", "date", "time", "client_ip", "dest_ip", "outcome", "bytes_out"]
    # windows logon/dynamic: id, timestamp, workstation, username/workstation, dest_ip, event_id, logon_id, outcome
    elif n >= 8 and ts(1) >= 0.55 and ip(4) >= 0.45 and eventid(5) >= 0.55:
        schema = ["source_row_id", "timestamp", "workstation", "username", "dest_ip", "event_id", "logon_id", "outcome"]

    if schema is None:
        return None
    if len(schema) < n:
        schema = schema + [f"extra_field_{i+1}" for i in range(len(schema), n)]
    return schema[:n]


def _is_generic_source_col(name: Any) -> bool:
    s = str(name).strip()
    return bool(re.fullmatch(r"(?i)(?:input[_ ]?)?(?:column|unnamed|extra[_ ]field)[_ ]?\d+", s) or re.fullmatch(r"\d+", s))

def _infer_column_name_from_values(col: pd.Series, pos: int, used: set) -> str:
    vals = _nonempty_str_series(col)
    ip_r = _ratio(vals, lambda x: bool(_IP_LIKE_RE.match(str(x))))
    ts_r = _ratio(vals, lambda x: bool(_TS_LIKE_RE.search(str(x))) or pd.notna(pd.to_datetime(x, errors="coerce")))
    date_r = _ratio(vals, _date_like)
    time_r = _ratio(vals, _time_like)
    method_r = _ratio(vals, lambda x: str(x).upper() in _HTTP_METHOD_SET)
    url_r = _ratio(vals, lambda x: str(x).lower().startswith(("http://", "https://")))
    path_r = _ratio(vals, lambda x: str(x).startswith("/"))
    ua_r = _ratio(vals, lambda x: bool(_UA_LIKE_RE.search(str(x))) or (len(str(x)) > 35 and " " in str(x)))
    host_r = _ratio(vals, lambda x: bool(_HOST_LIKE_RE.match(str(x).split(":", 1)[0].lower())) and not str(x).lower().startswith(("http://", "https://")))
    fam_r = _ratio(vals, lambda x: str(x).lower() in _KNOWN_LOG_FAMILIES)
    nums = pd.to_numeric(vals, errors="coerce")
    num_r = float(nums.notna().mean()) if len(vals) else 0.0
    status_r = float((nums.dropna().between(100, 599)).mean()) if nums.notna().any() else 0.0

    preferred = None
    if pos == 0 and num_r >= 0.75 and "source_row_id" not in used:
        preferred = "source_row_id"
    elif date_r >= 0.60 and "date" not in used:
        preferred = "date"
    elif time_r >= 0.60 and "time" not in used:
        preferred = "time"
    elif ts_r >= 0.55:
        preferred = "timestamp"
    elif ip_r >= 0.55:
        preferred = "client_ip" if "client_ip" not in used else "dest_ip"
    elif method_r >= 0.55:
        preferred = "method"
    elif url_r >= 0.50:
        preferred = "full_url"
    elif path_r >= 0.50:
        preferred = "url_path"
    elif status_r >= 0.65 and "status" not in used:
        preferred = "status"
    elif ua_r >= 0.35:
        preferred = "user_agent"
    elif host_r >= 0.45:
        preferred = "domain"
    elif fam_r >= 0.50:
        preferred = "log_type"
    elif num_r >= 0.75:
        preferred = "bytes_out" if "bytes_out" not in used else "bytes_in"

    if preferred and preferred not in used:
        return preferred
    return PRIMARY_COLS[pos] if pos < len(PRIMARY_COLS) else f"extra_field_{pos + 1}"


def _normalize_uploaded_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve real headers and infer semantic names for headerless uploads."""
    out = df.copy().dropna(how="all")
    schema = _infer_table_schema_from_values(out)
    if schema is not None and (all(_is_generic_source_col(c) for c in out.columns) or _headers_look_like_data(list(out.columns))):
        out.columns = schema
        return out.fillna("").astype(str)

    used: set = set()
    new_cols: List[str] = []
    for i, c in enumerate(out.columns):
        raw = str(c).strip()
        low = raw.lower()
        generated = _is_generic_source_col(raw) or (not raw) or low.startswith("unnamed") or raw.isdigit()
        value_header = bool(_header_cell_looks_like_data(raw) and _column_header_quality([raw]) == 0)
        if generated or value_header:
            name = _infer_column_name_from_values(out.iloc[:, i], i, used)
        else:
            name = raw
        if name in used:
            base = name
            k = 2
            while f"{base}_{k}" in used:
                k += 1
            name = f"{base}_{k}"
        used.add(name)
        new_cols.append(name)
    out.columns = new_cols
    return out.fillna("").astype(str)


_RAW_LOG_FAST_RE = re.compile(
    r"""(?ix)
    ^\s*(?:
        # Apache / nginx access log
        \d{1,3}(?:\.\d{1,3}){3}\s+\S+\s+\S+\s+\[|
        # syslog / linux auth
        [A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+|
        # auditd
        type=\w+\s+msg=audit\(|
        # key-value family logs: 2025-..Z dns client=...
        \d{4}-\d{2}-\d{2}T\S+\s+[A-Za-z_][A-Za-z0-9_.-]*\s+\w+=|
        # JSON anomaly rows
        \{\s*"
    )
    """
)
_SPACE_STYLE_WEB_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}\s+\d{1,3}(?:\.\d{1,3}){3}\s+(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+/",
    re.IGNORECASE,
)


def _preview_text_lines(data: bytes, max_lines: int = 80, max_bytes: int = 512_000) -> List[str]:
    text = data[:max_bytes].decode("utf-8", errors="ignore")
    return [ln for ln in text.splitlines() if ln.strip()][:max_lines]


def _looks_like_raw_log_line_fast(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if _RAW_LOG_FAST_RE.search(s) or _SPACE_STYLE_WEB_RE.search(s):
        return True
    first = s.split(",", 1)[0]
    return bool(_SPACE_STYLE_WEB_RE.search(first))


def _csv_rows_fast(data: bytes, delimiter: str = ",") -> List[List[str]]:
    text = data.decode("utf-8", errors="ignore")
    rows: List[List[str]] = []
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        for row in reader:
            if row and any(str(x).strip() for x in row):
                rows.append(["" if x is None else str(x) for x in row])
    except Exception:
        return []
    return rows


def _rows_to_df_fast(rows: List[List[str]]) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    counts = [len(r) for r in rows[:80] if r]
    if not counts or max(counts) <= 1:
        return None

    vals, freqs = np.unique(np.asarray(counts, dtype=int), return_counts=True)
    mode_width = int(vals[int(np.argmax(freqs))])
    mode_frac = float(np.max(freqs)) / max(len(counts), 1)
    if mode_frac < 0.70:
        return None

    first = rows[0]
    first_quality = _column_header_quality(first)
    first_is_data = _headers_look_like_data(first)
    has_header = bool(first_quality >= 2 and not first_is_data)

    width = mode_width
    data_rows = rows[1:] if has_header else rows
    if not data_rows:
        return None

    norm_rows: List[List[str]] = []
    for r in data_rows:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            r = r[: width - 1] + [",".join(r[width - 1:])]
        norm_rows.append(r)

    if has_header:
        cols = [str(c).strip() or f"Column {i+1}" for i, c in enumerate(first[:width])]
        if len(cols) < width:
            cols += [f"Column {i+1}" for i in range(len(cols), width)]
    else:
        cols = [f"Column {i+1}" for i in range(width)]

    try:
        df = pd.DataFrame(norm_rows, columns=cols).fillna("").astype(str)
    except Exception:
        return None
    return _normalize_uploaded_table_columns(df)


_LARGE_STRUCTURED_UPLOAD_BYTES = int(os.getenv("CYBER_STRUCTURED_STREAM_THRESHOLD_MB", "64")) * 1024 * 1024


def _infer_csv_layout_from_preview(rows: List[List[str]]) -> Optional[Tuple[bool, int, List[str]]]:
    """Infer CSV/TSV width/header/semantic columns from a small preview."""
    if not rows:
        return None
    counts = [len(r) for r in rows[:100] if r]
    if not counts or max(counts) <= 1:
        return None
    vals, freqs = np.unique(np.asarray(counts, dtype=int), return_counts=True)
    width = int(vals[int(np.argmax(freqs))])
    if width <= 1:
        return None
    mode_frac = float(np.max(freqs)) / max(len(counts), 1)
    if mode_frac < 0.55:
        return None

    first = rows[0]
    first_quality = _column_header_quality(first)
    first_is_data = _headers_look_like_data(first)
    has_header = bool(first_quality >= 2 and not first_is_data)
    data_rows = rows[1:] if has_header else rows
    if not data_rows:
        return None

    norm_rows: List[List[str]] = []
    for r in data_rows[:250]:
        if len(r) < width:
            rr = r + [""] * (width - len(r))
        elif len(r) > width:
            rr = r[: width - 1] + [",".join(r[width - 1:])]
        else:
            rr = r
        norm_rows.append(rr)

    if has_header:
        cols = [str(c).strip() or f"Column {i+1}" for i, c in enumerate(first[:width])]
        if len(cols) < width:
            cols += [f"Column {i+1}" for i in range(len(cols), width)]
    else:
        cols = [f"Column {i+1}" for i in range(width)]

    try:
        preview_df = pd.DataFrame(norm_rows, columns=cols).fillna("").astype(str)
        preview_df = _normalize_uploaded_table_columns(preview_df)
        columns = list(preview_df.columns)
    except Exception:
        columns = cols
    if len(columns) < width:
        columns += [f"extra_field_{i+1}" for i in range(len(columns), width)]
    return has_header, width, columns[:width]


def _iter_streamed_structured_csv(file_obj) -> Optional[Iterable[str]]:
    """Return a generator for large CSV/TSV uploads without loading all bytes."""
    name = str(getattr(file_obj, "name", "")).lower()
    ext = os.path.splitext(name)[1]
    if ext not in {".csv", ".tsv"}:
        return None
    delim = "\t" if ext == ".tsv" else ","
    size = int(getattr(file_obj, "size", 0) or 0)
    stream_small = os.getenv("CYBER_STREAM_ALL_TABLE_UPLOADS", "0") == "1"
    if size and size < _LARGE_STRUCTURED_UPLOAD_BYTES and not stream_small:
        return None
    try:
        file_obj.seek(0)
        sample = file_obj.read(512_000)
        file_obj.seek(0)
    except Exception:
        return None
    sample_b = sample.encode("utf-8", errors="ignore") if isinstance(sample, str) else bytes(sample or b"")
    preview = _preview_text_lines(sample_b)
    if not preview:
        return None
    raw_hits = sum(1 for ln in preview if _looks_like_raw_log_line_fast(ln))
    if raw_hits >= max(3, int(0.50 * len(preview))):
        return None
    rows = _csv_rows_fast(sample_b, delimiter=delim)
    layout = _infer_csv_layout_from_preview(rows)
    if layout is None:
        return None
    has_header, width, columns = layout

    def _gen() -> Iterable[str]:
        try:
            file_obj.seek(0)
        except Exception:
            pass
        # Avoid file_obj.getvalue(): stream decode row-by-row for 1GB uploads.
        wrapper = io.TextIOWrapper(file_obj, encoding="utf-8", errors="ignore", newline="")
        reader = csv.reader(wrapper, delimiter=delim)
        if has_header:
            try:
                next(reader, None)
            except Exception:
                pass
        for row in reader:
            if not row or not any(str(x).strip() for x in row):
                continue
            if len(row) < width:
                row = row + [""] * (width - len(row))
            elif len(row) > width:
                row = row[: width - 1] + [",".join(row[width - 1:])]
            obj = {columns[i]: ("" if row[i] is None else str(row[i])) for i in range(width)}
            yield _STRUCTURED_ROW_PREFIX + json.dumps(obj, ensure_ascii=False, default=str)

    return _gen()


def _read_upload_as_table(file_obj) -> Optional[pd.DataFrame]:
    """Fast, conservative structured-upload reader.

    Avoid calling pandas' slow delimiter sniffer on raw mixed-log corpora.  Only
    table-parse when the upload has a stable tabular shape; otherwise stream raw
    lines unchanged into the universal log parser.
    """
    name = str(getattr(file_obj, "name", "")).lower()
    ext = os.path.splitext(name)[1]
    if ext not in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson"}:
        return None
    # Large CSV/TSV is handled by _iter_streamed_structured_csv() so we never
    # materialize a 1GB upload just to detect a table schema.
    size = int(getattr(file_obj, "size", 0) or 0)
    if ext in {".csv", ".tsv"} and size >= _LARGE_STRUCTURED_UPLOAD_BYTES:
        return None
    try:
        file_obj.seek(0)
    except Exception:
        pass
    data = file_obj.getvalue()

    try:
        if ext in {".xlsx", ".xls"}:
            df = pd.read_excel(BytesIO(data), dtype=str).fillna("")
        elif ext in {".jsonl", ".ndjson"}:
            df = pd.read_json(BytesIO(data), lines=True, dtype=False).fillna("").astype(str)
        elif ext == ".json":
            df = pd.read_json(BytesIO(data), dtype=False).fillna("").astype(str)
        else:
            preview = _preview_text_lines(data)
            if not preview:
                return None
            raw_hits = sum(1 for ln in preview if _looks_like_raw_log_line_fast(ln))
            if raw_hits >= max(3, int(0.50 * len(preview))):
                return None
            delim = "\t" if ext == ".tsv" else ","
            rows = _csv_rows_fast(data, delimiter=delim)
            df = _rows_to_df_fast(rows)
            if df is None:
                return None
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = _normalize_uploaded_table_columns(df)
    if df.shape[1] <= 1 and str(df.columns[0]).strip().lower() not in {"raw", "raw_log", "line", "text"}:
        return None
    return df

def _safe_input_col_name(col: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(col).strip()).strip("_")
    return s or "field"


def _table_to_json_lines(df: pd.DataFrame) -> Iterable[str]:
    df = _normalize_uploaded_table_columns(df)
    for _, row in df.iterrows():
        obj = {}
        for k, v in row.items():
            if pd.isna(v):
                obj[str(k)] = ""
            elif isinstance(v, (pd.Timestamp, datetime)):
                obj[str(k)] = str(v)
            else:
                obj[str(k)] = v.item() if hasattr(v, "item") else v
        yield _STRUCTURED_ROW_PREFIX + json.dumps(obj, ensure_ascii=False, default=str)


def iter_lines_from_upload(file_obj) -> Iterable[str]:
    streamed_table = _iter_streamed_structured_csv(file_obj)
    if streamed_table is not None:
        yield from streamed_table
        return

    df = _read_upload_as_table(file_obj)
    if df is not None:
        yield from _table_to_json_lines(df)
        return

    try:
        file_obj.seek(0)
    except Exception:
        pass
    # Stream raw logs directly from the uploaded file.  This avoids duplicating
    # large uploads in memory with file_obj.getvalue().
    while True:
        try:
            raw = file_obj.readline()
        except Exception:
            break
        if not raw:
            break
        if isinstance(raw, bytes):
            ln = raw.decode("utf-8", errors="ignore")
        else:
            ln = str(raw)
        if ln.strip():
            yield ln.rstrip("\r\n")


def structured_row_to_record(line: str) -> Dict[str, Any]:
    payload = line[len(_STRUCTURED_ROW_PREFIX):]
    try:
        row = json.loads(payload)
    except Exception:
        return {"raw_log": payload, "log_type": "structured_upload"}
    if not isinstance(row, dict):
        return {"raw_log": str(row), "log_type": "structured_upload"}

    row = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items()}
    original_cols = list(row.keys())
    one = pd.DataFrame([row])
    one = canonicalize_columns(one)
    row_can = one.iloc[0].to_dict()

    # Recompose split date/time schemas before parsing/copying canonical fields.
    # canonicalize_columns maps a column literally named "time" to timestamp, so
    # prefer the original normalized row's date+time pair when present.
    d0 = _first_nonempty_local(row.get("date"), row_can.get("date"))
    t0 = _first_nonempty_local(row.get("time"), row_can.get("time"))
    if d0 and t0:
        row_can["timestamp"] = f"{d0} {t0}"
    elif not _first_nonempty_local(row_can.get("timestamp")):
        d = _first_nonempty_local(row_can.get("date"), row.get("date"))
        t = _first_nonempty_local(row_can.get("time"), row.get("time"))
        if d and t:
            row_can["timestamp"] = f"{d} {t}"

    raw_log = _first_nonempty_local(row_can.get("raw_log"), row_can.get("raw"), row_can.get("line"), row_can.get("text"))
    if not raw_log:
        raw_log = " ".join(f"{k}={row.get(k, '')}" for k in original_cols if str(row.get(k, '')).strip())

    try:
        parsed = _parse_line(raw_log) or {}
    except Exception:
        parsed = {}
    rec: Dict[str, Any] = dict(parsed)
    rec["raw_log"] = raw_log

    def _infer_structured_type() -> str:
        lt = _first_nonempty_local(row_can.get("log_type"), rec.get("log_type"))
        if lt and lt != "futuristic_unknown":
            return lt
        outc = str(row_can.get("outcome", "")).upper()
        if outc in {"RENEW", "ASSIGN", "RELEASE", "DHCPACK", "DHCPREQUEST", "DHCPOFFER", "DHCPDISCOVER"}:
            return "dhcp"
        if outc in {"ALLOW", "DENY", "BLOCK", "ACCEPT", "REJECT", "DROP"} and row_can.get("client_ip") and row_can.get("dest_ip"):
            return "firewall"
        if row_can.get("domain") and row_can.get("user_agent") and row_can.get("bytes_out"):
            return "proxy"
        if row_can.get("event_id") and row_can.get("workstation"):
            return "windows_event"
        if row_can.get("process") and row_can.get("command"):
            return "endpoint_process"
        return "structured_upload"

    rec["log_type"] = _infer_structured_type()

    canonical_names = set(PRIMARY_COLS) | {"timestamp_raw", "source_row_id", "src_port", "dst_port", "proto", "outcome", "severity", "event_id", "logon_id", "mac_address"}
    for k, v in row_can.items():
        if k in canonical_names and str(v).strip():
            rec[k] = v

    for k in original_cols:
        if _is_generic_source_col(k):
            continue
        safe = "input_" + _safe_input_col_name(k)
        if safe not in rec:
            rec[safe] = row.get(k, "")
    rec["input_source_columns"] = ",".join([c for c in original_cols if not _is_generic_source_col(c)])
    return rec


logs_iter: Optional[Iterable[str]] = None
n_logs_hint: Optional[int] = None

if input_method == "Paste Logs":
    log_text = st.text_area("📝 Paste Logs (one per line):", height=220, key="paste_logs")
    logs_iter = iter_lines_from_paste(log_text)
    logs_list = [ln for ln in (log_text or "").splitlines() if ln.strip()]
    n_logs_hint = len(logs_list)

    label_text = st.text_area("Optional labels (comma-separated 0/1):", height=70, key="paste_labels")
    if label_text.strip():
        labs = [int(x.strip()) for x in label_text.split(",") if x.strip() in ("0", "1")]
        if len(labs) != n_logs_hint:
            st.error("❌ Label count mismatch with logs.")
            st.stop()
        labels_provided = True
        y_true = np.array(labs, dtype=np.int8)
else:
    up_file = st.file_uploader("📂 Upload log file", type=None, key="upload_log")

    lab_file = st.file_uploader("📌 Optional labels (.csv/.xlsx)", type=["csv", "xlsx"], key="upload_labels")

    if up_file:
        logs_iter = iter_lines_from_upload(up_file)

    if lab_file:
        df_lab = pd.read_csv(lab_file) if lab_file.name.endswith(".csv") else pd.read_excel(lab_file)
        labs = df_lab.iloc[:, -1].dropna().astype(int).tolist()
        labels_provided = True
        y_true = np.array(labs, dtype=np.int8)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ============================================================
# Helpers
# ============================================================
def parse_lines_to_df(lines: List[str]) -> pd.DataFrame:
    parsed: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            if isinstance(ln, str) and ln.startswith(_STRUCTURED_ROW_PREFIX):
                d = structured_row_to_record(ln)
            else:
                d = _parse_line(ln)
            if d is None:
                d = {"raw_log": ln, "log_type": "futuristic_unknown"}
        except Exception:
            d = {"raw_log": ln, "log_type": "futuristic_unknown"}
        d["raw_log"] = d.get("raw_log", ln)
        parsed.append(d)
    df_raw = pd.DataFrame(parsed)

    for c in PRIMARY_COLS:
        if c not in df_raw.columns:
            df_raw[c] = np.nan
    if "url_path" not in df_raw.columns:
        df_raw["url_path"] = np.nan
    if "raw_log" not in df_raw.columns:
        df_raw["raw_log"] = lines

    df_raw = canonicalize_columns(df_raw)
    df_raw["timestamp_raw"] = df_raw.get("timestamp", "").fillna("").astype(str)
    return df_raw


def add_wall_time_and_odd(df_raw: pd.DataFrame, default_tz_min_non_suspicious: Optional[int] = None) -> pd.DataFrame:
    """Add wall-time + tz offset + odd-hours columns (FAST, hardened).

    Core behavior:
      - Local wall-time is derived from the log timestamp (explicit offset wins; otherwise infer).
      - UTC is only used as a *fallback* basis later when tz_force_utc==1 (bot/automation/VPN/proxy/tunnel-ish).
      - No .dt crashes: wall_dt is forced to datetime64[ns] before using .dt.

    Output columns (always present):
      wall_dt, tz_offset_min, tz_source, timestamp_parse_ok,
      odd_hours_local, local_hour_fraction, timestamp_local_str,
      parsed_timestamp_utc, timestamp_utc_str, odd_hours_utc,
      local_hour, utc_hour, local_dow, utc_dow,
      is_weekend_local, is_weekend_utc
    """
    df = df_raw.copy()

    # ---- defaults ----
    if default_tz_min_non_suspicious is None:
        # If user provides a sidebar setting later, we can set DEFAULT_TZ_MIN_NON_SUSPICIOUS globally.
        default_tz_min_non_suspicious = int(globals().get("DEFAULT_TZ_MIN_NON_SUSPICIOUS", 0) or 0)

    ws_s  = df.get("workstation", pd.Series("", index=df.index)).fillna("").astype(str)
    raw_s = df.get("raw_log", pd.Series("", index=df.index)).fillna("").astype(str)
    ts_s  = df.get("timestamp_raw", df.get("timestamp", pd.Series("", index=df.index))).fillna("").astype(str)

    # ---- candidate timestamp: prefer timestamp_raw, else extract from raw only when missing (vectorized) ----
    cand = ts_s.astype(str).str.strip().str.strip("[]")
    miss = cand.eq("") | cand.str.lower().isin({"nan", "none", "null", "-", "--", "notprovided", "unknown"})
    if miss.any():
        sub = raw_s.loc[miss]
        found = pd.Series("", index=sub.index, dtype=object)

        # order matters: more specific first
        extract_pats = [
            re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[ ]?[+-]\d{2}:?\d{2}))"),
            re.compile(r"(\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})"),
            re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})"),
            re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"),
            re.compile(r"(?:audit\()?\s*(\d{10}|\d{13})(?:\.\d+)?(?:\))?"),
            re.compile(r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}(?:\.\d+)?(?:\s*(?:AM|PM))?)", re.IGNORECASE),
        ]
        for rx in extract_pats:
            m = sub.str.extract(rx, expand=False).fillna("").astype(str)
            found = found.mask(found.eq(""), m)

        cand.loc[miss] = found

    cand = cand.fillna("").astype(str).str.strip().str.strip("[]")

    suspicious_hint = raw_s.str.contains(TZ_FORCE_UTC_RE, na=False)

    n = len(df)
    wall = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    off  = pd.Series(np.zeros(n, dtype=np.int32), index=df.index, dtype="int32")
    ok   = pd.Series(np.zeros(n, dtype=np.int8), index=df.index, dtype="int8")
    tz_source = pd.Series(["unknown"] * n, index=df.index, dtype=object)

    # ---- epoch (10/13 digits) ----
    m10 = cand.str.fullmatch(r"\d{10}", na=False)
    m13 = cand.str.fullmatch(r"\d{13}", na=False)
    if m10.any():
        t = pd.to_numeric(cand.loc[m10], errors="coerce")
        dt_utc = pd.to_datetime(t, unit="s", utc=True, errors="coerce")
        wall.loc[m10] = dt_utc.dt.tz_localize(None)
        off.loc[m10] = 0
        ok.loc[m10] = dt_utc.notna().astype("int8")
        tz_source.loc[m10] = "epoch"

    if m13.any():
        t = pd.to_numeric(cand.loc[m13], errors="coerce")
        dt_utc = pd.to_datetime(t, unit="ms", utc=True, errors="coerce")
        wall.loc[m13] = dt_utc.dt.tz_localize(None)
        off.loc[m13] = 0
        ok.loc[m13] = dt_utc.notna().astype("int8")
        tz_source.loc[m13] = "epoch"

    # ---- apache: 10/Oct/2000:13:55:36 +0530 ----
    m_ap = wall.isna() & cand.str.match(r"^\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4}$", na=False)
    if m_ap.any():
        ext = cand.loc[m_ap].str.extract(r"^(?P<dt>\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2})\s+(?P<tz>[+-]\d{4})$", expand=True)
        dt_part = ext["dt"].fillna("")
        tz_part = ext["tz"].fillna("")
        dt = pd.to_datetime(dt_part, format="%d/%b/%Y:%H:%M:%S", errors="coerce")
        dt = pd.to_datetime(dt, errors="coerce")
        wall.loc[m_ap] = dt
        # tz to minutes
        tok = tz_part.str.replace(":", "", regex=False)
        sign = np.where(tok.str.startswith("-"), -1, 1)
        hh = pd.to_numeric(tok.str[1:3], errors="coerce").fillna(0).astype(int)
        mm = pd.to_numeric(tok.str[3:5], errors="coerce").fillna(0).astype(int)
        off.loc[m_ap] = (sign * (hh * 60 + mm)).astype("int32").to_numpy()
        ok.loc[m_ap] = dt.notna().astype("int8")
        tz_source.loc[m_ap] = "explicit_offset"

    # ---- YYYY-MM-DD HH:MM:SS +0530 ----
    m_ymd_off = wall.isna() & cand.str.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}$", na=False)
    if m_ymd_off.any():
        ext = cand.loc[m_ymd_off].str.extract(r"^(?P<dt>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(?P<tz>[+-]\d{4})$", expand=True)
        dt_part = ext["dt"].fillna("")
        tz_part = ext["tz"].fillna("")
        dt = pd.to_datetime(dt_part, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        dt = pd.to_datetime(dt, errors="coerce")
        wall.loc[m_ymd_off] = dt
        tok = tz_part.str.replace(":", "", regex=False)
        sign = np.where(tok.str.startswith("-"), -1, 1)
        hh = pd.to_numeric(tok.str[1:3], errors="coerce").fillna(0).astype(int)
        mm = pd.to_numeric(tok.str[3:5], errors="coerce").fillna(0).astype(int)
        off.loc[m_ymd_off] = (sign * (hh * 60 + mm)).astype("int32").to_numpy()
        ok.loc[m_ymd_off] = dt.notna().astype("int8")
        tz_source.loc[m_ymd_off] = "explicit_offset"

    # ---- ISO-like with Z/+hh:mm/+hhmm token somewhere ----
    m_iso_tz = wall.isna() & cand.str.contains(r"(Z|[+-]\d{2}:?\d{2})\b", na=False)
    if m_iso_tz.any():
        # dt part (remove trailing tz token)
        dt_part = cand.loc[m_iso_tz].str.replace(r"(Z|[+-]\d{2}:?\d{2})\b", "", regex=True).str.strip()
        dt = pd.to_datetime(dt_part, errors="coerce")
        dt = pd.to_datetime(dt, errors="coerce")
        good = dt.notna()
        if good.any():
            idx = dt.index[good]
            wall.loc[idx] = dt.loc[idx]
            ok.loc[idx] = 1

        # offset minutes from token
        tok = cand.loc[m_iso_tz].str.extract(r"(Z|[+-]\d{2}:?\d{2})\b", expand=False).fillna("")
        tok_u = tok.astype(str).str.strip().str.upper()
        is_z = tok_u.eq("Z")
        tok_u = tok_u.mask(is_z, "+0000").str.replace(":", "", regex=False)
        sign = np.where(tok_u.str.startswith("-"), -1, 1)
        hh = pd.to_numeric(tok_u.str[1:3], errors="coerce").fillna(0).astype(int)
        mm = pd.to_numeric(tok_u.str[3:5], errors="coerce").fillna(0).astype(int)
        off.loc[m_iso_tz] = (sign * (hh * 60 + mm)).astype("int32").to_numpy()
        z_idx = tok_u.index[is_z.to_numpy(dtype=bool)]
        if len(z_idx):
            off.loc[z_idx] = 0
        tz_source.loc[m_iso_tz] = "explicit_offset"

    # ---- remaining naive formats (fast paths) ----
    rem = wall.isna()
    if rem.any():
        dt = pd.to_datetime(cand.loc[rem], errors="coerce")
        dt = pd.to_datetime(dt, errors="coerce")
        good = dt.notna()
        if good.any():
            idx = dt.index[good]
            wall.loc[idx] = dt.loc[idx]
            ok.loc[idx] = 1
            # keep tz_source as unknown/no_tz for now
            tz_source.loc[idx] = "no_tz"

    # ---- infer offset where not explicit ----
    need_off = wall.notna() & (tz_source.isin(["unknown", "no_tz"]))
    if need_off.any():
        # tz abbr in raw (only when NOT suspicious)
        m_abbr = need_off & (~suspicious_hint)
        if m_abbr.any():
            ab = raw_s.loc[m_abbr].str.extract(TZ_ABBR_RE, expand=False).fillna("").astype(str).str.upper()
            off_ab = ab.map(TZ_ABBR_OFFSETS_MIN)
            have = off_ab.notna()
            if have.any():
                idx = off_ab.index[have]
                off.loc[idx] = off_ab.loc[idx].astype("int32")
                tz_source.loc[idx] = "tz_abbr"

        # workstation inference (only when NOT suspicious)
        m_ws = need_off & (~suspicious_hint) & (tz_source.isin(["unknown", "no_tz"]))
        if m_ws.any():
            codes, uniq = pd.factorize(ws_s.loc[m_ws], sort=False)
            offs = np.fromiter(
                (infer_site_offset_min(u) if infer_site_offset_min(u) is not None else np.nan for u in uniq),
                dtype=float,
                count=len(uniq),
            )
            ws_off = offs[codes]
            have = ~np.isnan(ws_off)
            if have.any():
                idx = ws_s.loc[m_ws].index[have]
                off.loc[idx] = ws_off[have].astype(np.int32)
                tz_source.loc[idx] = "workstation"

        # default tz for naive timestamps (only when NOT suspicious)
        m_def = need_off & (~suspicious_hint) & (tz_source.isin(["unknown", "no_tz"]))
        if m_def.any():
            off.loc[m_def] = int(default_tz_min_non_suspicious)
            tz_source.loc[m_def] = "default"

        # suspicious + unknown => force UTC
        m_susp = need_off & suspicious_hint & (tz_source.isin(["unknown", "no_tz"]))
        if m_susp.any():
            off.loc[m_susp] = 0
            tz_source.loc[m_susp] = "suspicious_force_utc"

        # remaining: assume UTC
        m_left = need_off & (tz_source.isin(["unknown", "no_tz"]))
        if m_left.any():
            off.loc[m_left] = 0
            tz_source.loc[m_left] = "unknown_assumed"

    # ---- HARDEN: ensure wall is datetimelike (prevents .dt crash) ----
    wall = pd.to_datetime(wall, errors="coerce")
    fail = wall.isna()
    if fail.any():
        fallback_now = pd.Timestamp.now().floor("s")
        wall = wall.fillna(fallback_now)
        ok.loc[fail] = 0
        off.loc[fail] = 0
        tz_source.loc[fail] = "fallback_now"

    # ---- derive local ----
    hf = wall.dt.hour.astype(float) + wall.dt.minute.astype(float) / 60.0
    odd_local = ((hf >= 23.0) | (hf < 5.5)).astype(np.int8)
    ts_local_str = wall.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    local_hour = wall.dt.hour.fillna(0).astype(np.int16)
    local_dow = wall.dt.dayofweek.fillna(0).astype(np.int16)
    is_weekend_local = (local_dow >= 5).astype(np.int8)

    # ---- derive UTC using offset ----
    utc_naive = wall - pd.to_timedelta(off.astype("int64"), unit="m")
    fallback_now = pd.Timestamp.now().floor("s")
    utc_naive = pd.to_datetime(utc_naive, errors="coerce").fillna(fallback_now)
    ts_utc_str = utc_naive.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    if getattr(utc_naive.dt, "tz", None) is not None:
        utc = utc_naive.dt.tz_convert("UTC")
    else:
        utc = utc_naive.dt.tz_localize("UTC")

    hf_utc = utc.dt.hour.astype(float) + utc.dt.minute.astype(float) / 60.0
    odd_utc = ((hf_utc >= 23.0) | (hf_utc < 5.5)).astype(np.int8)

    utc_hour = utc.dt.hour.fillna(0).astype(np.int16)
    utc_dow = utc.dt.dayofweek.fillna(0).astype(np.int16)
    is_weekend_utc = (utc_dow >= 5).astype(np.int8)

    # ---- write ----
    df["wall_dt"] = wall
    df["tz_offset_min"] = off.astype("int32")
    df["tz_source"] = tz_source.astype(object)
    df["timestamp_parse_ok"] = ok.astype("int8")

    df["odd_hours_local"] = odd_local.astype("int8")
    df["local_hour_fraction"] = hf.astype(float)
    df["timestamp_local_str"] = ts_local_str.astype(str)

    df["parsed_timestamp_utc"] = utc
    df["timestamp_utc_str"] = ts_utc_str.astype(str)
    df["odd_hours_utc"] = odd_utc.astype("int8")

    df["local_hour"] = local_hour.astype("int16")
    df["utc_hour"] = utc_hour.astype("int16")
    df["local_dow"] = local_dow.astype("int16")
    df["utc_dow"] = utc_dow.astype("int16")
    df["is_weekend_local"] = is_weekend_local.astype("int8")
    df["is_weekend_utc"] = is_weekend_utc.astype("int8")

    return df
def pick_shap_model_name() -> Optional[str]:
    for nm in ["LightGBM", "XGBoost", "Random Forest", "Decision Tree"]:
        try:
            mdl, _, _ = load_supervised(nm)
            if mdl is not None:
                return nm
        except Exception:
            continue
    return None

def _norm_ip_text(x: Any) -> str:
    s = "" if x is None else str(x).strip().strip("[]")
    s = s.replace("[.]", ".")
    if s.lower() in {"", "nan", "none", "null", "notprovided", "unknown", "0", "0.0.0.0", "::"}:
        return s
    return s


def _ip_kind_for_semantics(ip_text: str) -> str:
    """Return local/public/unknown for SHAP explanation semantics."""
    ip_text = _norm_ip_text(ip_text)
    if not ip_text or ip_text.lower() in {"nan", "none", "null", "notprovided", "unknown"}:
        return "unknown"
    try:
        obj = ipaddress.ip_address(ip_text)
    except Exception:
        return "unknown"
    if obj.is_private or obj.is_loopback or obj.is_link_local or obj.is_multicast or obj.is_reserved or obj.is_unspecified:
        return "local"
    if getattr(obj, "is_global", False):
        return "public"
    return "unknown"


def _ip_bad_rep_semantic_state(row: pd.Series, bad_ips: Optional[set] = None) -> Tuple[str, str]:
    """Semantic SHAP state for ip_bad_rep.

    Desired display contract:
      - bad IP list hit => upward
      - local/private/special-only IP evidence => neutral
      - clean public IP evidence => downward
    """
    bad_ips = {str(x).strip().replace("[.]", ".") for x in (bad_ips or set()) if str(x).strip()}
    ips = [_norm_ip_text(row.get("client_ip", "")), _norm_ip_text(row.get("dest_ip", ""))]

    try:
        if float(row.get("ip_bad_truth", 0) or 0) > 0:
            return "bad", "bad_ip_list hit → upward"
    except Exception:
        pass

    for ip in ips:
        if ip and ip in bad_ips:
            return "bad", "bad_ip_list hit → upward"

    kinds = [_ip_kind_for_semantics(ip) for ip in ips if ip]
    has_public = any(k == "public" for k in kinds)
    has_local = any(k == "local" for k in kinds)

    if has_public:
        return "clean", "clean public IP not in bad_ip_list → downward"
    if has_local:
        return "local", "local/private IP → neutral"
    return "neutral", "no valid public/bad IP evidence → neutral"


def _attach_ip_bad_rep_semantics(res: Dict[str, Any], df_imp_local: pd.DataFrame, row_indices: List[int]) -> Dict[str, Any]:
    out = dict(res)
    states: List[str] = []
    notes: List[str] = []
    for idx in row_indices:
        try:
            row = df_imp_local.iloc[int(idx)]
            state, note = _ip_bad_rep_semantic_state(row, globals().get("BAD_IPS", set()))
        except Exception:
            state, note = "neutral", "no valid public/bad IP evidence → neutral"
        states.append(state)
        notes.append(note)
    out["_ip_bad_rep_states"] = states
    out["_ip_bad_rep_notes"] = notes
    return out


def compute_shap_rows(engine, mdl, X_df, row_indices: List[int], df_imp_local: pd.DataFrame, topk: int = 60):
    try:
        res = engine.compute_for_rows(
            mdl,
            X_df,
            row_indices=row_indices,
            ip_bad_truth=df_imp_local.get("ip_bad_truth", None),
            ip_private_truth=df_imp_local.get("ip_private_truth", None),
            topk=topk,
            semantic_ip=True
        )
    except TypeError:
        res = engine.compute_for_rows(mdl, X_df, row_indices=row_indices, topk=topk)
    return _attach_ip_bad_rep_semantics(res, df_imp_local, row_indices)


def _breakdown_from_vec(vec: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    mag = np.abs(arr)
    total = float(mag.sum())
    if total <= 1e-12:
        return {"pos_pct": 0.0, "neg_pct": 0.0, "neutral_pct": 100.0, "net_pct": 0.0}
    pos = float(mag[arr > 1e-12].sum())
    neg = float(mag[arr < -1e-12].sum())
    neu = float(mag[(arr >= -1e-12) & (arr <= 1e-12)].sum())
    return {"pos_pct": 100.0 * pos / total, "neg_pct": 100.0 * neg / total, "neutral_pct": 100.0 * neu / total, "net_pct": 100.0 * (pos - neg) / total}


def _apply_ip_bad_rep_semantic_shap(sv_rows: np.ndarray, feat_names: List[str], res: Dict[str, Any]) -> Tuple[np.ndarray, Optional[int], str]:
    arr = np.asarray(sv_rows, dtype=float).copy()
    ip_idx = None
    for i, name in enumerate(feat_names):
        if str(name) == "ip_bad_rep":
            ip_idx = i
            break
    if ip_idx is None:
        return arr, None, ""

    states = list(res.get("_ip_bad_rep_states", []))
    notes = list(res.get("_ip_bad_rep_notes", []))
    if not states:
        states = ["neutral"] * len(arr)
        notes = ["no valid public/bad IP evidence → neutral"] * len(arr)

    for r in range(len(arr)):
        state = states[r] if r < len(states) else states[-1]
        mag = abs(float(arr[r, ip_idx]))
        if state == "bad":
            arr[r, ip_idx] = mag
        elif state == "clean":
            arr[r, ip_idx] = -mag
        else:
            # local/private/unknown evidence should not be shown as pushing risk.
            arr[r, ip_idx] = 0.0

    if len(states) == 1:
        note = notes[0] if notes else "ip_bad_rep semantic direction applied"
    else:
        from collections import Counter
        c = Counter(states)
        note = "ip_bad_rep semantic mix: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
    return arr, ip_idx, note


def build_shap_payload(title: str, res: Dict[str, Any], which: str, model_name: str) -> Dict[str, Any]:
    sv_rows = np.asarray(res["sv_rows"], dtype=float)
    feat_names = list(res["feat_names"])
    feat_disp = [("odd_hours_used" if n == "odd_hours" else n) for n in feat_names]

    sv_sem, ip_idx, ip_note = _apply_ip_bad_rep_semantic_shap(sv_rows, feat_names, res)

    if which == "row":
        v_dir = sv_sem[0]
        v_mag = np.abs(sv_sem[0])
    else:
        v_dir = sv_sem.mean(axis=0)
        v_mag = np.abs(sv_sem).mean(axis=0)

    br = _breakdown_from_vec(v_dir)

    total = float(v_mag.sum()) + 1e-12
    pct = 100.0 * (v_mag / total)
    order = list(np.argsort(-pct)[: min(30, len(pct))])
    # Keep ip_bad_rep visible in the table/chart even when local/private IPs
    # correctly make it neutral with zero contribution.
    if ip_idx is not None and ip_idx not in order:
        order.append(int(ip_idx))

    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [f"{feat_disp[i]} {_arrow(float(v_dir[i]))}" for i in order]
    ax.barh(labels[::-1], pct[order][::-1])
    ax.set_xlabel("Contribution % (sum=100)")
    ax.set_title(title)
    plt.tight_layout()
    png = _fig_to_png(fig)

    semantic_notes = []
    for i in order:
        if ip_idx is not None and i == ip_idx:
            semantic_notes.append(ip_note)
        else:
            semantic_notes.append("")

    top_rows = pd.DataFrame({
        "feature": [feat_disp[i] for i in order],
        "direction": [_arrow(float(v_dir[i])) for i in order],
        "shap_mean": [float(v_dir[i]) for i in order],
        "shap_abs_mean": [float(v_mag[i]) for i in order],
        "contribution_pct": [float(pct[i]) for i in order],
        "semantic_note": semantic_notes,
    })

    summary = (
        f"model={model_name} · "
        f"pos={float(br.get('pos_pct', 0)):.2f}% · "
        f"neg={float(br.get('neg_pct', 0)):.2f}% · "
        f"neutral={float(br.get('neutral_pct', 0)):.2f}% · "
        f"net={float(br.get('net_pct', 0)):.2f}%"
    )
    return {"title": title, "summary": summary, "png": png, "top_rows": top_rows}

# ============================================================
# Main classify
# ============================================================
if ui_btn("🚀 Classify Logs", type="primary", key="btn_classify"):
    if logs_iter is None:
        st.error("❌ No logs provided.")
        st.stop()

    for k in list(st.session_state.keys()):
        if k.startswith("shap_") or k.startswith("pdf_") or k in (
            "df_imp_all", "results_df_full", "results_df_preview",
            "prob_raw", "prob_used", "predictions", "combined_primary",
            "threshold", "threshold_type", "used_model_name", "prob_source",
            "exfil_domain", "exfil_ip"
        ):
            st.session_state.pop(k, None)

    # Accumulators (best-effort)
    preview_frames: List[pd.DataFrame] = []
    df_imp_chunks: List[pd.DataFrame] = []  # used for exfil + PDF; can be large if huge dataset

    prob_raw_all_list: List[np.ndarray] = []
    prob_adj_all_list: List[np.ndarray] = []
    combined_primary_all_list: List[np.ndarray] = []

    total_rows = 0

    with st.spinner("Processing logs (chunked)…"):
        buf_lines: List[str] = []
        for ln in logs_iter:
            buf_lines.append(ln)
            if len(buf_lines) < int(chunk_size):
                continue

            df_raw = add_wall_time_and_odd(parse_lines_to_df(buf_lines))
            df_imp = IMPUTER.impute_df(df_raw)

            # preserve time columns
            for c in ["wall_dt", "tz_offset_min", "tz_source", "timestamp_parse_ok",
                      "odd_hours_local", "odd_hours_utc",
                      "local_hour", "utc_hour", "local_dow", "utc_dow", "is_weekend_local", "is_weekend_utc",
                      "local_hour_fraction", "timestamp_local_str", "timestamp_utc_str", "parsed_timestamp_utc"]:
                if c in df_raw.columns and c not in df_imp.columns:
                    df_imp[c] = df_raw[c].values

            # ipv6 primary
            df_imp = add_ipv6_primary_conditions(df_imp, raw_col="raw_log")
            df_imp["ipv6_primary_flag"] = pd.to_numeric(df_imp.get("ipv6_primary_flag", 0), errors="coerce").fillna(0).astype(np.int8)

            # primary_flag
            if callable(compute_primary_flags):
                try:
                    df_imp["primary_flag"] = compute_primary_flags(df_imp, BUNDLE).astype(np.int8)
                except Exception:
                    df_imp["primary_flag"] = np.int8(0)
            else:
                df_imp["primary_flag"] = np.int8(0)

            # ip_bad_truth + ip_private_truth
            cip = df_imp.get("client_ip", pd.Series("", index=df_imp.index)).fillna("").astype(str).str.replace("[.]", ".", regex=False)
            dip = df_imp.get("dest_ip", pd.Series("", index=df_imp.index)).fillna("").astype(str).str.replace("[.]", ".", regex=False)
            df_imp["ip_bad_truth"] = ((cip.isin(BAD_IPS)) | (dip.isin(BAD_IPS))).astype(np.int8)

            def _is_private(ip: str) -> int:
                try:
                    if not ip:
                        return 0
                    x = ipaddress.ip_address(ip)
                    return 1 if (x.is_private or x.is_loopback or x.is_link_local) else 0
                except Exception:
                    return 0
            df_imp["ip_private_truth"] = (cip.map(_is_private) | dip.map(_is_private)).astype(np.int8)


            # ---- Time basis: UTC only for bot/automation/VPN/proxy/tunnel-ish logs ----
            raw_hint_base = df_imp.get("raw_log", "").fillna("").astype(str).str.contains(TZ_FORCE_UTC_RE, na=False).to_numpy(dtype=bool)
            proxy_hint = df_imp.get("raw_log", "").fillna("").astype(str).str.contains(PROXY_HINT_RE, na=False).to_numpy(dtype=bool)
            ua_hint_bot = df_imp.get("user_agent", "").fillna("").astype(str).str.contains(BOT_UA_RE, na=False).to_numpy(dtype=bool)
            cmd_hint_auto = df_imp.get("command", "").fillna("").astype(str).str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
            proc_hint_auto = df_imp.get("process", "").fillna("").astype(str).str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
            tunnel_hint = df_imp.get("raw_log", "").fillna("").astype(str).str.contains(TUNNEL_TEXT_RE, na=False).to_numpy(dtype=bool)
            ipv6_hint = (pd.to_numeric(df_imp.get("ipv6_primary_flag", 0), errors="coerce").fillna(0).to_numpy(dtype=int) > 0) | \
                        df_imp.get("raw_log", "").fillna("").astype(str).str.contains(IPV6_TUNNEL_ADDR_HINT_RE, na=False).to_numpy(dtype=bool)

            # Proxy alone is *not* enough to force UTC; require at least one more automation/tunnel signal.
            tz_force_utc = (
                raw_hint_base
                | ua_hint_bot
                | cmd_hint_auto
                | proc_hint_auto
                | tunnel_hint
                | ipv6_hint
                | (proxy_hint & (ua_hint_bot | cmd_hint_auto | proc_hint_auto | tunnel_hint | ipv6_hint))
            )
            df_imp["tz_force_utc"] = tz_force_utc.astype(np.int8)

            # Used timestamp string for ALL time-derived features (peak_hour/weekend/hour/etc.)
            local_ts = df_imp.get("timestamp_local_str", df_imp.get("timestamp", "")).fillna("").astype(str)
            utc_ts = df_imp.get("timestamp_utc_str", "")
            if isinstance(utc_ts, pd.Series):
                utc_ts = utc_ts.fillna("").astype(str)
            else:
                # derive from parsed UTC if needed
                utc_ts = pd.to_datetime(df_imp.get("parsed_timestamp_utc", pd.NaT), utc=True, errors="coerce").dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
            df_imp["timestamp_used_str"] = np.where(tz_force_utc, utc_ts, local_ts).astype(str)

            # Keep canonical timestamp aligned with model basis (does not destroy timestamp_raw)
            df_imp["timestamp"] = df_imp["timestamp_used_str"].astype(str)

            # Odd-hours used (local by default, UTC only when tz_force_utc==1)
            odd_local_arr = pd.to_numeric(df_imp.get("odd_hours_local", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)
            odd_utc_arr = pd.to_numeric(df_imp.get("odd_hours_utc", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)
            odd_used = np.where(tz_force_utc, odd_utc_arr, odd_local_arr).astype(np.int8)
            df_imp["odd_hours_used"] = odd_used.astype(np.int8)
            df_imp["odd_used_is_utc"] = tz_force_utc.astype(np.int8)

            # Extra derived time columns (used-basis) for dashboard/PDF audit
            lh = pd.to_numeric(df_imp.get("local_hour", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int16)
            uh = pd.to_numeric(df_imp.get("utc_hour", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int16)
            ld = pd.to_numeric(df_imp.get("local_dow", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int16)
            ud = pd.to_numeric(df_imp.get("utc_dow", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int16)
            wkl = pd.to_numeric(df_imp.get("is_weekend_local", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)
            wku = pd.to_numeric(df_imp.get("is_weekend_utc", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)

            df_imp["hour_used"] = np.where(tz_force_utc, uh, lh).astype(np.int16)
            df_imp["hour"] = df_imp["hour_used"].astype(np.int16)
            df_imp["dow_used"] = np.where(tz_force_utc, ud, ld).astype(np.int16)
            df_imp["is_weekend_used"] = np.where(tz_force_utc, wku, wkl).astype(np.int8)

            # Feature engineering (force LOCAL wall time)
            df_fe = df_imp.copy()
            df_fe["timestamp"] = df_imp.get("timestamp_used_str", df_imp.get("timestamp_local_str", df_imp.get("timestamp", ""))).astype(str)
            if "raw" not in df_fe.columns:
                df_fe["raw"] = df_fe.get("raw_log", "").astype(str)
            X = _run_feature_engineering(df_fe).copy()
            # odd_hours basis already computed (local by default; UTC only when tz_force_utc==1)
            X["odd_hours"] = odd_used.astype(int)

            # whitelist hit (needed for conf debug; safe even if whitelist empty)
            df_imp["whitelist_hit"] = whitelist_hit_series(df_imp.get("domain", pd.Series("", index=df_imp.index)), WHITELIST).astype(np.int8)

            # confidential debug (FIX: this is the reason your flags were always 0)
            conf_bool, conf_code, conf_dbg = compute_confidential_primary_debug(df_imp, X=X, odd_used=odd_used)
            df_imp["confidential_primary_flag"] = conf_bool.astype(np.int8)
            df_imp["conf_reason_code"] = conf_code.astype(np.int8)
            for kdbg, vdbg in conf_dbg.items():
                df_imp[kdbg] = np.asarray(vdbg).astype(np.int8)

            combined_primary = (
                df_imp["primary_flag"].astype(int).to_numpy()
                | df_imp["ipv6_primary_flag"].astype(int).to_numpy()
                | df_imp["confidential_primary_flag"].astype(int).to_numpy()
            ).astype(int)

            suspicious_ctx = compute_suspicious_context(df_imp, X, combined_primary, odd_used)

            # Keep UTC only for bot/automation/VPN/proxy/tunnel-ish logs (tz_force_utc)
            df_imp["suspicious_context"] = suspicious_ctx.astype(np.int8)
            df_imp["odd_hours_used"] = odd_used.astype(np.int8)
            df_imp["odd_used_is_utc"] = pd.to_numeric(df_imp.get("tz_force_utc", 0), errors="coerce").fillna(0).astype(np.int8)
            X["odd_hours"] = odd_used.astype(int)

            # scoring
            X_scaled, scaler_feats, _ = prepare_model_matrix(X, SCALER, BUNDLE_FEATURE_COLS)

            prob_raw = None
            thr_saved = None
            used_model_name = None
            prob_source = None
            fast_primary_all = (
                bool(globals().get("FAST_PRIMARY_ALL", True))
                and len(df_imp) <= int(globals().get("FAST_PRIMARY_MAX_ROWS", 5000) or 5000)
                and len(combined_primary) > 0
                and bool(np.all(np.asarray(combined_primary, dtype=int) > 0))
            )
            if fast_primary_all:
                # If every row is already decided by high-confidence primary rules,
                # loading supervised/MoE artifacts adds latency without changing the
                # final verdict.  Keep probability explicit and auditable.
                prob_raw = np.full(len(df_imp), 0.999, dtype=float)
                thr_saved = None
                used_model_name = "Primary override fast path"
                prob_source = "Primary rules only; ML skipped because all rows are overridden"

            if prob_raw is None and model_choice == "MoE Hybrid (Supervised+ISO)" and MOE_AVAILABLE:
                try:
                    moe_meta = _fast_joblib_load(MOE_META_PATH)
                    moe_cols = _fast_joblib_load(MOE_FEAT_PATH)
                    moe_exps = _fast_joblib_load(MOE_EXP_PATH)
                    iso = _fast_joblib_load(ISO_PATH)
                    thr_saved = float(_fast_joblib_load(MOE_THR_PATH))

                    X_df_scaled = pd.DataFrame(np.asarray(X_scaled), columns=list(scaler_feats), index=df_imp.index)

                    parts = []
                    moe_srcs: List[str] = []
                    for nm in moe_exps:
                        mdl, cal, _thr = load_supervised(nm)
                        p = predict_proba_with_onnx_acceleration(mdl, cal, X_scaled, use_calibrator=True, model_name=str(nm))
                        moe_srcs.append(globals().get("LAST_INFERENCE_SOURCE", ""))
                        parts.append(np.asarray(p, dtype=np.float32).reshape(-1, 1))

                    s = iso.decision_function(X_df_scaled).astype(np.float64)
                    a = -s
                    a = (a - np.min(a)) / (np.ptp(a) + 1e-9)
                    parts.append(a.astype(np.float32).reshape(-1, 1))

                    for kf in ["ipv6_tunnel_any", "whitelist_suspicious_combo", "timestamp_suspicious_tz", "odd_hours"]:
                        if kf in moe_cols:
                            v = pd.to_numeric(X_df_scaled.get(kf, 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
                            parts.append(v)

                    M = np.hstack(parts).astype(np.float32)
                    pmeta = predict_proba_with_onnx_acceleration(moe_meta, None, M, use_calibrator=False, model_name="MoE_meta", role_hint="meta")
                    moe_srcs.append("meta=" + globals().get("LAST_INFERENCE_SOURCE", ""))
                    prob_raw = np.clip(pmeta, 0.0, 1.0)
                    used_model_name = "MoE Hybrid"
                    prob_source = "MoE(meta)+ISO; " + summarize_inference_sources(moe_srcs)
                except Exception:
                    prob_raw = None
                    thr_saved = None

            if prob_raw is None:
                if model_choice == "Ensemble (excl CatBoost)":
                    order = ["LightGBM", "XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"]
                    probs = []
                    thrs = []
                    ensemble_srcs: List[str] = []
                    for nm in order:
                        mdl, cal, thr = load_supervised(nm)
                        if mdl is None:
                            continue
                        try:
                            p = predict_proba_with_onnx_acceleration(mdl, cal, X_scaled, use_calibrator=use_calibrator, model_name=str(nm))
                            ensemble_srcs.append(f"{nm}=" + globals().get("LAST_INFERENCE_SOURCE", ""))
                            probs.append(p)
                            if thr is not None:
                                thrs.append(float(thr))
                        except Exception:
                            continue
                    if probs:
                        prob_raw = np.mean(np.vstack(probs), axis=0)
                        thr_saved = float(np.mean(thrs)) if thrs else None
                        used_model_name = "Ensemble(excl CatBoost)"
                        prob_source = "Ensemble mean probability; " + summarize_inference_sources(ensemble_srcs)
                else:
                    mdl, cal, thr = load_supervised(model_choice)
                    if mdl is not None:
                        try:
                            prob_raw = predict_proba_with_onnx_acceleration(mdl, cal, X_scaled, use_calibrator=use_calibrator, model_name=str(model_choice))
                            thr_saved = float(thr) if thr is not None else None
                            used_model_name = model_choice
                            prob_source = "Model probability; " + globals().get("LAST_INFERENCE_SOURCE", "")
                        except Exception:
                            prob_raw = None

            if prob_raw is None:
                fw = BUNDLE_FEATURE_WEIGHTS
                cols = [c for c in fw.keys() if c in X.columns]
                if cols:
                    w = np.array([float(fw[c]) for c in cols], dtype=float)
                    ss = (X[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy() @ w).astype(float)
                    ss0 = ss - np.nanmin(ss)
                    prob_raw = ss0 / (np.nanmax(ss0) + 1e-9) if np.nanmax(ss0) > 0 else np.zeros(len(df_imp), dtype=float)
                else:
                    prob_raw = np.zeros(len(df_imp), dtype=float)
                used_model_name = "Fallback"
                prob_source = "Fallback(feature_weights)"
                thr_saved = None

            # whitelist dampening (only if whitelist_hit AND NOT suspicious_context AND info_complete)
            info_complete = compute_info_complete(df_imp)
            prob_adj, wl_hit, wl_factor, wl_applied = apply_whitelist(
                prob_raw,
                df_imp.get("domain", pd.Series("", index=df_imp.index)),
                suspicious_mask=suspicious_ctx,
                whitelist_set=WHITELIST,
                mode=wl_mode,
                custom_factor=(wl_custom if wl_mode == "Custom" else None),
                info_complete=info_complete,
            )
            df_imp["whitelist_hit"] = wl_hit.astype(np.int8)
            df_imp["whitelist_mode"] = wl_mode
            df_imp["whitelist_factor"] = float(wl_factor)
            df_imp["whitelist_applied"] = wl_applied.astype(np.int8)
            df_imp["whitelist_info_complete"] = info_complete.astype(np.int8)

            df_imp["probability_raw"] = np.asarray(prob_raw, dtype=float)
            df_imp["probability"] = np.asarray(prob_adj, dtype=float)
            df_imp["used_model_name"] = str(used_model_name or model_choice)
            df_imp["inference_backend"] = str(prob_source or "")
            df_imp["onnx_accelerated"] = np.int8(1 if "ONNX Runtime" in str(prob_source or "") else 0)

            # store arrays
            prob_raw_all_list.append(np.asarray(prob_raw, dtype=np.float32))
            prob_adj_all_list.append(np.asarray(prob_adj, dtype=np.float32))
            combined_primary_all_list.append(np.asarray(combined_primary, dtype=np.int8))

            # stash chunk for later
            df_imp_chunks.append(df_imp)
            total_rows += len(df_imp)

            # preview
            if sum(len(x) for x in preview_frames) < int(max_preview_rows):
                preview_frames.append(df_imp.copy())

            buf_lines = []

        # last chunk
        if buf_lines:
            # reuse by pushing through same path
            # easiest: put back into an iterator-like single chunk
            df_raw = add_wall_time_and_odd(parse_lines_to_df(buf_lines))
            df_imp = IMPUTER.impute_df(df_raw)
            for c in ["wall_dt", "tz_offset_min", "tz_source", "timestamp_parse_ok",
                      "odd_hours_local", "odd_hours_utc",
                      "local_hour", "utc_hour", "local_dow", "utc_dow", "is_weekend_local", "is_weekend_utc",
                      "local_hour_fraction", "timestamp_local_str", "timestamp_utc_str", "parsed_timestamp_utc"]:
                if c in df_raw.columns and c not in df_imp.columns:
                    df_imp[c] = df_raw[c].values

            df_imp = add_ipv6_primary_conditions(df_imp, raw_col="raw_log")
            df_imp["ipv6_primary_flag"] = pd.to_numeric(df_imp.get("ipv6_primary_flag", 0), errors="coerce").fillna(0).astype(np.int8)

            if callable(compute_primary_flags):
                try:
                    df_imp["primary_flag"] = compute_primary_flags(df_imp, BUNDLE).astype(np.int8)
                except Exception:
                    df_imp["primary_flag"] = np.int8(0)
            else:
                df_imp["primary_flag"] = np.int8(0)

            cip = df_imp.get("client_ip", pd.Series("", index=df_imp.index)).fillna("").astype(str).str.replace("[.]", ".", regex=False)
            dip = df_imp.get("dest_ip", pd.Series("", index=df_imp.index)).fillna("").astype(str).str.replace("[.]", ".", regex=False)
            df_imp["ip_bad_truth"] = ((cip.isin(BAD_IPS)) | (dip.isin(BAD_IPS))).astype(np.int8)

            def _is_private(ip: str) -> int:
                try:
                    if not ip:
                        return 0
                    x = ipaddress.ip_address(ip)
                    return 1 if (x.is_private or x.is_loopback or x.is_link_local) else 0
                except Exception:
                    return 0
            df_imp["ip_private_truth"] = (cip.map(_is_private) | dip.map(_is_private)).astype(np.int8)


            # ---- Time basis: UTC only for bot/automation/VPN/proxy/tunnel-ish logs ----
            raw_hint_base = df_imp.get("raw_log", "").fillna("").astype(str).str.contains(TZ_FORCE_UTC_RE, na=False).to_numpy(dtype=bool)
            proxy_hint = df_imp.get("raw_log", "").fillna("").astype(str).str.contains(PROXY_HINT_RE, na=False).to_numpy(dtype=bool)
            ua_hint_bot = df_imp.get("user_agent", "").fillna("").astype(str).str.contains(BOT_UA_RE, na=False).to_numpy(dtype=bool)
            cmd_hint_auto = df_imp.get("command", "").fillna("").astype(str).str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
            proc_hint_auto = df_imp.get("process", "").fillna("").astype(str).str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
            tunnel_hint = df_imp.get("raw_log", "").fillna("").astype(str).str.contains(TUNNEL_TEXT_RE, na=False).to_numpy(dtype=bool)
            ipv6_hint = (pd.to_numeric(df_imp.get("ipv6_primary_flag", 0), errors="coerce").fillna(0).to_numpy(dtype=int) > 0) | \
                        df_imp.get("raw_log", "").fillna("").astype(str).str.contains(IPV6_TUNNEL_ADDR_HINT_RE, na=False).to_numpy(dtype=bool)

            # Proxy alone is *not* enough to force UTC; require at least one more automation/tunnel signal.
            tz_force_utc = (
                raw_hint_base
                | ua_hint_bot
                | cmd_hint_auto
                | proc_hint_auto
                | tunnel_hint
                | ipv6_hint
                | (proxy_hint & (ua_hint_bot | cmd_hint_auto | proc_hint_auto | tunnel_hint | ipv6_hint))
            )
            df_imp["tz_force_utc"] = tz_force_utc.astype(np.int8)

            # Used timestamp string for ALL time-derived features (peak_hour/weekend/hour/etc.)
            local_ts = df_imp.get("timestamp_local_str", df_imp.get("timestamp", "")).fillna("").astype(str)
            utc_ts = df_imp.get("timestamp_utc_str", "")
            if isinstance(utc_ts, pd.Series):
                utc_ts = utc_ts.fillna("").astype(str)
            else:
                # derive from parsed UTC if needed
                utc_ts = pd.to_datetime(df_imp.get("parsed_timestamp_utc", pd.NaT), utc=True, errors="coerce").dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
            df_imp["timestamp_used_str"] = np.where(tz_force_utc, utc_ts, local_ts).astype(str)

            # Keep canonical timestamp aligned with model basis (does not destroy timestamp_raw)
            df_imp["timestamp"] = df_imp["timestamp_used_str"].astype(str)

            # Odd-hours used (local by default, UTC only when tz_force_utc==1)
            odd_local_arr = pd.to_numeric(df_imp.get("odd_hours_local", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)
            odd_utc_arr = pd.to_numeric(df_imp.get("odd_hours_utc", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)
            odd_used = np.where(tz_force_utc, odd_utc_arr, odd_local_arr).astype(np.int8)
            df_imp["odd_hours_used"] = odd_used.astype(np.int8)
            df_imp["odd_used_is_utc"] = tz_force_utc.astype(np.int8)

            # Extra derived time columns (used-basis) for dashboard/PDF audit
            lh = pd.to_numeric(df_imp.get("local_hour", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int16)
            uh = pd.to_numeric(df_imp.get("utc_hour", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int16)
            ld = pd.to_numeric(df_imp.get("local_dow", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int16)
            ud = pd.to_numeric(df_imp.get("utc_dow", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int16)
            wkl = pd.to_numeric(df_imp.get("is_weekend_local", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)
            wku = pd.to_numeric(df_imp.get("is_weekend_utc", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)

            df_imp["hour_used"] = np.where(tz_force_utc, uh, lh).astype(np.int16)
            df_imp["hour"] = df_imp["hour_used"].astype(np.int16)
            df_imp["dow_used"] = np.where(tz_force_utc, ud, ld).astype(np.int16)
            df_imp["is_weekend_used"] = np.where(tz_force_utc, wku, wkl).astype(np.int8)

            df_fe = df_imp.copy()
            df_fe["timestamp"] = df_imp.get("timestamp_used_str", df_imp.get("timestamp_local_str", df_imp.get("timestamp", ""))).astype(str)
            if "raw" not in df_fe.columns:
                df_fe["raw"] = df_fe.get("raw_log", "").astype(str)
            X = _run_feature_engineering(df_fe).copy()

            # odd_hours basis already computed (local by default; UTC only when tz_force_utc==1)
            X["odd_hours"] = odd_used.astype(int)


            df_imp["whitelist_hit"] = whitelist_hit_series(df_imp.get("domain", pd.Series("", index=df_imp.index)), WHITELIST).astype(np.int8)

            conf_bool, conf_code, conf_dbg = compute_confidential_primary_debug(df_imp, X=X, odd_used=odd_used)
            df_imp["confidential_primary_flag"] = conf_bool.astype(np.int8)
            df_imp["conf_reason_code"] = conf_code.astype(np.int8)
            for kdbg, vdbg in conf_dbg.items():
                df_imp[kdbg] = np.asarray(vdbg).astype(np.int8)

            combined_primary = (
                df_imp["primary_flag"].astype(int).to_numpy()
                | df_imp["ipv6_primary_flag"].astype(int).to_numpy()
                | df_imp["confidential_primary_flag"].astype(int).to_numpy()
            ).astype(int)

            suspicious_ctx = compute_suspicious_context(df_imp, X, combined_primary, odd_used)
            df_imp["suspicious_context"] = suspicious_ctx.astype(np.int8)
            df_imp["odd_hours_used"] = odd_used.astype(np.int8)
            df_imp["odd_used_is_utc"] = pd.to_numeric(df_imp.get("tz_force_utc", 0), errors="coerce").fillna(0).astype(np.int8)
            X["odd_hours"] = odd_used.astype(int)

            X_scaled, scaler_feats, _ = prepare_model_matrix(X, SCALER, BUNDLE_FEATURE_COLS)

            prob_raw = None
            thr_saved = None
            used_model_name = None
            prob_source = None
            fast_primary_all = (
                bool(globals().get("FAST_PRIMARY_ALL", True))
                and len(df_imp) <= int(globals().get("FAST_PRIMARY_MAX_ROWS", 5000) or 5000)
                and len(combined_primary) > 0
                and bool(np.all(np.asarray(combined_primary, dtype=int) > 0))
            )
            if fast_primary_all:
                # If every row is already decided by high-confidence primary rules,
                # loading supervised/MoE artifacts adds latency without changing the
                # final verdict.  Keep probability explicit and auditable.
                prob_raw = np.full(len(df_imp), 0.999, dtype=float)
                thr_saved = None
                used_model_name = "Primary override fast path"
                prob_source = "Primary rules only; ML skipped because all rows are overridden"

            if prob_raw is None and model_choice == "MoE Hybrid (Supervised+ISO)" and MOE_AVAILABLE:
                try:
                    moe_meta = _fast_joblib_load(MOE_META_PATH)
                    moe_cols = _fast_joblib_load(MOE_FEAT_PATH)
                    moe_exps = _fast_joblib_load(MOE_EXP_PATH)
                    iso = _fast_joblib_load(ISO_PATH)
                    thr_saved = float(_fast_joblib_load(MOE_THR_PATH))

                    X_df_scaled = pd.DataFrame(np.asarray(X_scaled), columns=list(scaler_feats), index=df_imp.index)

                    parts = []
                    moe_srcs: List[str] = []
                    for nm in moe_exps:
                        mdl, cal, _thr = load_supervised(nm)
                        p = predict_proba_with_onnx_acceleration(mdl, cal, X_scaled, use_calibrator=True, model_name=str(nm))
                        moe_srcs.append(globals().get("LAST_INFERENCE_SOURCE", ""))
                        parts.append(np.asarray(p, dtype=np.float32).reshape(-1, 1))

                    s_iso = iso.decision_function(X_df_scaled).astype(np.float64)
                    a = -s_iso
                    a = (a - np.min(a)) / (np.ptp(a) + 1e-9)
                    parts.append(a.astype(np.float32).reshape(-1, 1))

                    for kf in ["ipv6_tunnel_any", "whitelist_suspicious_combo", "timestamp_suspicious_tz", "odd_hours"]:
                        if kf in moe_cols:
                            v = pd.to_numeric(X_df_scaled.get(kf, 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
                            parts.append(v)

                    M = np.hstack(parts).astype(np.float32)
                    pmeta = predict_proba_with_onnx_acceleration(moe_meta, None, M, use_calibrator=False, model_name="MoE_meta", role_hint="meta")
                    moe_srcs.append("meta=" + globals().get("LAST_INFERENCE_SOURCE", ""))
                    prob_raw = np.clip(pmeta, 0.0, 1.0)
                    used_model_name = "MoE Hybrid"
                    prob_source = "MoE(meta)+ISO; " + summarize_inference_sources(moe_srcs)
                except Exception:
                    prob_raw = None
                    thr_saved = None

            if prob_raw is None:
                if model_choice == "Ensemble (excl CatBoost)":
                    order = ["LightGBM", "XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"]
                    probs = []
                    thrs = []
                    ensemble_srcs: List[str] = []
                    for nm in order:
                        mdl, cal, thr = load_supervised(nm)
                        if mdl is None:
                            continue
                        try:
                            p = predict_proba_with_onnx_acceleration(mdl, cal, X_scaled, use_calibrator=use_calibrator, model_name=str(nm))
                            ensemble_srcs.append(f"{nm}=" + globals().get("LAST_INFERENCE_SOURCE", ""))
                            probs.append(p)
                            if thr is not None:
                                thrs.append(float(thr))
                        except Exception:
                            continue
                    if probs:
                        prob_raw = np.mean(np.vstack(probs), axis=0)
                        thr_saved = float(np.mean(thrs)) if thrs else None
                        used_model_name = "Ensemble(excl CatBoost)"
                        prob_source = "Ensemble mean probability; " + summarize_inference_sources(ensemble_srcs)
                else:
                    mdl, cal, thr = load_supervised(model_choice)
                    if mdl is not None:
                        try:
                            prob_raw = predict_proba_with_onnx_acceleration(mdl, cal, X_scaled, use_calibrator=use_calibrator, model_name=str(model_choice))
                            thr_saved = float(thr) if thr is not None else None
                            used_model_name = model_choice
                            prob_source = "Model probability; " + globals().get("LAST_INFERENCE_SOURCE", "")
                        except Exception:
                            prob_raw = None

            if prob_raw is None:
                fw = BUNDLE_FEATURE_WEIGHTS
                cols = [c for c in fw.keys() if c in X.columns]
                if cols:
                    w = np.array([float(fw[c]) for c in cols], dtype=float)
                    ss = (X[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy() @ w).astype(float)
                    ss0 = ss - np.nanmin(ss)
                    prob_raw = ss0 / (np.nanmax(ss0) + 1e-9) if np.nanmax(ss0) > 0 else np.zeros(len(df_imp), dtype=float)
                else:
                    prob_raw = np.zeros(len(df_imp), dtype=float)
                used_model_name = "Fallback"
                prob_source = "Fallback(feature_weights)"
                thr_saved = None

            info_complete = compute_info_complete(df_imp)
            prob_adj, wl_hit, wl_factor, wl_applied = apply_whitelist(
                prob_raw,
                df_imp.get("domain", pd.Series("", index=df_imp.index)),
                suspicious_mask=suspicious_ctx,
                whitelist_set=WHITELIST,
                mode=wl_mode,
                custom_factor=(wl_custom if wl_mode == "Custom" else None),
                info_complete=info_complete,
            )

            df_imp["whitelist_hit"] = wl_hit.astype(np.int8)
            df_imp["whitelist_mode"] = wl_mode
            df_imp["whitelist_factor"] = float(wl_factor)
            df_imp["whitelist_applied"] = wl_applied.astype(np.int8)
            df_imp["whitelist_info_complete"] = info_complete.astype(np.int8)
            df_imp["probability_raw"] = np.asarray(prob_raw, dtype=float)
            df_imp["probability"] = np.asarray(prob_adj, dtype=float)
            df_imp["used_model_name"] = str(used_model_name or model_choice)
            df_imp["inference_backend"] = str(prob_source or "")
            df_imp["onnx_accelerated"] = np.int8(1 if "ONNX Runtime" in str(prob_source or "") else 0)

            prob_raw_all_list.append(np.asarray(prob_raw, dtype=np.float32))
            prob_adj_all_list.append(np.asarray(prob_adj, dtype=np.float32))
            combined_primary_all_list.append(np.asarray(combined_primary, dtype=np.int8))
            df_imp_chunks.append(df_imp)
            total_rows += len(df_imp)

            if sum(len(x) for x in preview_frames) < int(max_preview_rows):
                preview_frames.append(df_imp.copy())

    # concat arrays
    prob_raw_all = np.concatenate(prob_raw_all_list, axis=0) if prob_raw_all_list else np.zeros(0, dtype=np.float32)
    prob_adj_all = np.concatenate(prob_adj_all_list, axis=0) if prob_adj_all_list else np.zeros(0, dtype=np.float32)
    combined_primary_all = np.concatenate(combined_primary_all_list, axis=0) if combined_primary_all_list else np.zeros(0, dtype=np.int8)

    df_imp_all = pd.concat(df_imp_chunks, axis=0, ignore_index=True) if df_imp_chunks else pd.DataFrame()

    # threshold
    prev_thr = st.session_state.get("last_batch_threshold", None)
    threshold = None
    threshold_type = "Saved/Default"

    thr_saved_val = None
    try:
        # try get thr from selected model once
        _mdl, _cal, _thr = load_supervised(model_choice)
        thr_saved_val = float(_thr) if _thr is not None else None
    except Exception:
        thr_saved_val = None

    if labels_provided and y_true is not None:
        if len(y_true) != len(df_imp_all):
            st.warning(f"Labels provided but count mismatch: labels={len(y_true)} logs={len(df_imp_all)}. Metrics disabled.")
            labels_provided = False
            y_true = None
        elif len(np.unique(y_true)) == 2:
            try:
                prec_c, rec_c, th = precision_recall_curve(y_true, prob_adj_all)
                th = np.append(th, 1.0)
                f1c = (2 * prec_c * rec_c) / (prec_c + rec_c + 1e-9)
                threshold = float(th[np.argmax(f1c)])
                threshold_type = "Batch(F1)"
            except Exception:
                threshold = float(thr_saved_val) if thr_saved_val is not None else 0.5
                threshold_type = "Saved/Default"
        else:
            threshold = float(thr_saved_val) if thr_saved_val is not None else 0.5
            threshold_type = "Saved/Default"
    else:
        if force_enabled and prev_thr is not None:
            threshold = float(forced_thr)
            threshold_type = "Forced"
        elif carry_over and prev_thr is not None:
            threshold = float(prev_thr)
            threshold_type = "Carryover"
        else:
            threshold = float(thr_saved_val) if thr_saved_val is not None else 0.5
            threshold_type = "Saved/Default"

    if threshold is None:
        threshold = 0.5
    st.session_state["last_batch_threshold"] = float(threshold)

    # final predictions with override
    pred_prob = (prob_adj_all >= float(threshold)).astype(np.int8)
    pred_final = np.where(combined_primary_all > 0, 1, pred_prob).astype(np.int8)

    df_imp_all["override_applied"] = (combined_primary_all > 0).astype(np.int8)

    # ============================================================
    # Binary decision labels only
    # ------------------------------------------------------------
    # Suspicious context is still computed internally for whitelist damping,
    # alerting, and override diagnostics, but it is no longer a displayed
    # classification category.  Every row is classified as exactly one of:
    #   - malicious
    #   - non-malicious / legit
    # ============================================================
    conf_tag_scanner = (pd.to_numeric(df_imp_all.get("conf_tag_scanner", 0), errors="coerce").fillna(0).to_numpy() > 0)
    conf_tag_threat = (pd.to_numeric(df_imp_all.get("conf_tag_threat", 0), errors="coerce").fillna(0).to_numpy() > 0)

    risk_label = np.where(pred_final.astype(int) == 1, "malicious", "non-malicious").astype(object)
    activity_type = np.where(conf_tag_threat, "threat", np.where(conf_tag_scanner, "scanner", "other")).astype(object)
    label_verbose = risk_label.copy()
    prediction_risk = np.where(pred_final.astype(int) == 1, "🚨 Malicious", "✅ Non-Malicious / Legit").astype(object)

    df_imp_all["risk_label"] = risk_label
    df_imp_all["activity_type"] = activity_type
    df_imp_all["label_verbose"] = label_verbose
    df_imp_all["prediction_risk"] = prediction_risk
    df_imp_all["binary_decision_policy"] = "probability_threshold_or_primary_override"

    def _reason_row(p: int, v6: int, c: int) -> str:
        r = []
        if p: r.append("artifact_primary")
        if v6: r.append("ipv6_primary")
        if c: r.append("confidential_primary")
        return "|".join(r) if r else ""

    df_imp_all["override_reason"] = [
        _reason_row(int(p), int(v6), int(c))
        for p, v6, c in zip(
            pd.to_numeric(df_imp_all.get("primary_flag", 0), errors="coerce").fillna(0).astype(int),
            pd.to_numeric(df_imp_all.get("ipv6_primary_flag", 0), errors="coerce").fillna(0).astype(int),
            pd.to_numeric(df_imp_all.get("confidential_primary_flag", 0), errors="coerce").fillna(0).astype(int),
        )
    ]

    df_imp_all["prediction"] = np.where(pred_final == 1, "🚨 Malicious", "✅ Non-Malicious / Legit")
    df_imp_all["threshold"] = float(threshold)
    df_imp_all["threshold_type"] = threshold_type

    # exfil tables
    exfil_domain, exfil_ip = compute_exfil_tables(df_imp_all, pred_final, topk=int(pdf_exfil_topk))

    # build results dfs
    display_cols = [
        "raw_log",
        "timestamp_local_str", "timestamp_utc_str", "timestamp_used_str", "tz_offset_min", "tz_source", "tz_force_utc",
        "local_hour", "utc_hour", "hour_used", "local_dow", "utc_dow", "dow_used", "is_weekend_used",
        "odd_hours_local", "odd_hours_utc", "odd_hours_used", "odd_used_is_utc",
        "primary_flag", "ipv6_primary_flag",
        "confidential_primary_flag", "conf_reason_code",
        "conf_hit_explicit", "conf_hit_critical", "conf_hit_ops", "conf_hit_ipv6_combo", "conf_hit_post_put",
        "conf_hit_404", "conf_hit_wl_combo", "conf_hit_moderate_combo",
        "conf_benign_asserted", "conf_tag_scanner", "conf_tag_threat", "conf_tag_suspicious",
        "override_applied", "override_reason",
        "suspicious_context",
        "risk_label", "activity_type", "label_verbose", "prediction_risk", "binary_decision_policy",
        "whitelist_hit", "whitelist_mode", "whitelist_factor", "whitelist_applied", "whitelist_info_complete",
        "used_model_name", "inference_backend", "onnx_accelerated",
        "probability_raw", "probability", "prediction",
    ] + PRIMARY_COLS
    input_cols = [c for c in df_imp_all.columns if str(c).startswith("input_") and not _is_generic_source_col(str(c).replace("input_", "", 1))]
    display_cols = _dedupe_columns(display_cols + input_cols)

    for c in display_cols:
        if c not in df_imp_all.columns:
            df_imp_all[c] = ""

    results_full = _ensure_unique_df_columns(df_imp_all[display_cols].copy())
    results_preview = pd.concat(preview_frames, axis=0, ignore_index=True) if preview_frames else df_imp_all.head(0)
    results_preview = _ensure_unique_df_columns(results_preview.reindex(columns=display_cols, fill_value="").head(int(max_preview_rows)).copy())

    # store
    st.session_state["df_imp_all"] = df_imp_all
    st.session_state["results_df_full"] = results_full
    st.session_state["results_df_preview"] = results_preview
    st.session_state["predictions"] = pred_final
    st.session_state["prob_raw"] = prob_raw_all
    st.session_state["prob_used"] = prob_adj_all
    st.session_state["combined_primary"] = combined_primary_all
    st.session_state["threshold"] = float(threshold)
    st.session_state["threshold_type"] = threshold_type
    st.session_state["exfil_domain"] = exfil_domain
    st.session_state["exfil_ip"] = exfil_ip

    st.success("✅ Done. Scroll for dashboard + alerts + SHAP + PDF.")

# ============================================================
# Dashboard
# ============================================================
if "results_df_preview" in st.session_state:
    df_imp_all = st.session_state.get("df_imp_all", pd.DataFrame())
    results_preview = st.session_state["results_df_preview"]
    results_full = st.session_state["results_df_full"]

    preds = np.asarray(st.session_state.get("predictions", np.zeros(len(results_full))), dtype=int)
    prob_used = np.asarray(st.session_state.get("prob_used", np.zeros(len(results_full))), dtype=float)
    prob_raw = np.asarray(st.session_state.get("prob_raw", np.zeros(len(results_full))), dtype=float)
    combined_primary = np.asarray(st.session_state.get("combined_primary", np.zeros(len(results_full))), dtype=int)

    thr = float(st.session_state.get("threshold", 0.5))
    thr_type = st.session_state.get("threshold_type", "Saved/Default")

    n_total = int(len(results_full))

    # Binary label (for model metrics)
    n_mal = int(np.sum(preds))

    # Binary breakdown for display
    n_non = int(n_total - n_mal)

    def _pct(x: int) -> float:
        return (100.0 * x / n_total) if n_total > 0 else 0.0

    pct_mal = _pct(n_mal)
    pct_non = _pct(n_non)

    st.markdown(
        f"<div class='block'><b>Summary:</b> "
        f"<span class='bad'>{n_mal} Malicious</span> ({pct_mal:.2f}%) | "
        f"<span class='good'>{n_non} Non-Malicious / Legit</span> ({pct_non:.2f}%) | "
        f"Threshold={thr:.4f} ({thr_type})</div>",
        unsafe_allow_html=True,
    )

    with st.expander("⚡ Inference backend / ONNX status", expanded=False):
        if "inference_backend" in df_imp_all.columns:
            bvc = df_imp_all["inference_backend"].fillna("").astype(str).value_counts().reset_index()
            bvc.columns = ["backend", "rows"]
            ui_df(bvc)
        else:
            st.caption("Inference backend metadata is unavailable for this run.")
        if ONNX_VALIDATION_NOTES:
            notes = pd.DataFrame({"onnx_artifact": list(ONNX_VALIDATION_NOTES.keys()), "note": list(ONNX_VALIDATION_NOTES.values())})
            ui_df(notes.tail(25))

    st.subheader("📋 Results Preview (Full results are downloadable)")
    ui_df(results_preview)

    ui_dl(
        "⬇️ Download FULL Results CSV",
        data=results_full.to_csv(index=False).encode("utf-8", errors="ignore"),
        file_name="cyber_results_full.csv",
        mime="text/csv",
        key="dl_results_full_csv",
    )

    # Debug: confidential reason codes
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("🧭 Override Debug (why flags fired)")
    ui_df(pd.DataFrame({"code": list(CONF_REASON_LEGEND.keys()), "meaning": list(CONF_REASON_LEGEND.values())}))
    if "conf_reason_code" in df_imp_all.columns:
        vc = pd.to_numeric(df_imp_all["conf_reason_code"], errors="coerce").fillna(0).astype(int).value_counts().sort_index()
        dbg = pd.DataFrame({"code": vc.index, "count": vc.values})
        dbg["meaning"] = dbg["code"].map(CONF_REASON_LEGEND)
        ui_df(dbg)

    # Exfil
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("💧 Data Exfiltration (bytes_out) — Top Domain/IP")
    exfil_domain = st.session_state.get("exfil_domain", pd.DataFrame())
    exfil_ip = st.session_state.get("exfil_ip", pd.DataFrame())
    if isinstance(exfil_domain, pd.DataFrame) and len(exfil_domain) > 0:
        st.markdown("**Top Domains by bytes_out**")
        ui_df(exfil_domain)
    else:
        st.info("No domain exfil table available.")
    if isinstance(exfil_ip, pd.DataFrame) and len(exfil_ip) > 0:
        st.markdown("**Top Dest IPs by bytes_out**")
        ui_df(exfil_ip)
    else:
        st.info("No dest_ip exfil table available.")

    # Metrics
    metrics_payload = None
    if labels_provided and y_true is not None and len(np.unique(y_true)) == 2 and len(y_true) == len(preds):
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.subheader("🎯 Evaluation Metrics (Labeled)")

        acc = float(accuracy_score(y_true, preds))
        f1v = float(f1_score(y_true, preds, zero_division=1))
        prec = float(precision_score(y_true, preds, zero_division=1))
        rec = float(recall_score(y_true, preds, zero_division=1))
        roc_auc_val = float(roc_auc_score(y_true, prob_used))
        pr_auc_val = float(average_precision_score(y_true, prob_used))

        st.markdown(
            f"<div class='block'>Accuracy: <b>{acc:.2%}</b> | F1: <b>{f1v:.2%}</b> | "
            f"Precision: <b>{prec:.2%}</b> | Recall: <b>{rec:.2%}</b> | "
            f"ROC-AUC: <b>{roc_auc_val:.2%}</b> | PR-AUC: <b>{pr_auc_val:.2%}</b></div>",
            unsafe_allow_html=True,
        )

        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        ui_df(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        roc_png = None
        pr_png = None

        if show_curves:
            try:
                fpr, tpr, _ = roc_curve(y_true, prob_used)
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.plot(fpr, tpr)
                ax.plot([0, 1], [0, 1])
                ax.set_title(f"ROC Curve (AUC={roc_auc_val:.3f})")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                st.pyplot(fig)
                roc_png = _fig_to_png(fig)
            except Exception:
                roc_png = None

            try:
                p_curve, r_curve, _ = precision_recall_curve(y_true, prob_used)
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.plot(r_curve, p_curve)
                ax.set_title(f"Precision-Recall Curve (AP={pr_auc_val:.3f})")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                st.pyplot(fig)
                pr_png = _fig_to_png(fig)
            except Exception:
                pr_png = None
        else:
            try:
                fpr, tpr, _ = roc_curve(y_true, prob_used)
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.plot(fpr, tpr)
                ax.plot([0, 1], [0, 1])
                ax.set_title(f"ROC Curve (AUC={roc_auc_val:.3f})")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                roc_png = _fig_to_png(fig)
            except Exception:
                roc_png = None
            try:
                p_curve, r_curve, _ = precision_recall_curve(y_true, prob_used)
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.plot(r_curve, p_curve)
                ax.set_title(f"Precision-Recall Curve (AP={pr_auc_val:.3f})")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                pr_png = _fig_to_png(fig)
            except Exception:
                pr_png = None

        metrics_payload = {
            "line": f"Accuracy={acc:.3f} | F1={f1v:.3f} | Precision={prec:.3f} | Recall={rec:.3f} | ROC-AUC={roc_auc_val:.3f} | PR-AUC={pr_auc_val:.3f}",
            "cm": cm.tolist(),
            "roc_png": roc_png,
            "pr_png": pr_png,
        }

    # Alerts
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("🔔 Alerts (Odd-hours uses local unless suspicious)")
    if enable_alerts and len(df_imp_all) > 0:
        odd_used = pd.to_numeric(df_imp_all.get("odd_hours_used", df_imp_all.get("odd_hours_local", 0)), errors="coerce").fillna(0).astype(int).to_numpy().astype(bool)

        if alert_threshold_mode == "Manual" and manual_alert_thr is not None:
            alert_thr = float(manual_alert_thr)
            thr_src = "manual"
        else:
            alert_thr = float(st.session_state.get("threshold", 0.5))
            thr_src = "main_threshold"

        cond = odd_used & ((combined_primary > 0) | (preds == 1) | (prob_used >= alert_thr))
        idx = np.where(cond)[0].tolist()

        st.caption(f"Alert threshold: **{alert_thr:.3f}** (mode: {thr_src}) · triggered={len(idx)}")

        if not idx:
            st.info("No alerts triggered.")
            alerts_df = pd.DataFrame(columns=results_full.columns)
        else:
            alerts_df = results_full.iloc[idx].copy()
            ui_df(alerts_df.head(5000))

        if email_enable and email_to.strip():
            if ui_btn("📧 Send alert email now", type="secondary", key="btn_send_email"):
                if alerts_df.empty:
                    st.warning("No alerts to email.")
                elif not (smtp_host and smtp_user and smtp_pass and smtp_from):
                    st.error("SMTP settings incomplete. Provide SMTP host/user/pass/from.")
                else:
                    try:
                        msg = EmailMessage()
                        msg["Subject"] = f"[Cyber Alerts] {len(alerts_df)} odd-hour alerts"
                        msg["From"] = smtp_from
                        msg["To"] = email_to.strip()
                        lines = [
                            f"Alerts: {len(alerts_df)}",
                            f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
                            "",
                            "Top alerts (idx | prob_adj | prob_raw | odd_used | local_time | domain | log_type):",
                        ]
                        for i0, row in alerts_df.head(80).reset_index(drop=True).iterrows():
                            lines.append(
                                f"- {i0} | {row.get('probability','')} | {row.get('probability_raw','')} | {row.get('odd_hours_used','')} | "
                                f"{row.get('timestamp_local_str','')} | {row.get('domain','')} | {row.get('log_type','')}"
                            )
                        msg.set_content("\n".join(lines))
                        with smtplib.SMTP(smtp_host, int(smtp_port), timeout=20) as s:
                            s.starttls()
                            s.login(smtp_user, smtp_pass)
                            s.send_message(msg)
                        st.success("Email sent.")
                    except Exception as e:
                        st.error(f"Email failed: {e}")

    # SHAP (lazy; no model/feature recomputation unless user explicitly clicks)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("🧠 SHAP (lazy explanation mode)")

    if not bool(globals().get("enable_shap_dashboard", False)):
        st.info("SHAP dashboard is disabled for faster inference. Enable it in Sidebar → Advanced when explanations are needed.")
    else:
        df_shap_base = df_imp_all.head(int(max_preview_rows)).copy()
        n = len(df_shap_base)
        if n == 0:
            st.info("No rows available for SHAP.")
        else:
            st.caption("v7 lazy mode: SHAP models and feature matrices are loaded only after you click a SHAP button.")
            st.session_state.setdefault("sel_row", 0)
            st.session_state["sel_row"] = max(0, min(int(st.session_state["sel_row"]), max(0, n - 1)))

            if n <= 1:
                r = 0
                st.markdown("Only one row: **0**")
            else:
                c1, c2 = st.columns([3, 1])
                with c1:
                    r_slider = st.slider("Row (preview index)", 0, n - 1, int(st.session_state["sel_row"]), 1, key="shap_row_slider")
                with c2:
                    r = int(st.number_input("Go to row", 0, n - 1, int(r_slider), 1, key="shap_row_num"))
                st.session_state["sel_row"] = r

            st.text_area("Raw log (selected)", str(df_shap_base.iloc[r].get("raw_log", "")), height=140, key="raw_sel")
            st.session_state.setdefault("subset_text", "")
            subset_text = st.text_input("Subset rows (comma-separated, supports ranges like 3-10) — preview indices", key="subset_text")

            cA, cB, cC = st.columns(3)
            with cA:
                do_row = ui_btn("Compute Row SHAP", type="secondary", key="btn_shap_row")
            with cB:
                do_subset = ui_btn("Compute Subset Avg SHAP", type="secondary", key="btn_shap_subset")
            with cC:
                do_all = ui_btn("Compute Overall Avg SHAP", type="secondary", key="btn_shap_all")

            if do_row or do_subset or do_all:
                with st.spinner("Preparing SHAP only now…"):
                    SHAP_ENGINE_RUN = get_shap_engine()
                    if SHAP_ENGINE_RUN is None:
                        st.info("SHAP engine unavailable (ShapEngine/shap not loaded).")
                    else:
                        shap_model_name = pick_shap_model_name()
                        if shap_model_name is None:
                            st.info("No SHAP-capable tree model found (needs LightGBM/XGBoost/RF/DT).")
                        else:
                            mdl_shap, _, _ = load_supervised(shap_model_name)
                            if mdl_shap is None:
                                st.info(f"SHAP model missing: {shap_model_name}")
                            else:
                                try:
                                    df_fe = df_shap_base.copy()
                                    df_fe["timestamp"] = df_shap_base.get("timestamp_used_str", df_shap_base.get("timestamp_local_str", df_shap_base.get("timestamp", ""))).astype(str)
                                    if "raw" not in df_fe.columns:
                                        df_fe["raw"] = df_fe.get("raw_log", "").astype(str)

                                    X = _run_feature_engineering(df_fe).copy()
                                    X["odd_hours"] = pd.to_numeric(df_shap_base.get("odd_hours_used", df_shap_base.get("odd_hours_local", 0)), errors="coerce").fillna(0).astype(int).to_numpy()
                                    X_scaled, scaler_feats, _ = prepare_model_matrix(X, SCALER, BUNDLE_FEATURE_COLS)
                                    X_df = pd.DataFrame(np.asarray(X_scaled), columns=list(scaler_feats), index=df_shap_base.index)

                                    if do_row:
                                        try:
                                            res = compute_shap_rows(SHAP_ENGINE_RUN, mdl_shap, X_df, [r], df_shap_base, topk=80)
                                            st.session_state["shap_row"] = build_shap_payload(f"{shap_model_name} SHAP (Row {r})", res, "row", shap_model_name)
                                        except Exception as e:
                                            st.error(f"Row SHAP failed: {e}")

                                    if do_subset:
                                        idxs = parse_index_list(subset_text, n)
                                        if not idxs:
                                            st.warning("No valid subset indices.")
                                        else:
                                            try:
                                                res = compute_shap_rows(SHAP_ENGINE_RUN, mdl_shap, X_df, idxs, df_shap_base, topk=80)
                                                st.session_state["shap_subset"] = build_shap_payload(
                                                    f"{shap_model_name} SHAP (Subset avg rows={idxs[:25]}{'...' if len(idxs)>25 else ''})",
                                                    res, "mean", shap_model_name
                                                )
                                            except Exception as e:
                                                st.error(f"Subset SHAP failed: {e}")

                                    if do_all:
                                        cap = min(n, 2000000)
                                        idxs = list(range(cap))
                                        try:
                                            res = compute_shap_rows(SHAP_ENGINE_RUN, mdl_shap, X_df, idxs, df_shap_base, topk=80)
                                            st.session_state["shap_all"] = build_shap_payload(f"{shap_model_name} SHAP (Overall avg rows=0..{cap-1})", res, "mean", shap_model_name)
                                        except Exception as e:
                                            st.error(f"Overall SHAP failed: {e}")
                                except Exception as e:
                                    st.error(f"SHAP preparation failed: {e}")

            for key, label in [("shap_row", "Selected Row"), ("shap_subset", "Subset Avg"), ("shap_all", "Overall Avg")]:
                p = st.session_state.get(key)
                if not p:
                    continue
                st.markdown(f"**{label}** — {p.get('summary','')}")
                st.image(p.get("png"), caption=p.get("title", label))
                tr = p.get("top_rows")
                if isinstance(tr, pd.DataFrame):
                    ui_df(tr)

    # PDF
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("📄 PDF Report")

    if ui_btn("🧾 Generate PDF (prepare download)", type="secondary", key="btn_pdf"):
        with st.spinner("Building PDF …"):
            df_pdf = results_full.head(int(pdf_max_rows)).copy()

            shap_payloads = None
            if include_shap_pdf:
                shap_payloads = {}
                for k in ("shap_row", "shap_subset", "shap_all"):
                    p = st.session_state.get(k)
                    if p and isinstance(p, dict):
                        shap_payloads[k] = {
                            "title": p.get("title", ""),
                            "summary": p.get("summary", ""),
                            "png": p.get("png", None),
                            "top_rows": p.get("top_rows", None),
                        }
                if not shap_payloads:
                    shap_payloads = None

            meta = {
                "summary_line": (
                    f"Threshold={st.session_state.get('threshold', 0.5):.4f} ({st.session_state.get('threshold_type','')}) · "
                    f"Total={n_total} · Malicious={n_mal} ({pct_mal:.2f}%) · Non-malicious/Legit={n_non} ({pct_non:.2f}%) · "
                    f"WhitelistMode={wl_mode}"
                ),
                "summary_block": (
                    f"Total entries: {n_total}\n"
                    f"Malicious entries: {n_mal} ({pct_mal:.2f}%)\n"
                    f"Non-malicious / legit entries: {n_non} ({pct_non:.2f}%)\n"
                    f"Threshold used: {st.session_state.get('threshold', 0.5):.4f} ({st.session_state.get('threshold_type','')})\n"
                    f"Whitelist mode: {wl_mode}\n"
                ),
            }

            pdf_bytes, pdf_name, note = build_pdf_bytes(
                df_entries=df_pdf,
                meta=meta,
                metrics=metrics_payload,
                shap_payloads=shap_payloads,
                exfil_domain=st.session_state.get("exfil_domain", None),
                exfil_ip=st.session_state.get("exfil_ip", None),
                raw_log_max_chars=int(pdf_raw_max_chars),
            )
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["pdf_name"] = pdf_name
            st.session_state["pdf_note"] = note

            if not pdf_bytes:
                st.error(f"PDF generation failed. {note or 'Unknown error.'}")
            elif note:
                st.warning(note)

    if st.session_state.get("pdf_bytes"):
        ui_dl("📥 Download PDF", st.session_state["pdf_bytes"], st.session_state["pdf_name"], "application/pdf", key="dl_pdf")
