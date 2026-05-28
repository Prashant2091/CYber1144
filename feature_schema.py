"""Canonical model feature schemas extracted from the Kaggle training notebooks.

The feature-engineering function may produce additional audit/pass-through
columns.  The supervised models, scalers, and calibrators must receive exactly
one of these numeric schemas, depending on the artifacts they were trained with.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
import re

# Final `q` used in cyber-new44 before label split (without label).
MODEL_FEATURES_38: List[str] = [
    "ip_bad_rep",
    "suspicious_geo",
    "malicious_lolbin_ua",
    "odd_hours",
    "high_bytes_out",
    "domain_cat_Rare Domain",
    "combined_rare_suspicious_ua",
    "first_cloud_use",
    "suspicious_url",
    "is_weekend",
    "peak_hour",
    "hour_of_day_bin",
    "day_of_week_bin",
    "avg_bytes_per_hour_bin",
    "user_activity_freq_bin",
    "ip_request_freq_bin",
    "interaction_ip_geo",
    "interaction_ua_odd_hours",
    "burst_activity",
    "rare_user_agent",
    "requests_per_ip_hour",
    "ip_freq_enc",
    "rare_suspicious_activity",
    "domain_freq",
    "ua_activity_10min",
    "ip_request_deviation",
    "critical_events",
    "ip_geo_malicious_interaction",
    "odd_hour_lolbin_ua",
    "extreme_ip_request_spike",
    "data_exfil_ratio",
    "critical_mod_success",
    "username_risk_score",
    "command_risk_score",
    "workstation_risk_score",
    "session_intensity",
    "behavioral_consistency",
    "session_multi_device_risk",
]

# Some artifact cells retained location_anomaly_score in feature_weights / df.drop training.
MODEL_FEATURES_39: List[str] = [
    *MODEL_FEATURES_38[:35],
    "location_anomaly_score",
    *MODEL_FEATURES_38[35:],
]

# Updated notebook variants add the IPv6/timestamp/whitelist features.
MODEL_FEATURES_41: List[str] = [
    *MODEL_FEATURES_38,
    "ipv6_tunnel_any",
    "whitelist_suspicious_combo",
    "timestamp_suspicious_tz",
]

MODEL_FEATURES_42: List[str] = [
    *MODEL_FEATURES_39,
    "ipv6_tunnel_any",
    "whitelist_suspicious_combo",
    "timestamp_suspicious_tz",
]

SCHEMA_BY_N = {
    38: MODEL_FEATURES_38,
    39: MODEL_FEATURES_39,
    41: MODEL_FEATURES_41,
    42: MODEL_FEATURES_42,
}

# Numeric runtime feature union.  Keep helper component flags that may be used
# by MoE/override code, but exclude raw strings such as user_agent/domain/raw.
RUNTIME_FEATURE_COLUMNS: List[str] = []
for _schema in (MODEL_FEATURES_42, MODEL_FEATURES_41, MODEL_FEATURES_39, MODEL_FEATURES_38):
    for _c in _schema:
        if _c not in RUNTIME_FEATURE_COLUMNS:
            RUNTIME_FEATURE_COLUMNS.append(_c)
for _c in [
    "whitelist_hit",
    "ipv6_present_any",
    "ipv6_mapped_any",
    "ipv6_6to4_any",
    "ipv6_teredo_any",
    "ipv6_isatap_any",
    "ipv6_nat64_any",
    "threat_level_int",
    "method_POST_ratio",
    "method_PUT_ratio",
    "method_MOD_ratio",
    "method_window_count",
    "method_has_mod",
    "method_POST_ratio_z",
    "method_PUT_ratio_z",
    "method_MOD_ratio_z",
]:
    if _c not in RUNTIME_FEATURE_COLUMNS:
        RUNTIME_FEATURE_COLUMNS.append(_c)

PASSTHROUGH_OR_LABEL_COLUMNS = {
    "label", "prediction", "prediction_risk", "risk_label", "activity_type", "label_verbose",
    "static_score", "static_score_noisy", "anomaly_score", "parsed_timestamp",
    "timestamp", "client_ip", "dest_ip", "method", "url_path", "status", "bytes_out", "bytes_in",
    "domain", "full_url", "referrer", "user_agent", "username", "workstation", "process", "command",
    "raw", "raw_log", "log_type", "source_log_type", "country", "threat_level",
    "ipv6_present_any", "ipv6_mapped_any", "ipv6_6to4_any", "ipv6_teredo_any", "ipv6_isatap_any", "ipv6_nat64_any",
    "whitelist_hit",
}

# threat_level_int is an intermediate ordinal in FE, but it was not in the
# final q-list used for supervised training.  Treat as bad unless a real
# extended_feature_columns artifact explicitly chooses it and expected_n matches.
ALWAYS_BAD_MODEL_COLUMNS = {
    "label", "threat_level", "user_agent", "domain", "timestamp", "client_ip", "method", "url_path",
    "bytes_out", "bytes_in", "status", "log_type", "raw", "raw_log", "full_url", "referrer", "dest_ip",
    "username", "workstation", "process", "command", "parsed_timestamp", "static_score", "anomaly_score",
}


def feature_key(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def sanitize_feature_list(cols: object) -> List[str]:
    if cols is None:
        return []
    if not isinstance(cols, (list, tuple)):
        try:
            cols = list(cols)  # type: ignore[arg-type]
        except Exception:
            return []
    out: List[str] = []
    seen = set()
    for c in cols:  # type: ignore[assignment]
        s = str(c).strip()
        if not s or s.lower() == "nan" or s in seen:
            continue
        if s == "label":
            continue
        seen.add(s)
        out.append(s)
    return out


def schema_has_pass_through(cols: Sequence[str]) -> bool:
    keys = {feature_key(c) for c in cols}
    bad = {feature_key(c) for c in ALWAYS_BAD_MODEL_COLUMNS}
    if keys & bad:
        return True
    if any(str(c).startswith("__pad_feature_") for c in cols):
        return True
    # Generic f0/f1/... means feature names were lost; use canonical schema if width is known.
    if cols and all(re.fullmatch(r"f\d+", str(c)) for c in cols):
        return True
    return False


def infer_schema_from_columns(cols: Sequence[str]) -> List[str]:
    keys = {feature_key(c) for c in cols}
    has_extra = any(feature_key(c) in keys for c in ["ipv6_tunnel_any", "whitelist_suspicious_combo", "timestamp_suspicious_tz"])
    has_location = feature_key("location_anomaly_score") in keys
    if has_extra and has_location:
        return list(MODEL_FEATURES_42)
    if has_extra:
        return list(MODEL_FEATURES_41)
    if has_location:
        return list(MODEL_FEATURES_39)
    return list(MODEL_FEATURES_38)


def clean_model_feature_columns(cols: object, expected_n: Optional[int] = None) -> List[str]:
    """Return a safe model schema.

    Rules:
      1. If a real extended_feature_columns artifact is valid, keep it.
      2. If the candidate leaks FE audit/pass-through columns, replace it with
         the canonical schema matching scaler/model width.
      3. If feature names were lost, use the known schema by width.
    """
    cleaned = sanitize_feature_list(cols)
    try:
        expected_n = int(expected_n) if expected_n is not None else None
    except Exception:
        expected_n = None

    if expected_n in SCHEMA_BY_N:
        canonical = list(SCHEMA_BY_N[int(expected_n)])
        if not cleaned:
            return canonical
        if len(cleaned) != int(expected_n) or schema_has_pass_through(cleaned):
            return canonical
        return cleaned

    if not cleaned:
        return []

    if schema_has_pass_through(cleaned):
        # If the artifact was accidentally the full FE output, infer the nearest
        # trained q-list rather than truncating through threat_level/raw columns.
        return infer_schema_from_columns(cleaned)

    return cleaned


__all__ = [
    "MODEL_FEATURES_38", "MODEL_FEATURES_39", "MODEL_FEATURES_41", "MODEL_FEATURES_42",
    "RUNTIME_FEATURE_COLUMNS", "SCHEMA_BY_N", "clean_model_feature_columns", "feature_key",
    "schema_has_pass_through", "infer_schema_from_columns", "sanitize_feature_list",
]
