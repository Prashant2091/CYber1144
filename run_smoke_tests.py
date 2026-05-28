"""Smoke tests for the optimized cyber Streamlit pipeline.

These tests do not require trained model files. They validate the parser,
imputer, IPv6 primary conditions, feature engineering, artifact alignment, and
Streamlit app utility import path.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_core_pipeline() -> None:
    from artifacts import Bundle, compute_primary_flags, prepare_model_matrix
    from feature_engineering import our_custom_feature_engineering_function
    from imputer import ForensicImputer, PRIMARY_COLS
    from ipv6_primary_conditions import add_ipv6_primary_conditions
    from log_parser import parse_log_line_universal

    logs = [
        "2026-05-10 02:10:00 +0530 10.0.0.5 GET /index.html 200 1200 300 Mozilla/5.0,ok,foo,bar,legit,none,none,low example.com",
        "waf\t2026-05-10T00:20:00Z WAF client_ip=61.177.56.27 method=GET uri=/wp-login.php status=403 action=BLOCK rule=SQLI score=9",
        "edr\t2026-05-10T01:22:00Z EDR host=DC-01 user=admin proc=powershell.exe verdict=malicious args=\"powershell -enc AAA\"",
        "flow\t2026-05-10T03:30:00Z flow src=2002:cb00:7100::1 dst=8.8.8.8 proto=TCP bytes_out=999999999 bytes_in=100 action=ALLOW tunnel=6to4 download=payload",
        "not-a-known-format with hxxp://login-verify.xyz payload beacon",
    ]


    json_row = parse_log_line_universal('{"timestamp":"2026-05-10T01:00:00Z","source_ip":"1.2.3.4","destination_ip":"5.6.7.8","useragent":"curl/8","url":"https://example.com/a"}')
    _assert(json_row["client_ip"] == "1.2.3.4", "JSON/table source_ip was not canonicalized")
    _assert(json_row["dest_ip"] == "5.6.7.8", "JSON/table destination_ip was not canonicalized")
    _assert(json_row["user_agent"] == "curl/8", "JSON/table useragent was not preserved")

    df_raw = pd.DataFrame([parse_log_line_universal(line) for line in logs])
    for col in PRIMARY_COLS:
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    imp = ForensicImputer(
        priors={"column_defaults": {"user_agent": "artifact-agent", "workstation": "artifact-host"}},
        top_domains={"google.com", "github.com", "microsoft.com", "amazon.com"},
    ).impute_df(df_raw)
    _assert("NotProvided" not in "|".join(imp.astype(str).agg("|".join, axis=1).tolist()), "old NotProvided token leaked into imputation")
    _assert((imp["user_agent"].astype(str).str.len() > 0).all(), "advanced imputation failed to fill user_agent")
    imp = add_ipv6_primary_conditions(imp)
    imp["primary_flag"] = compute_primary_flags(
        imp,
        Bundle(model_dir=str(ROOT), bad_ips={"61.177.56.27"}, bad_domains=set()),
    )
    imp["tz_force_utc"] = 1
    imp["timestamp_local_str"] = imp["timestamp"].astype(str)
    imp["tz_offset_min"] = 0
    imp["wall_dt"] = pd.to_datetime(imp["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)

    features = our_custom_feature_engineering_function(imp, whitelist_domains={"example.com"})
    _assert(len(features) == len(logs), "feature row count mismatch")
    _assert("ipv6_tunnel_any" in features.columns, "missing IPv6 tunnel feature")
    _assert("whitelist_suspicious_combo" in features.columns, "missing whitelist combo feature")
    for leaked in ["threat_level", "user_agent", "domain", "timestamp", "client_ip", "raw"]:
        _assert(leaked not in features.columns, f"pass-through/debug column leaked into runtime feature matrix: {leaked}")
    _assert(int(imp["primary_flag"].sum()) >= 2, "primary flags did not fire on WAF/EDR samples")
    _assert(int(imp["ipv6_primary_flag"].sum()) >= 1, "IPv6 primary flag did not fire")

    # Verify space/underscore feature alias repair.
    aliased = features.rename(columns={"domain_cat_Rare Domain": "domain_cat_Rare_Domain"})
    arr, cols, xdf = prepare_model_matrix(
        aliased,
        scaler=None,
        feature_cols=["domain_cat_Rare Domain", "odd_hours", "ipv6_tunnel_any"],
    )
    _assert(arr.shape == (len(logs), 3), "prepared matrix shape mismatch")
    _assert(cols[0] == "domain_cat_Rare Domain", "feature alias did not preserve requested schema")




def test_family_aware_override_rules() -> None:
    from feature_engineering import our_custom_feature_engineering_function
    from imputer import ForensicImputer, PRIMARY_COLS
    from ipv6_primary_conditions import add_ipv6_primary_conditions
    from log_parser import parse_log_line_universal
    from override_rules import compute_override_signals

    cases = [
        # Benign / should NOT primary override
        ("proxy_ok", "1,2025-11-01 00:00:00,192.168.1.10,github.com,google.com,Mozilla/5.0 (X11; Linux x86_64) Firefox/146.0,41478,", 0),
        ("dhcp_ok", "11,11/01/25,00:24:33,Renew,192.168.1.121,NYC-LEG-L0010,EB99931717C2", 0),
        ("email_clean", "2025-11-01T00:00:19Z email_sec from=it@company.com to=harvey.seligman verdict=CLEAN action=DELIVER urls=0", 0),
        ("edr_benign", "2025-11-01T00:01:11Z edr host=LON-FIN-L0090 user=james.shipp verdict=Benign proc=chrome.exe", 0),
        ("mfa_approve", "2025-11-01T00:00:10Z mfa user=william.brun method=PUSH result=APPROVE ip=8.8.8.8 device=iPhone", 0),
        ("firewall_allow_low", "1,2025-11-01,00:00:06,192.168.1.238,8.8.8.8,ALLOW,14082,", 0),
        # Malicious / should primary override
        ("waf_sqli", "2025-11-01T00:00:10Z waf action=BLOCK client_ip=94.190.43.52 method=GET uri=\"/login?u=' OR 1=1--\" rule=SQLI score=98", 1),
        ("ids_c2", "2025-11-01T00:00:28Z ids alert=\"ET MALWARE Possible C2\" severity=2 src=192.168.1.177 dst=185.112.83.116 proto=TCP", 1),
        ("apache_put", "64.188.21.227 - - [07/Jun/2021:07:50:36 +0000] \"PUT /login HTTP/1.1\" 500 3094 \"-\" \"Mozilla/5.0\"", 1),
        ("proxy_ioc", "5,2025-11-01 00:00:19,192.168.1.175,e-files.download,media.vietnamflash.com,NetSpider,276616,", 1),
        ("flow_ioc", "2025-11-01T00:00:36Z flow src=192.168.1.87:51310 dst=185.112.83.116:443 proto=TCP bytes_out=359884 bytes_in=5141 pkts=289 dur=38s", 1),
        ("os_lolbin", "type=EXECVE msg=audit(1730419272.123:101): argc=3 a0=\"curl\" a1=\"-fsSL\" a2=\"http://e-files.download/p.sh\"", 1),
        ("k8s_priv", "2025-11-01T00:00:45Z k8s_audit user=admin1 verb=patch resource=clusterrolebindings obj=cluster-admin decision=allow", 1),
    ]
    rows = []
    for _name, line, _expected in cases:
        rows.append(parse_log_line_universal(line))
    df_raw = pd.DataFrame(rows)
    for col in PRIMARY_COLS:
        if col not in df_raw.columns:
            df_raw[col] = np.nan
    imp = ForensicImputer(top_domains={"google.com", "github.com", "microsoft.com", "amazon.com"}).impute_df(df_raw)
    imp = add_ipv6_primary_conditions(imp)
    # Minimal time scaffolding for feature engineering.
    imp["wall_dt"] = pd.to_datetime(imp["timestamp"], errors="coerce", utc=True).dt.tz_localize(None).fillna(pd.Timestamp("2025-11-01 00:00:00"))
    imp["tz_offset_min"] = 0
    imp["timestamp_local_str"] = imp["wall_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    imp["timestamp_utc_str"] = imp["timestamp_local_str"]
    imp["odd_hours_used"] = 1
    X = our_custom_feature_engineering_function(imp, whitelist_domains={"google.com", "github.com", "microsoft.com", "amazon.com"})
    flags, codes, dbg = compute_override_signals(
        imp,
        X=X,
        odd_used=np.ones(len(imp), dtype=np.int8),
        bad_ips={"185.112.83.116", "94.190.43.52"},
        bad_domains={"e-files.download"},
        top_domains={"google.com", "github.com", "microsoft.com", "amazon.com"},
    )
    for i, (name, _line, expected) in enumerate(cases):
        _assert(int(flags[i]) == expected, f"family-aware override failed for {name}: expected {expected}, got {int(flags[i])}, code={int(codes[i])}")
    _assert(dbg["conf_tag_suspicious"].dtype == bool, "debug suspicious tag should be bool")

    # Regression: first-row malicious text must not contaminate missing command
    # defaults for later benign rows in the same batch.
    contam_lines = [
        "2024-09-02 04:57:15 +0530 192.168.182.199 GET /page/36 500 4996 9301 Internet Explorer (compatible,Suricata Malware user-agent rules,Malware,Malware,https://rules.example/rules,high,low,medium,Detection rule,network flow direction internal --> external simplepooltips.com",
        "1,2025-11-01 00:00:00,192.168.1.10,github.com,google.com,Mozilla/5.0 (X11; Linux x86_64) Firefox/146.0,41478,",
    ]
    df2 = pd.DataFrame([parse_log_line_universal(x) for x in contam_lines])
    for col in PRIMARY_COLS:
        if col not in df2.columns:
            df2[col] = np.nan
    imp2 = ForensicImputer(top_domains={"google.com", "github.com"}).impute_df(df2)
    flags2, codes2, _ = compute_override_signals(imp2, top_domains={"google.com", "github.com"})
    _assert(int(flags2[1]) == 0, f"benign proxy was contaminated by previous malicious row, code={int(codes2[1])}")



def test_primary_column_imputation_no_unresolved() -> None:
    from imputer import ForensicImputer, PRIMARY_COLS
    from log_parser import parse_log_line_universal

    line = (
        "2024-03-15 00:51:06 +0530 192.168.179.213 POST /page/91 500 3520 2676 "
        "gopher,crawlers / bad robots / suspicious spiders / junk web-scrapers / malicious spammers"
        "Resource Development,Bots & Vulnerability Scanner,Bots & Vulnerability Scanner,"
        "https://raw.githubusercontent.com/mitchellkrogza/nginx-ultimate-bad-bot-blocker/master/_generator_lists/bad-user-agents.list,"
        "medium,low,medium,Detection rule,network flow direction internal --> external onlinadverts.com"
    )
    df = pd.DataFrame([parse_log_line_universal(line)])
    for col in PRIMARY_COLS:
        if col not in df.columns:
            df[col] = np.nan
    imp = ForensicImputer(top_domains={"google.com", "github.com"}).impute_df(df)
    bad_tokens = ["NotProvided", "unresolved-host", "unresolved-process", "unknown_host", "unknown_process"]
    joined = "|".join(imp.astype(str).iloc[0].tolist())
    for tok in bad_tokens:
        _assert(tok.lower() not in joined.lower(), f"primary imputation leaked placeholder token: {tok}")
    _assert(str(imp.loc[0, "dest_ip"]) != "0.0.0.0", "dest_ip remained unspecified 0.0.0.0")
    _assert(str(imp.loc[0, "workstation"]).startswith("WEBCLIENT-") or str(imp.loc[0, "workstation"]).startswith("EXTWEBCLIENT-"), f"workstation was not context-imputed: {imp.loc[0, 'workstation']}")
    _assert(str(imp.loc[0, "process"]) in {"scanner_client", "http_post_client"}, f"process was not context-imputed: {imp.loc[0, 'process']}")
    _assert(str(imp.loc[0, "username"]).startswith("web_user_"), f"username was not pseudonymized by context: {imp.loc[0, 'username']}")

    # Artifact repair maps from the imputation notebook must win over generated fallbacks.
    maps = {
        "repair_maps": {
            "ip_to_ws": {"192.168.179.213": "NYC-WEB-L1234"},
            "ip_to_user": {"192.168.179.213": "svc_web_nyc_424242"},
        }
    }
    imp2 = ForensicImputer(resolver_state=maps, top_domains={"google.com"}).impute_df(df)
    _assert(imp2.loc[0, "workstation"] == "NYC-WEB-L1234", "repair_maps ip_to_ws did not override generated workstation")
    _assert(imp2.loc[0, "username"] == "svc_web_nyc_424242", "repair_maps ip_to_user did not override generated username")


def test_training_schema_alignment() -> None:
    from artifacts import prepare_model_matrix
    from feature_schema import MODEL_FEATURES_41, MODEL_FEATURES_42, clean_model_feature_columns

    bad_full_fe_cols = [
        "ip_bad_rep", "suspicious_geo", "malicious_lolbin_ua", "odd_hours",
        "high_bytes_out", "domain_cat_Rare Domain", "combined_rare_suspicious_ua",
        "first_cloud_use", "suspicious_url", "is_weekend", "peak_hour",
        "hour_of_day_bin", "day_of_week_bin", "avg_bytes_per_hour_bin",
        "user_activity_freq_bin", "ip_request_freq_bin", "interaction_ip_geo",
        "interaction_ua_odd_hours", "burst_activity", "rare_user_agent",
        "requests_per_ip_hour", "ip_freq_enc", "rare_suspicious_activity",
        "ua_freq", "domain_freq", "ua_activity_10min", "ip_request_deviation",
        "critical_events", "ip_geo_malicious_interaction", "odd_hour_lolbin_ua",
        "extreme_ip_request_spike", "timestamp_suspicious_tz",
        "data_exfil_ratio", "threat_level", "threat_level_int",
        "critical_mod_success", "username_risk_score", "command_risk_score",
        "workstation_risk_score", "location_anomaly_score", "session_intensity",
        "behavioral_consistency", "session_multi_device_risk", "ipv6_tunnel_any",
        "whitelist_hit", "whitelist_suspicious_combo", "user_agent", "domain",
        "timestamp", "client_ip", "method", "url_path", "bytes_out", "bytes_in",
        "status", "log_type", "raw",
    ]
    cols41 = clean_model_feature_columns(bad_full_fe_cols, expected_n=41)
    _assert(cols41 == MODEL_FEATURES_41, "41-feature schema did not recover from full FE output")
    cols42 = clean_model_feature_columns(bad_full_fe_cols, expected_n=42)
    _assert(cols42 == MODEL_FEATURES_42, "42-feature schema did not recover from full FE output")
    _assert("threat_level" not in cols42 and "user_agent" not in cols42, "pass-through column survived schema cleaning")

    class DummyScaler:
        n_features_in_ = 41
        def transform(self, X):
            return X

    X = pd.DataFrame({c.replace(" ", "_"): [1.0, 0.0] for c in MODEL_FEATURES_41})
    X["threat_level"] = ["HIGH", "LOW"]
    X["raw"] = ["a", "b"]
    arr, cols, xdf = prepare_model_matrix(X, DummyScaler(), bad_full_fe_cols)
    _assert(arr.shape == (2, 41), "model matrix did not align to 41-feature trained schema")
    _assert(cols == MODEL_FEATURES_41, "prepared columns are not the exact 41-feature schema")
    _assert("domain_cat_Rare Domain" in cols, "space-containing feature name was not preserved")

class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _DummyUI:
    def expander(self, *_args, **_kwargs):
        return _DummyContext()

    def spinner(self, *_args, **_kwargs):
        return _DummyContext()

    def columns(self, *_args, **_kwargs):
        return [_DummyContext(), _DummyContext(), _DummyContext()]

    def header(self, *_args, **_kwargs):
        return None

    def caption(self, *_args, **_kwargs):
        return None

    def code(self, *_args, **_kwargs):
        return None

    def markdown(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def success(self, *_args, **_kwargs):
        return None

    def dataframe(self, *_args, **_kwargs):
        return None

    def subheader(self, *_args, **_kwargs):
        return None

    def pyplot(self, *_args, **_kwargs):
        return None

    def image(self, *_args, **_kwargs):
        return None

    def button(self, *_args, **_kwargs):
        return False

    def download_button(self, *_args, **_kwargs):
        return False

    def checkbox(self, *_args, **kwargs):
        return kwargs.get("value", False)

    def text_input(self, *_args, **kwargs):
        return kwargs.get("value", "")

    def text_area(self, *_args, **_kwargs):
        return ""

    def number_input(self, *_args, **kwargs):
        return kwargs.get("value", 0)

    def selectbox(self, *_args, **kwargs):
        options = _args[1] if len(_args) > 1 else kwargs.get("options", [])
        index = kwargs.get("index", 0)
        return options[index] if options else None

    def radio(self, *_args, **kwargs):
        options = _args[1] if len(_args) > 1 else kwargs.get("options", [])
        return options[0] if options else None

    def slider(self, *_args, **kwargs):
        return kwargs.get("value", 0)

    def file_uploader(self, *_args, **_kwargs):
        return None


class _DummyStreamlit(types.ModuleType, _DummyUI):
    def __init__(self):
        types.ModuleType.__init__(self, "streamlit")
        self.sidebar = _DummyUI()
        self.session_state = {}

    def set_page_config(self, *_args, **_kwargs):
        return None

    def stop(self):
        raise RuntimeError("streamlit stop called during smoke import")


def test_app_utility_import() -> None:
    sys.modules["streamlit"] = _DummyStreamlit()
    os.environ["MODEL_DIR"] = tempfile.mkdtemp(prefix="cyber_models_empty_")

    import utils.app as app  # noqa: WPS433

    _assert(str(ROOT / "artifacts.py") in str(app.ART_SRC), "app did not load local optimized artifacts.py")

    df = pd.DataFrame(
        {
            "timestamp": ["2026-05-10 02:10:00 +0530", "bad"],
            "raw_log": ["normal browser request", "unparseable"],
            "workstation": ["DEL-PC", ""],
        }
    )
    out = app.add_wall_time_and_odd(df)
    _assert(int(out.loc[0, "timestamp_parse_ok"]) == 1, "timestamp parser failed explicit +0530")
    _assert(str(out.loc[0, "timestamp_utc_str"]).startswith("2026-05-09 20:40"), "UTC conversion mismatch")
    _assert(int(out.loc[0, "odd_hours_local"]) == 1, "odd local hour mismatch")

    # Regression guard for Streamlit/PyArrow duplicate-column crash.
    cols = ["raw_log", "prediction"] + list(app.PRIMARY_COLS)
    deduped = app._dedupe_columns(cols)
    _assert(len(deduped) == len(set(deduped)), "display columns still contain duplicates")
    dup_df = pd.DataFrame([[1, 2]], columns=["raw_log", "raw_log"])
    fixed_df = app._ensure_unique_df_columns(dup_df)
    _assert(not fixed_df.columns.has_duplicates, "ui_df duplicate-column safeguard failed")
    numeric_df = pd.DataFrame([["t", "1.1.1.1"]], columns=[0, 1])
    named_df = app._ensure_unique_df_columns(numeric_df)
    _assert(list(named_df.columns)[:2] == ["timestamp", "client_ip"], "numeric/Column 1 headers were not mapped to semantic names")
    input_generic_df = pd.DataFrame([["a", "b"]], columns=["input_Column_1", "input_Column_2"])
    input_named_df = app._ensure_unique_df_columns(input_generic_df)
    _assert("input_Column_1" not in input_named_df.columns and "input_Column_2" not in input_named_df.columns, "generic input_Column names leaked to display")

    csv_upload = io.BytesIO(b"timestamp,source_ip,destination_ip,useragent,url\n2026-05-10T01:00:00Z,1.2.3.4,5.6.7.8,curl/8,https://example.com/a\n")
    csv_upload.name = "events.csv"
    table_lines = list(app.iter_lines_from_upload(csv_upload))
    _assert(len(table_lines) == 1, "uploaded CSV table was not converted to one structured row")
    _assert(table_lines[0].startswith(app._STRUCTURED_ROW_PREFIX), "structured upload prefix missing")
    table_obj = json.loads(table_lines[0][len(app._STRUCTURED_ROW_PREFIX):])
    rec = app.structured_row_to_record(table_lines[0])
    _assert(rec.get("client_ip") == "1.2.3.4" and rec.get("dest_ip") == "5.6.7.8", "uploaded CSV semantic headers were not preserved")
    _assert("input_source_ip" in rec or "input_client_ip" in rec, "original input columns were not retained")

    headerless_proxy = io.BytesIO(b"1,2025-11-01 00:00:00,192.168.1.10,github.com,google.com,Mozilla/5.0 Firefox/146.0,41478,\n2,2025-11-01 00:00:03,192.168.1.85,microsoft.com,google.com,Mozilla/5.0 Chrome/143.0,47801,\n")
    headerless_proxy.name = "proxy.csv"
    proxy_lines = list(app.iter_lines_from_upload(headerless_proxy))
    _assert(len(proxy_lines) == 2, "headerless proxy CSV lost first data row")
    proxy_obj = json.loads(proxy_lines[0][len(app._STRUCTURED_ROW_PREFIX):])
    _assert(set(["timestamp", "client_ip", "domain", "referrer", "user_agent", "bytes_out"]).issubset(proxy_obj.keys()), f"headerless proxy CSV schema was not semantic: {list(proxy_obj.keys())}")
    _assert(not any(str(k).lower().startswith("column") for k in proxy_obj.keys()), "Column-N keys leaked in structured object")
    proxy_rec = app.structured_row_to_record(proxy_lines[0])
    _assert(proxy_rec.get("log_type") == "proxy" and proxy_rec.get("domain") == "github.com", "headerless proxy record was not reconstructed semantically")

    headerless_dhcp = io.BytesIO(b"11,11/01/25,00:24:33,Renew,192.168.1.121,NYC-LEG-L0010,EB99931717C2\n")
    headerless_dhcp.name = "dhcp.csv"
    dhcp_line = list(app.iter_lines_from_upload(headerless_dhcp))[0]
    dhcp_obj = json.loads(dhcp_line[len(app._STRUCTURED_ROW_PREFIX):])
    _assert("date" in dhcp_obj and "time" in dhcp_obj and "client_ip" in dhcp_obj, "headerless DHCP CSV schema was not inferred")
    dhcp_rec = app.structured_row_to_record(dhcp_line)
    _assert(dhcp_rec.get("log_type") == "dhcp" and "11/01/25 00:24:33" in dhcp_rec.get("timestamp", ""), "DHCP date/time were not recomposed")

    # Regression: mixed raw-log corpora with commas must not go through slow
    # pandas table inference or lose the original raw line.
    raw_mixed = io.BytesIO((
        "2024-03-15 00:51:06 +0530 192.168.179.213 POST /page/91 500 3520 2676 gopher,crawlers / bad robots,Detection rule,network flow direction internal --> external onlinadverts.com\n"
        "45.161.33.88 - - [08/Sep/2021:18:46:06 +0000] \"GET /contact HTTP/1.1\" 404 2545 \"-\" \"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML\"\n"
        "2025-11-01T00:00:28Z ids alert=\"ET MALWARE Possible C2\" severity=2 src=192.168.1.177 dst=185.112.83.116 proto=TCP\n"
    ).encode("utf-8"))
    raw_mixed.name = "mixed_raw_logs.csv"
    t0 = time.perf_counter()
    raw_lines = list(app.iter_lines_from_upload(raw_mixed))
    elapsed = time.perf_counter() - t0
    _assert(len(raw_lines) == 3, "mixed raw CSV-like upload lost rows")
    _assert(not any(line.startswith(app._STRUCTURED_ROW_PREFIX) for line in raw_lines), "mixed raw logs were incorrectly table-parsed")
    _assert(elapsed < 1.0, f"mixed raw CSV sniff path is too slow: {elapsed:.3f}s")

    # Binary UI/report guard: suspicious may remain an internal signal, but it
    # must not appear as a classification category.
    app_src = Path(app.__file__).read_text(encoding="utf-8")
    for forbidden in ["Suspicious entries:", "Suspicious (Threat)", "Suspicious (Scanner)", "Suspicious</span>"]:
        _assert(forbidden not in app_src, f"suspicious classification display string still present: {forbidden}")

    # SHAP ip_bad_rep semantic contract.
    def _ip_shap_direction(row_dict, bad_ips=None):
        app.BAD_IPS = set(bad_ips or [])
        base_res = {
            "sv_rows": np.array([[0.821, 0.25]], dtype=float),
            "feat_names": ["ip_bad_rep", "other_feature"],
            "breakdown_rows": [{"pos_pct": 100.0, "neg_pct": 0.0, "neutral_pct": 0.0, "net_pct": 100.0}],
            "breakdown_mean": {"pos_pct": 100.0, "neg_pct": 0.0, "neutral_pct": 0.0, "net_pct": 100.0},
        }
        ctx = pd.DataFrame([row_dict])
        res = app._attach_ip_bad_rep_semantics(base_res, ctx, [0])
        payload = app.build_shap_payload("test", res, "row", "LightGBM")
        tr = payload["top_rows"]
        r0 = tr[tr["feature"].eq("ip_bad_rep")].iloc[0]
        return str(r0["direction"]), float(r0["shap_mean"]), str(r0.get("semantic_note", ""))

    d, v, note = _ip_shap_direction({"client_ip": "192.168.1.10", "dest_ip": "0.0.0.0", "ip_bad_truth": 0})
    _assert(d == "→" and abs(v) <= 1e-12 and "local" in note.lower(), "local/private ip_bad_rep SHAP was not neutral")
    d, v, note = _ip_shap_direction({"client_ip": "192.168.1.10", "dest_ip": "185.112.83.116", "ip_bad_truth": 1}, {"185.112.83.116"})
    _assert(d == "↑" and v > 0 and "bad_ip_list" in note, "bad-list ip_bad_rep SHAP was not upward")
    d, v, note = _ip_shap_direction({"client_ip": "8.8.8.8", "dest_ip": "1.1.1.1", "ip_bad_truth": 0}, {"185.112.83.116"})
    _assert(d == "↓" and v < 0 and "clean public" in note, "clean public ip_bad_rep SHAP was not downward")


def main() -> None:
    test_core_pipeline()
    test_family_aware_override_rules()
    test_primary_column_imputation_no_unresolved()
    test_training_schema_alignment()
    test_app_utility_import()
    print("OK: parser, imputer, IPv6, features, artifacts, and app utility checks passed.")


if __name__ == "__main__":
    main()
