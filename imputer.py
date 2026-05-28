
from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd

PRIMARY_COLS = [
    "timestamp",
    "client_ip",
    "dest_ip",
    "method",
    "url_path",
    "status",
    "bytes_out",
    "bytes_in",
    "domain",
    "full_url",
    "referrer",
    "user_agent",
    "username",
    "workstation",
    "process",
    "command",
    "log_type",
    "raw_log",
]

_TEXT_COLS = [c for c in PRIMARY_COLS if c not in {"status", "bytes_out", "bytes_in"}]
_NUMERIC_COLS = ["status", "bytes_out", "bytes_in"]
_URL_RE = re.compile(r'(?i)\bhttps?://[^\s\'"<>()]+')
_HOST_SCAN_RE = re.compile(r'(?i)\b(?:[a-z0-9-]{1,63}\.)+[a-z]{2,24}\b')
_FILEEXT_BLACKLIST = {
    "png","jpg","jpeg","gif","svg","webp","css","js","json","txt","csv","xml","pdf","doc","docx","xls","xlsx","ppt","pptx","zip","gz","tar","rar","7z","ico","woff","woff2","ttf",
}
_PUBLIC_UNKNOWN_TOKENS = {
    "", "notprovided", "unknown", "none", "null", "nan", "-", "--",
    "na", "n/a", "missing", "missing_token", "not_available", "not-available",
    "unresolved", "unresolved-host", "unresolved-process",
    "unknown_host", "unknown-host", "unknown_process", "unknown-process", "unknown_user",
    "0.0.0.0", "::", "::0", "[::]",
}

_COMMON_TLDS = {
    "com","net","org","edu","gov","mil","int","io","ai","dev","app","cloud","co","us","uk","in","de","fr","es","it","nl","no","se","fi","dk","ch","au","ca","jp","kr","cn","sg","hk","br","mx","za","ru","pl","cz","eu","info","biz","xyz","top","link","download","site","online","tech","me","tv","cc","ws","wiki"
}
_PERSONAL_DOTTED_RE = re.compile(r"^[a-z]{2,}\.[a-z]{2,}$", re.I)


_DEFAULT_TEXT_PRIOR_KEYS = (
    "column_defaults",
    "text_defaults",
    "categorical_defaults",
    "most_frequent",
    "mode_values",
    "fill_values",
    "imputation_defaults",
)

_MISSING_TEXT_RE = re.compile(
    r"^(?:|unknown|unknown-domain|unknown_domain|null|none|nan|na|n/a|-|--|"
    r"notprovided|not_provided|missing_token|missing|not_available|not-available|"
    r"unresolved|unresolved-host|unresolved-process|unknown_host|unknown-host|"
    r"unknown_process|unknown-process|unknown_user|0\.0\.0\.0|::|::0|\[::\])$",
    re.I,
)

_COLUMN_FALLBACKS = {
    "client_ip": "10.255.0.1",
    "dest_ip": "198.51.100.1",
    "method": "GET",
    "url_path": "/",
    "domain": "event.local",
    "full_url": "http://event.local/",
    "referrer": "direct",
    "user_agent": "generic-browser",
    "username": "event_user",
    "workstation": "WS-AUTO-0000",
    "process": "event_process",
    "command": "observed_event",
    "log_type": "unknown",
    "raw_log": "",
}

def _is_missing_text_value(x: Any) -> bool:
    if x is None:
        return True
    try:
        if isinstance(x, float) and np.isnan(x):
            return True
    except Exception:
        pass
    return bool(_MISSING_TEXT_RE.match(str(x).strip()))

def _first_prior_value(mapping: Dict[str, Any], col: str):
    if not isinstance(mapping, dict):
        return None
    candidates = [col, col.lower(), col.upper(), f"default_{col}", f"{col}_default"]
    for key in candidates:
        if key in mapping and not _is_missing_text_value(mapping[key]):
            return mapping[key]
    return None

def _clean_prior_default(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            cleaned = _clean_prior_default(item)
            if cleaned is not None:
                return cleaned
        return None
    if isinstance(value, dict):
        for key in ("value", "default", "mode", "fill", "impute"):
            if key in value:
                cleaned = _clean_prior_default(value[key])
                if cleaned is not None:
                    return cleaned
        # Some artifacts store value -> count maps. Pick the highest-count non-missing key.
        try:
            ranked = sorted(value.items(), key=lambda kv: float(kv[1]), reverse=True)
            for k, _ in ranked:
                cleaned = _clean_prior_default(k)
                if cleaned is not None:
                    return cleaned
        except Exception:
            return None
    s = str(value).strip()
    return None if _is_missing_text_value(s) else s

_IPV4_RE_SAFE = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")

def sanitize_domain_series(host: pd.Series) -> pd.Series:
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

def _is_public_ip(ip_str: str) -> bool:
    ip_str = (ip_str or "").strip()
    try:
        ip = ipaddress.ip_address(ip_str)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified)
    except Exception:
        return False

def _domain_from_url(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").strip().lower()
    except Exception:
        return ""

def _path_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        path = p.path or ""
        if p.query:
            path += "?" + p.query
        return path
    except Exception:
        return ""

def _scan_host_from_text(text: str) -> str:
    """Extract a likely network hostname without mistaking usernames for domains.

    Raw CSV rows often contain personal usernames such as john.doe before any
    real URL/domain. Treat dotted personal-name tokens as non-domains unless
    their suffix is a known TLD. This prevents imputation from filling the
    domain column with names like kenneth.dorsey.
    """
    if not text:
        return ""
    for m in _HOST_SCAN_RE.finditer(str(text).lower()):
        host = (m.group(0) or "").strip(".").lower()
        if not host:
            continue
        tld = host.rsplit(".", 1)[-1]
        if tld in _FILEEXT_BLACKLIST:
            continue
        if tld not in _COMMON_TLDS and _PERSONAL_DOTTED_RE.match(host):
            continue
        return host
    return ""

def _is_missing_text_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    return s.eq("") | s.str.lower().isin(_PUBLIC_UNKNOWN_TOKENS | {"na", "n/a", "missing", "not_available", "not-available"})


def _coerce_text(series: pd.Series, fill_value: str = "") -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    s = s.replace({"nan": "", "None": "", "NULL": "", "null": ""})
    fill_value = str(fill_value or "").strip()
    return s.mask(s.map(_is_missing_text_value), fill_value)


def _first_nonempty(*vals: Any) -> str:
    for v in vals:
        if v is None:
            continue
        try:
            if isinstance(v, float) and np.isnan(v):
                continue
        except Exception:
            pass
        if isinstance(v, (list, tuple)):
            for vv in v:
                out = _first_nonempty(vv)
                if out:
                    return out
            continue
        s = str(v).strip()
        if s and s.lower() not in _PUBLIC_UNKNOWN_TOKENS and s.lower() not in {"na", "n/a", "missing", "not_available", "not-available"}:
            return s
    return ""


def _lookup_artifact_default(priors: Dict[str, Any], col: str) -> str:
    """Return a learned/default value from artifacts when present.

    Supports common artifact shapes:
      - priors['column_defaults'][col]
      - priors['column_modes'][col]
      - priors['defaults'][col]
      - priors['modes'][col]
      - priors[f'default_{col}']
      - priors[col]['mode'|'default'|'most_common']
    """
    if not isinstance(priors, dict):
        return ""
    for key in ("column_defaults", "column_modes", "defaults", "modes", "most_common", "fill_values"):
        obj = priors.get(key)
        if isinstance(obj, dict):
            val = _first_nonempty(obj.get(col), obj.get(str(col).lower()), obj.get(str(col).upper()))
            if val:
                return val
    val = _first_nonempty(priors.get(f"default_{col}"), priors.get(f"mode_{col}"), priors.get(col))
    if val and not isinstance(priors.get(col), dict):
        return val
    obj = priors.get(col)
    if isinstance(obj, dict):
        return _first_nonempty(obj.get("default"), obj.get("mode"), obj.get("most_common"), obj.get("median"))
    return ""


def _semantic_fallback(col: str) -> str:
    """Typed neutral fallback used only after artifact/context imputation fails.

    This intentionally avoids the previous blanket 'NotProvided' token.
    """
    return {
        "timestamp": "1970-01-01 00:00:00 +0000",
        "client_ip": "0.0.0.0",
        "dest_ip": "0.0.0.0",
        "method": "GET",
        "url_path": "/",
        "domain": "unknown.local",
        "full_url": "http://unknown.local/",
        "referrer": "direct",
        "user_agent": "unknown-agent",
        "username": "unknown_user",
        "workstation": "unknown_host",
        "process": "unknown_process",
        "command": "unknown_command",
        "log_type": "unknown",
        "raw_log": "",
    }.get(col, "unknown")


def _resolve_map(resolver_state: Dict[str, Any], map_names: tuple[str, ...], key: str) -> str:
    if not isinstance(resolver_state, dict) or not key:
        return ""
    key_s = str(key).strip()
    for name in map_names:
        obj = resolver_state.get(name)
        if isinstance(obj, dict):
            val = _first_nonempty(obj.get(key_s), obj.get(key_s.lower()), obj.get(key_s.upper()))
            if val:
                return val
    return ""


# ------------------------------------------------------------------------------
# Advanced deterministic imputation helpers (artifact-first, row-context second)
# ------------------------------------------------------------------------------
def _stable_int(seed: Any) -> int:
    import hashlib
    b = str(seed if seed is not None else "").encode("utf-8", errors="ignore")
    return int(hashlib.md5(b).hexdigest()[:12], 16)


def _normalize_domain_token(x: Any) -> str:
    s = "" if x is None else str(x).strip().lower()
    if not s or _is_missing_text_value(s):
        return ""
    s = re.sub(r"^https?://", "", s)
    s = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    s = s.split("@")[-1].split(":")[0].strip("[] .")
    return s


def _is_private_ip_str(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(str(ip_str).strip().strip("[]"))
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified)
    except Exception:
        return False


def _ip_pseudo_domain(ip_str: Any) -> str:
    ip_s = "" if ip_str is None else str(ip_str).strip().strip("[]")
    if not ip_s or _is_missing_text_value(ip_s):
        return "network.local"
    token = re.sub(r"[^a-zA-Z0-9]+", "-", ip_s).strip("-").lower()
    return f"ip-{token}.{'internal' if _is_private_ip_str(ip_s) else 'external'}"


_SITE_CODES = np.array(["NYC", "LON", "SFO", "TKY", "DEL", "BLR", "SIN", "AMS", "FRA", "SEA", "CHI", "DAL"], dtype=object)
_GENERIC_USER_TOKENS = _PUBLIC_UNKNOWN_TOKENS | {"anonymous", "guest", "corp_user", "external_user", "user", "unknownuser", "unknown_user"}
_WEAK_WORKSTATION_RE = re.compile(r"^(?:workstation-?\d*|externalclient-?\d*|host|localhost|computer|device|pc|ws|wks|unknown-host|unknown_host|unresolved-host|unresolved)$", re.I)
_EXE_RE = re.compile(r"(?i)(?:^|[\\/\s\"])([a-z0-9_.-]+\.exe|powershell|pwsh|cmd|curl|wget|bash|sh|zsh|python3?|perl|ruby|nc|netcat|msxsl|certutil|bitsadmin|mshta|rundll32|regsvr32|wevtutil)(?:$|[\s\"/:])")


def _tokenize_safe(x: Any, max_len: int = 24) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s[:max_len] or "event")


def _stable_code(seed: Any, width: int = 4) -> str:
    mod = 10 ** int(width)
    return str(_stable_int(seed) % mod).zfill(int(width))


def _ip_kind(ip_s: Any) -> str:
    try:
        ip = ipaddress.ip_address(str(ip_s).strip().strip("[]"))
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return "internal"
        if ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            return "special"
        return "external"
    except Exception:
        return "unknown"


def _site_from_any(workstation: Any = "", client_ip: Any = "", seed: Any = "") -> str:
    ws = str(workstation or "").upper()
    for code in _SITE_CODES:
        if re.search(rf"(?:^|[-_/\.]){re.escape(str(code))}(?:[-_/\.]|$)", ws):
            return str(code)
    ip_s = str(client_ip or "").strip()
    m = re.match(r"^192\.168\.(\d{1,3})\.", ip_s)
    if m:
        vlan = int(m.group(1))
        return str(_SITE_CODES[vlan % len(_SITE_CODES)])
    return str(_SITE_CODES[_stable_int(str(seed) + "|site") % len(_SITE_CODES)])


def _ua_device_one(ua: Any) -> str:
    u = str(ua or "").lower()
    if not u or _is_missing_text_value(u):
        return "unknown"
    if re.search(r"sqlmap|nikto|nmap|masscan|zgrab|crawler|spider|scanner|bot|pavuk|httrack|wget|curl|python-requests|go-http-client|libwww|headless|selenium", u):
        return "scanner"
    if "windows" in u:
        return "windows"
    if "android" in u:
        return "android"
    if "iphone" in u or "ipad" in u or "ios" in u:
        return "ios"
    if "mac" in u:
        return "mac"
    if "linux" in u or "x11" in u:
        return "linux"
    if re.search(r"chrome|firefox|safari|edge|mozilla", u):
        return "browser"
    return _tokenize_safe(u, 16)


def _ua_device_series(ua: pd.Series) -> pd.Series:
    return ua.fillna("").astype(str).map(_ua_device_one)


def _family_hint_series(out: pd.DataFrame) -> pd.Series:
    idx = out.index
    logt = out.get("log_type", pd.Series("", index=idx)).fillna("").astype(str).str.lower()
    raw = out.get("raw_log", pd.Series("", index=idx)).fillna("").astype(str).str.lower()
    blob = logt + " " + raw
    fam = pd.Series("event", index=idx, dtype="object")
    fam.loc[blob.str.contains(r"dhcp|type7_asset|\brenew\b|\bassign\b", regex=True, na=False)] = "dhcp"
    fam.loc[blob.str.contains(r"proxy|apache|web|edge|waf|apigw|lb|envoy|type1_space|type4_web|type5_proxy", regex=True, na=False)] = "web"
    fam.loc[blob.str.contains(r"dns", regex=True, na=False)] = "dns"
    fam.loc[blob.str.contains(r"flow|zeek|firewall|tls|cloud_flow|type8_firewall", regex=True, na=False)] = "network"
    fam.loc[blob.str.contains(r"edr|auditd|linux_auth|mac_es|fim|type3_proc|type6_event", regex=True, na=False)] = "endpoint"
    fam.loc[blob.str.contains(r"\b(?:idp|ad_auth|pam|aaa|mfa)\b|type9_dynamic", regex=True, na=False)] = "identity"
    fam.loc[blob.str.contains(r"cloud_audit|saas|gh_audit|cicd|db_audit|objstore|secrets|kms|dlp|casb", regex=True, na=False)] = "cloud"
    fam.loc[blob.str.contains(r"k8s|container|docker", regex=True, na=False)] = "container"
    fam.loc[blob.str.contains(r"email_sec", regex=True, na=False)] = "email"
    fam.loc[blob.str.contains(r"ics", regex=True, na=False)] = "ics"

    # Log type is a stronger family signal than free text. Raw threat strings
    # often contain phrases like "network flow direction" or "spammers", which
    # must not override parser-derived web/proxy family.
    fam.loc[logt.str.contains(r"proxy|apache|web|edge|waf|apigw|lb|envoy|type1_space|type4_web|type5_proxy", regex=True, na=False)] = "web"
    fam.loc[logt.str.contains(r"dns", regex=True, na=False)] = "dns"
    fam.loc[logt.str.contains(r"flow|zeek|firewall|tls|cloud_flow|type8_firewall", regex=True, na=False)] = "network"
    fam.loc[logt.str.contains(r"dhcp|type7_asset", regex=True, na=False)] = "dhcp"
    fam.loc[logt.str.contains(r"edr|auditd|linux_auth|mac_es|fim|type3_proc|type6_event", regex=True, na=False)] = "endpoint"
    fam.loc[logt.str.contains(r"\b(?:idp|ad_auth|pam|aaa|mfa)\b|type9_dynamic", regex=True, na=False)] = "identity"
    fam.loc[logt.str.contains(r"cloud_audit|saas|gh_audit|cicd|db_audit|objstore|secrets|kms|dlp|casb", regex=True, na=False)] = "cloud"
    fam.loc[logt.str.contains(r"k8s|container|docker", regex=True, na=False)] = "container"
    fam.loc[logt.str.contains(r"email_sec", regex=True, na=False)] = "email"
    fam.loc[logt.str.contains(r"ics", regex=True, na=False)] = "ics"
    return fam


def _resolver_container(resolver_state: Dict[str, Any]) -> list[Dict[str, Any]]:
    if not isinstance(resolver_state, dict):
        return []
    out = [resolver_state]
    for k in ("repair_maps", "maps", "resolver_maps", "advanced_imputation", "artifacts"):
        obj = resolver_state.get(k)
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _resolve_any_map(resolver_state: Dict[str, Any], names: tuple[str, ...], key: Any) -> str:
    if _is_missing_text_value(key):
        return ""
    key_s = str(key).strip()
    keys = [key_s, key_s.lower(), key_s.upper()]
    for cont in _resolver_container(resolver_state):
        for name in names:
            obj = cont.get(name)
            if isinstance(obj, dict):
                for k in keys:
                    val = _first_nonempty(obj.get(k))
                    if val:
                        return val
    return ""


def _looks_like_real_domain(dom: Any) -> bool:
    d = _normalize_domain_token(dom)
    if not d or d.endswith((".local", ".internal", ".invalid")):
        return False
    if _IPV4_RE_SAFE.match(d):
        return False
    if "." not in d:
        return False
    tld = d.rsplit(".", 1)[-1]
    return bool(2 <= len(tld) <= 24 and re.match(r"^[a-z0-9-]+$", tld))


def _synthetic_ip_from_seed(seed: Any, internal: bool = False) -> str:
    h = _stable_int(str(seed) + "|ip")
    if internal:
        return f"10.255.{(h >> 8) % 256}.{(h % 254) + 1}"
    # RFC 5737 TEST-NET-2: deterministic external placeholder, never 0.0.0.0.
    return f"198.51.100.{(h % 254) + 1}"


def _row_dest_ip_fallbacks(out: pd.DataFrame, resolver_state: Dict[str, Any] | None = None) -> pd.Series:
    idx = out.index
    resolver_state = resolver_state if isinstance(resolver_state, dict) else {}
    dom = out.get("domain", pd.Series("", index=idx)).fillna("").astype(str)
    cip = out.get("client_ip", pd.Series("", index=idx)).fillna("").astype(str)
    raw = out.get("raw_log", pd.Series("", index=idx)).fillna("").astype(str)
    fam = _family_hint_series(out)
    vals = pd.Series("", index=idx, dtype="object")
    # Artifact resolver maps win.
    for i in idx:
        d = dom.loc[i]
        mapped = _resolve_any_map(resolver_state, ("domain_to_ip", "dns", "domain_resolver", "host_to_ip", "domain_ip_map"), d)
        if mapped and not _is_missing_text_value(mapped):
            vals.loc[i] = str(mapped)
    miss = vals.map(_is_missing_text_value)
    if miss.any():
        real_dom = dom.map(_looks_like_real_domain)
        vals.loc[miss & real_dom] = (dom.loc[miss & real_dom] + "|" + raw.loc[miss & real_dom]).map(lambda x: _synthetic_ip_from_seed(x, internal=False))
        still = vals.map(_is_missing_text_value)
        vals.loc[still & fam.isin(["dhcp", "endpoint", "identity", "network", "container", "ics"])] = (cip.loc[still & fam.isin(["dhcp", "endpoint", "identity", "network", "container", "ics"])] + "|" + fam.loc[still & fam.isin(["dhcp", "endpoint", "identity", "network", "container", "ics"])]).map(lambda x: _synthetic_ip_from_seed(x, internal=True))
        still = vals.map(_is_missing_text_value)
        vals.loc[still] = (raw.loc[still] + "|dest").map(lambda x: _synthetic_ip_from_seed(x, internal=False))
    return vals


def _row_client_ip_fallbacks(out: pd.DataFrame) -> pd.Series:
    idx = out.index
    raw = out.get("raw_log", pd.Series("", index=idx)).fillna("").astype(str)
    fam = _family_hint_series(out)
    vals = pd.Series("", index=idx, dtype="object")
    vals.loc[fam.isin(["web", "dhcp", "endpoint", "identity", "network", "container", "ics"])] = (raw.loc[fam.isin(["web", "dhcp", "endpoint", "identity", "network", "container", "ics"])] + "|src").map(lambda x: _synthetic_ip_from_seed(x, internal=True))
    vals.loc[vals.map(_is_missing_text_value)] = (raw.loc[vals.map(_is_missing_text_value)] + "|src").map(lambda x: _synthetic_ip_from_seed(x, internal=True))
    return vals


def _row_workstation_fallbacks(out: pd.DataFrame, resolver_state: Dict[str, Any] | None = None) -> pd.Series:
    idx = out.index
    resolver_state = resolver_state if isinstance(resolver_state, dict) else {}
    cip = out.get("client_ip", pd.Series("", index=idx)).fillna("").astype(str)
    ua = out.get("user_agent", pd.Series("", index=idx)).fillna("").astype(str)
    username = out.get("username", pd.Series("", index=idx)).fillna("").astype(str)
    proc = out.get("process", pd.Series("", index=idx)).fillna("").astype(str)
    raw = out.get("raw_log", pd.Series("", index=idx)).fillna("").astype(str)
    fam = _family_hint_series(out)
    device = _ua_device_series(ua)
    vals = pd.Series("", index=idx, dtype="object")
    for i in idx:
        c = cip.loc[i]
        u = username.loc[i]
        key_ipua = f"{c}||{device.loc[i]}"
        mapped = _first_nonempty(
            _resolve_any_map(resolver_state, ("ipua_to_ws", "ipua_to_workstation", "ip_device_to_workstation"), key_ipua),
            _resolve_any_map(resolver_state, ("ip_to_ws", "ip_to_workstation", "client_ip_to_workstation"), c),
            _resolve_any_map(resolver_state, ("user_to_ws", "user_to_workstation", "username_to_workstation", "user_host"), u),
        )
        if mapped:
            vals.loc[i] = mapped
    miss = vals.map(_is_missing_text_value)
    if miss.any():
        for i in vals.index[miss]:
            site = _site_from_any("", cip.loc[i], raw.loc[i])
            h = _stable_code(str(cip.loc[i]) + "|" + str(ua.loc[i]) + "|" + str(username.loc[i]) + "|" + str(proc.loc[i]))
            kind = _ip_kind(cip.loc[i])
            prefix = {
                "web": "WEBCLIENT",
                "dhcp": "DHCPHOST",
                "network": "NETNODE",
                "endpoint": "ENDPOINT",
                "identity": "IDHOST",
                "cloud": "CLOUDNODE",
                "container": "K8SNODE",
                "email": "MAILNODE",
                "ics": "ICSNODE",
            }.get(str(fam.loc[i]), "HOST")
            if kind == "external":
                prefix = "EXT" + prefix
            vals.loc[i] = f"{prefix}-{site}-{h}"
    return vals


def _proc_from_command(command: Any) -> str:
    c = str(command or "")
    m = _EXE_RE.search(c)
    if m:
        token = m.group(1).lower()
        if token == "cmd":
            token = "cmd.exe"
        return token
    return ""


def _row_process_fallbacks(out: pd.DataFrame) -> pd.Series:
    idx = out.index
    cmd = out.get("command", pd.Series("", index=idx)).fillna("").astype(str)
    ua = out.get("user_agent", pd.Series("", index=idx)).fillna("").astype(str)
    method = out.get("method", pd.Series("", index=idx)).fillna("").astype(str).str.lower()
    fam = _family_hint_series(out)
    vals = cmd.map(_proc_from_command)
    miss = vals.map(_is_missing_text_value)
    if miss.any():
        dev = _ua_device_series(ua)
        for i in vals.index[miss]:
            f = str(fam.loc[i])
            if f == "web":
                vals.loc[i] = "scanner_client" if dev.loc[i] == "scanner" else f"http_{method.loc[i] or 'request'}_client"
            elif f == "dns":
                vals.loc[i] = "dns_query"
            elif f == "network":
                vals.loc[i] = "network_flow"
            elif f == "dhcp":
                vals.loc[i] = "dhcp_client"
            elif f == "endpoint":
                vals.loc[i] = "endpoint_event"
            elif f == "identity":
                vals.loc[i] = "identity_auth"
            elif f == "cloud":
                vals.loc[i] = "cloud_audit"
            elif f == "container":
                vals.loc[i] = "container_event"
            elif f == "email":
                vals.loc[i] = "email_security"
            elif f == "ics":
                vals.loc[i] = "ics_control"
            else:
                vals.loc[i] = "event_process"
    return vals.astype(str)


def _row_username_fallbacks(out: pd.DataFrame, resolver_state: Dict[str, Any] | None = None) -> pd.Series:
    idx = out.index
    resolver_state = resolver_state if isinstance(resolver_state, dict) else {}
    cip = out.get("client_ip", pd.Series("", index=idx)).fillna("").astype(str)
    ws = out.get("workstation", pd.Series("", index=idx)).fillna("").astype(str)
    proc = out.get("process", pd.Series("", index=idx)).fillna("").astype(str)
    raw = out.get("raw_log", pd.Series("", index=idx)).fillna("").astype(str)
    fam = _family_hint_series(out)
    vals = pd.Series("", index=idx, dtype="object")
    for i in idx:
        k_ws_proc = f"{ws.loc[i]}||{proc.loc[i]}"
        k_ip_proc = f"{cip.loc[i]}||{proc.loc[i]}"
        mapped = _first_nonempty(
            _resolve_any_map(resolver_state, ("wsproc_to_user", "workstation_process_to_user"), k_ws_proc),
            _resolve_any_map(resolver_state, ("ws_to_user", "workstation_to_user"), ws.loc[i]),
            _resolve_any_map(resolver_state, ("ipproc_to_user", "client_ip_process_to_user"), k_ip_proc),
            _resolve_any_map(resolver_state, ("ip_to_user", "client_ip_to_user", "ip_username"), cip.loc[i]),
        )
        if mapped:
            vals.loc[i] = mapped
    miss = vals.map(_is_missing_text_value)
    if miss.any():
        for i in vals.index[miss]:
            site = _site_from_any(ws.loc[i], cip.loc[i], raw.loc[i]).lower()
            role = {
                "web": "web_user",
                "dhcp": "device_user",
                "network": "net_user",
                "endpoint": "endpoint_user",
                "identity": "identity_user",
                "cloud": "cloud_user",
                "container": "container_user",
                "email": "mail_user",
                "ics": "ics_user",
            }.get(str(fam.loc[i]), "event_user")
            if re.search(r"(?i)system|svchost|lsass|winlogon|wevtutil|services", str(proc.loc[i])):
                role = "SYSTEM"
                vals.loc[i] = role
            else:
                vals.loc[i] = f"{role}_{site}_{_stable_code(str(ws.loc[i]) + '|' + str(cip.loc[i]) + '|' + str(proc.loc[i]), 6)}"
    return vals.astype(str)


def _row_method_fallbacks(out: pd.DataFrame) -> pd.Series:
    idx = out.index
    raw = out.get("raw_log", pd.Series("", index=idx)).fillna("").astype(str)
    fam = _family_hint_series(out)
    vals = pd.Series("GET", index=idx, dtype="object")
    vals.loc[fam.isin(["identity", "cloud", "endpoint", "container", "email", "ics"])] = "POST"
    vals.loc[fam.eq("dns")] = "GET"
    # Try to extract an HTTP verb from raw before family fallback.
    m = raw.str.extract(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b", flags=re.I, expand=False)
    use = m.notna()
    vals.loc[use] = m.loc[use].str.upper()
    return vals


def _row_url_path_fallbacks(out: pd.DataFrame) -> pd.Series:
    idx = out.index
    raw = out.get("raw_log", pd.Series("", index=idx)).fillna("").astype(str)
    fam = _family_hint_series(out)
    vals = pd.Series("/", index=idx, dtype="object")
    vals.loc[fam.eq("dns")] = "/dns/query"
    vals.loc[fam.eq("network")] = "/network/flow"
    vals.loc[fam.eq("dhcp")] = "/dhcp/lease"
    vals.loc[fam.eq("endpoint")] = "/endpoint/event"
    vals.loc[fam.eq("identity")] = "/identity/auth"
    vals.loc[fam.eq("cloud")] = "/cloud/audit"
    vals.loc[fam.eq("container")] = "/container/event"
    vals.loc[fam.eq("email")] = "/email/security"
    vals.loc[fam.eq("ics")] = "/ics/event"
    path = raw.str.extract(r"\b(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+([^\s\"]+)", flags=re.I, expand=False)
    use = path.notna()
    vals.loc[use] = path.loc[use].astype(str)
    return vals


def _prepare_top_domain_values(top_domains: Any) -> tuple[str, ...]:
    """Normalize/sort top-domain artifacts once, not once per row.

    v6 sorted the full top-domain set inside _pick_top_domain() for every
    missing domain/referrer. With top-1m artifacts, even 100 logs could appear
    stuck in "Processing logs".
    """
    if not top_domains:
        return tuple()
    if isinstance(top_domains, tuple):
        return top_domains
    vals = []
    seen = set()
    try:
        iterator = iter(top_domains)
    except Exception:
        iterator = iter([top_domains])
    try:
        cap = int(os.getenv("TOP_DOMAIN_IMPUTE_CAP", "50000") or "50000")
    except Exception:
        cap = 50000
    cap = max(1000, min(cap, 250000))
    for x in iterator:
        v = _normalize_domain_token(x)
        if v and v not in seen:
            seen.add(v)
            vals.append(v)
            if len(vals) >= cap:
                break
    vals.sort()
    return tuple(vals)


def _pick_top_domain(seed: Any, top_domains: Any) -> str:
    vals = top_domains if isinstance(top_domains, tuple) else _prepare_top_domain_values(top_domains)
    if not vals:
        return ""
    return vals[_stable_int(seed) % len(vals)]


def _row_domain_fallbacks(out: pd.DataFrame, top_domains: set[str]) -> pd.Series:
    """Typed domain fallback. Uses top domains only for non-suspicious web/proxy rows.

    This avoids the old one-token fill and prevents top-domain artifacts from
    hiding security/endpoint/DHCP/network semantics.
    """
    idx = out.index
    logt = out.get("log_type", pd.Series("", index=idx)).fillna("").astype(str).str.lower()
    raw = out.get("raw_log", pd.Series("", index=idx)).fillna("").astype(str)
    cmd = out.get("command", pd.Series("", index=idx)).fillna("").astype(str)
    proc = out.get("process", pd.Series("", index=idx)).fillna("").astype(str)
    ua = out.get("user_agent", pd.Series("", index=idx)).fillna("").astype(str)
    path = out.get("url_path", pd.Series("", index=idx)).fillna("").astype(str)
    dip = out.get("dest_ip", pd.Series("", index=idx)).fillna("").astype(str)
    cip = out.get("client_ip", pd.Series("", index=idx)).fillna("").astype(str)
    seed = raw + "|" + cip + "|" + dip + "|" + path + "|" + ua
    blob = (raw + " " + cmd + " " + proc + " " + ua + " " + path).str.lower()

    suspicious = blob.str.contains(r"malware|phish|trojan|c2|beacon|ransom|mimikatz|lsass|/\.env|wp-admin|union\s+select|powershell\s+-enc|certutil|bitsadmin|msxsl|curl\s+.*\|\s*(?:sh|bash)|nc\s+\d+\.\d+\.\d+\.\d+\s+4444", regex=True, na=False)
    web_like = logt.str.contains(r"proxy|apache|web|edge|waf|lb|apigw|envoy|type1_space|type4_web_csv|type5_proxy_csv", regex=True, na=False)
    network_like = logt.str.contains(r"flow|zeek|tls|dns|firewall|cloud_flow|type8_firewall_csv", regex=True, na=False)
    dhcp_like = logt.str.contains(r"dhcp|type7_asset_csv", regex=True, na=False) | blob.str.contains(r"\b(?:renew|assign|dhcpack|dhcprequest|dhcpoffer)\b", regex=True, na=False)
    os_like = logt.str.contains(r"linux_auth|auditd|mac_es|fim|edr|type3_proc_csv|type6_event_csv", regex=True, na=False)
    identity_like = logt.str.contains(r"idp|ad_auth|mfa|aaa|pam|type9_dynamic_csv", regex=True, na=False)
    cloud_like = logt.str.contains(r"cloud|saas|gh_audit|cicd|db_audit|objstore|secrets|kms|dlp|casb", regex=True, na=False)
    container_like = logt.str.contains(r"k8s|container|docker", regex=True, na=False)

    vals = pd.Series("event.local", index=idx, dtype="object")
    vals.loc[dhcp_like] = "dhcp.local"
    vals.loc[os_like] = "endpoint.local"
    vals.loc[identity_like] = "identity.local"
    vals.loc[cloud_like] = "cloud.local"
    vals.loc[container_like] = "container.local"
    vals.loc[network_like] = dip.loc[network_like].map(_ip_pseudo_domain)

    # Public web/proxy rows may use top-domain artifacts if clean. Suspicious
    # web/proxy rows get a neutral external placeholder so they are not hidden.
    clean_web = web_like & (~suspicious)
    if clean_web.any() and top_domains:
        vals.loc[clean_web] = seed.loc[clean_web].map(lambda x: _pick_top_domain(x, top_domains) or "web.local")
    vals.loc[web_like & suspicious] = "suspicious.external"
    vals.loc[web_like & (~clean_web) & (~suspicious)] = vals.loc[web_like & (~clean_web) & (~suspicious)].replace("event.local", "web.local")

    # If we have a concrete public dest IP, expose it as a pseudo-domain unless
    # a more semantic family value is already in place.
    miss_semantic = vals.isin(["event.local", "network.local", "web.local"])
    has_dip = dip.str.len().gt(0) & ~dip.map(_is_missing_text_value)
    vals.loc[miss_semantic & has_dip] = dip.loc[miss_semantic & has_dip].map(_ip_pseudo_domain)
    return vals


def _row_referrer_fallbacks(out: pd.DataFrame, top_domains: set[str]) -> pd.Series:
    idx = out.index
    raw = out.get("raw_log", pd.Series("", index=idx)).fillna("").astype(str)
    cip = out.get("client_ip", pd.Series("", index=idx)).fillna("").astype(str)
    dom = out.get("domain", pd.Series("", index=idx)).fillna("").astype(str)
    logt = out.get("log_type", pd.Series("", index=idx)).fillna("").astype(str).str.lower()
    web_like = logt.str.contains(r"proxy|apache|web|edge|waf|lb|apigw|envoy|type1_space|type4_web_csv|type5_proxy_csv", regex=True, na=False)
    vals = pd.Series("direct", index=idx, dtype="object")
    if top_domains:
        seed = raw + "|" + cip + "|" + dom + "|ref"
        picked = seed.map(lambda x: _pick_top_domain(x, top_domains))
        vals.loc[web_like & picked.str.len().gt(0)] = "https://" + picked.loc[web_like & picked.str.len().gt(0)] + "/"
    return vals


def _content_group_series(path_s: pd.Series, url_s: pd.Series) -> pd.Series:
    text = (path_s.fillna("").astype(str) + " " + url_s.fillna("").astype(str)).str.lower()
    return pd.Series(np.select(
        [
            text.str.contains(r"upload|multipart|import|put|attach", regex=True, na=False),
            text.str.contains(r"download|export|backup|dump|\.zip|\.gz|\.tgz|\.csv|\.xlsx|\.pdf", regex=True, na=False),
            text.str.contains(r"api|auth|login|token|session", regex=True, na=False),
            text.str.contains(r"\.css|\.js|\.png|\.jpg|\.jpeg|\.gif|\.svg|static|assets", regex=True, na=False),
        ],
        ["upload", "download", "api", "static"],
        default="page",
    ), index=path_s.index)


def _deterministic_lognormal(seed_s: pd.Series, median: float, sigma: float, clip_max: int) -> pd.Series:
    vals = []
    mu = float(np.log1p(max(median, 0.0)))
    sig = max(float(sigma), 0.05)
    for seed in seed_s.astype(str).tolist():
        # Deterministic pseudo-normal via two stable uniforms.
        a = ((_stable_int(seed + "|a") % 1_000_000) + 1) / 1_000_001.0
        b = ((_stable_int(seed + "|b") % 1_000_000) + 1) / 1_000_001.0
        z = np.sqrt(-2.0 * np.log(a)) * np.cos(2.0 * np.pi * b)
        vals.append(int(min(max(np.expm1(mu + sig * z), 0.0), clip_max)))
    return pd.Series(vals, index=seed_s.index, dtype="float64")


def _impute_bytes_contextual(out: pd.DataFrame, byte_priors: dict | None = None) -> pd.DataFrame:
    """Deterministic, context-aware bytes filling inspired by the training notebook.

    Artifact priors win. When absent, values are generated from method/status/content
    class instead of zeroing everything.
    """
    method = out.get("method", pd.Series("GET", index=out.index)).fillna("GET").astype(str).str.upper()
    status = pd.to_numeric(out.get("status", pd.Series(200, index=out.index)), errors="coerce").fillna(200).astype(int)
    path = out.get("url_path", pd.Series("/", index=out.index)).fillna("/").astype(str)
    url = out.get("full_url", pd.Series("", index=out.index)).fillna("").astype(str)
    dom = out.get("domain", pd.Series("", index=out.index)).fillna("").astype(str)
    seed = out.get("raw_log", pd.Series("", index=out.index)).fillna("").astype(str) + "|" + method + "|" + status.astype(str) + "|" + dom + "|" + path
    group = _content_group_series(path, url)

    # bytes_out = response/outbound from server perspective in web data, but also
    # exfil/outbound in flow data. Keep observed semantics; only fill missing.
    bo = pd.to_numeric(out["bytes_out"], errors="coerce") if "bytes_out" in out.columns else pd.Series(np.nan, index=out.index)
    bi = pd.to_numeric(out["bytes_in"], errors="coerce") if "bytes_in" in out.columns else pd.Series(np.nan, index=out.index)
    bo = bo.where(bo >= 0, np.nan)
    bi = bi.where(bi >= 0, np.nan)

    no_body = method.eq("HEAD") | status.isin([204, 304])
    if no_body.any():
        bo.loc[bo.isna() & no_body] = 0
        bi.loc[bi.isna() & no_body] = 80

    pri = byte_priors if isinstance(byte_priors, dict) else {}
    if pri:
        # Accept common artifact shapes: default_bytes_*, median maps, or method/status group maps.
        for target, series in [("bytes_out", bo), ("bytes_in", bi)]:
            miss = series.isna()
            if not miss.any():
                continue
            default_key = f"default_{target}"
            try:
                default_val = float(pri.get(default_key, np.nan))
            except Exception:
                default_val = np.nan
            map_key = f"{target}_median_by_key"
            med_map = pri.get(map_key) if isinstance(pri.get(map_key), dict) else {}
            if med_map:
                k = method + "|" + status.astype(str).str[0] + "xx|" + group
                mapped = k.map(med_map)
                use = miss & mapped.notna()
                if use.any():
                    series.loc[use] = pd.to_numeric(mapped.loc[use], errors="coerce")
            if np.isfinite(default_val):
                series.loc[series.isna()] = default_val

    miss_bo = bo.isna()
    if miss_bo.any():
        med = pd.Series(1800.0, index=out.index)
        med.loc[group.eq("static")] = 900.0
        med.loc[group.eq("api")] = 2400.0
        med.loc[group.eq("download")] = 180_000.0
        med.loc[group.eq("upload")] = 3200.0
        med.loc[method.isin(["POST", "PUT", "PATCH"])] = med.loc[method.isin(["POST", "PUT", "PATCH"])].clip(lower=2600.0)
        bo.loc[miss_bo] = _deterministic_lognormal(seed.loc[miss_bo] + "|bo", median=float(med.loc[miss_bo].median() if len(med.loc[miss_bo]) else 1800), sigma=0.9, clip_max=250_000_000).values

    miss_bi = bi.isna()
    if miss_bi.any():
        med = pd.Series(650.0, index=out.index)
        med.loc[group.eq("api")] = 1500.0
        med.loc[group.eq("upload")] = 80_000.0
        med.loc[group.eq("download")] = 900.0
        med.loc[method.isin(["POST", "PUT", "PATCH"])] = med.loc[method.isin(["POST", "PUT", "PATCH"])].clip(lower=2200.0)
        bi.loc[miss_bi] = _deterministic_lognormal(seed.loc[miss_bi] + "|bi", median=float(med.loc[miss_bi].median() if len(med.loc[miss_bi]) else 650), sigma=1.0, clip_max=250_000_000).values

    out["bytes_out"] = bo.fillna(0).clip(lower=0)
    out["bytes_in"] = bi.fillna(0).clip(lower=0)
    return out

@dataclass
class ForensicImputer:
    priors: Dict[str, Any] = field(default_factory=dict)
    resolver_state: Dict[str, Any] = field(default_factory=dict)
    bytes_priors: Dict[str, Any] = field(default_factory=dict)
    top_domains: set[str] = field(default_factory=set)
    _top_domain_values: tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)

    def __post_init__(self) -> None:
        # Prepared lazily: a top-1m set can be large, and many batches already
        # contain domains so no top-domain fallback is needed.
        self._top_domain_values = tuple()

    def _get_top_domain_values(self) -> tuple[str, ...]:
        if not self._top_domain_values and self.top_domains:
            self._top_domain_values = _prepare_top_domain_values(self.top_domains)
        return self._top_domain_values

    def _artifact_default(self, col: str) -> Optional[str]:
        """Resolve column defaults from bundled imputation artifacts/priors.

        Supported artifact shapes are intentionally broad so this works with
        column_defaults.pkl, imputation_defaults.pkl, training priors, resolver
        state, or nested {column: {default/mode/value: ...}} structures.
        """
        sources = []
        for container in (self.priors, self.resolver_state):
            if not isinstance(container, dict):
                continue
            sources.append(container)
            for key in _DEFAULT_TEXT_PRIOR_KEYS:
                nested = container.get(key)
                if isinstance(nested, dict):
                    sources.append(nested)
        for src in sources:
            val = _first_prior_value(src, col)
            cleaned = _clean_prior_default(val)
            if cleaned is not None:
                return cleaned
        return None

    def _text_default(self, col: str, out: pd.DataFrame) -> str:
        prior = self._artifact_default(col)
        if prior is not None:
            return prior

        # Column-aware forensic fallbacks. These are not one global hardcoded
        # token; they preserve feature semantics and keep downstream models stable.
        if col == "timestamp":
            # Prefer already-derived raw timestamp if present, otherwise use a
            # neutral UTC epoch-like instant that is explicit and parseable.
            if "timestamp_raw" in out.columns:
                vals = out["timestamp_raw"].fillna("").astype(str).str.strip()
                vals = vals[~vals.map(_is_missing_text_value)]
                if len(vals):
                    return str(vals.iloc[0])
            return datetime(1970, 1, 1, tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")
        if col == "command" and "raw_log" in out.columns:
            vals = out["raw_log"].fillna("").astype(str).str.strip()
            vals = vals[~vals.map(_is_missing_text_value)]
            if len(vals):
                return str(vals.iloc[0])[:512]
        return _COLUMN_FALLBACKS.get(col, f"missing_{col}")

    def impute_df(self, df: pd.DataFrame, fill_text: Optional[str] = None) -> pd.DataFrame:
        """Impute missing parsed-log values.

        fill_text is kept for backward compatibility. When None, every text
        column is filled from artifact priors/resolver state or column-aware
        forensic defaults instead of the old single NotProvided token.
        """
        out = df.copy()

        for c in PRIMARY_COLS:
            if c not in out.columns:
                out[c] = np.nan if c in _NUMERIC_COLS else ""

        # Normalize / strip core strings.
        for c in _TEXT_COLS:
            out[c] = out[c].fillna("").astype(str).str.strip()

        # Domain / URL / path salvage.
        out["full_url"] = out["full_url"].fillna("").astype(str).str.strip()
        out["url_path"] = out["url_path"].fillna("").astype(str).str.strip()
        out["domain"] = sanitize_domain_series(out.get("domain", pd.Series("", index=out.index)))

        dom_from_url = out["full_url"].where(out["full_url"].astype(str).str.startswith(("http://", "https://"), na=False), "").map(_domain_from_url)
        out["domain"] = out["domain"].fillna(dom_from_url.mask(dom_from_url.eq(""), np.nan))

        path_from_url = out["full_url"].where(out["full_url"].astype(str).str.startswith(("http://", "https://"), na=False), "").map(_path_from_url)
        out["url_path"] = out["url_path"].mask(out["url_path"].astype(str).str.strip().eq(""), path_from_url)

        if "referrer" in out.columns:
            ref_dom = out["referrer"].fillna("").astype(str).map(_domain_from_url)
            out["domain"] = out["domain"].fillna(ref_dom.mask(ref_dom.eq(""), np.nan))

        raw_host = out["raw_log"].fillna("").astype(str).map(_scan_host_from_text)
        out["domain"] = out["domain"].fillna(raw_host.mask(raw_host.eq(""), np.nan))

        # Salvage full_url from domain + path when possible.
        missing_full = out["full_url"].fillna("").astype(str).map(_is_missing_text_value)
        can_build = missing_full & out["domain"].fillna("").astype(str).ne("") & out["url_path"].fillna("").astype(str).ne("")
        if can_build.any():
            path = out.loc[can_build, "url_path"].astype(str)
            prefix = np.where(path.str.startswith("/"), "", "/")
            out.loc[can_build, "full_url"] = "http://" + out.loc[can_build, "domain"].astype(str) + prefix + path

        # Last-resort domain salvage from dest_ip if public.
        miss_dom = out["domain"].isna() | out["domain"].astype(str).str.strip().eq("")
        if miss_dom.any():
            dip = out.loc[miss_dom, "dest_ip"].fillna("").astype(str).str.strip()
            public_mask = dip.map(_is_public_ip)
            if public_mask.any():
                idx = dip.index[public_mask]
                out.loc[idx, "domain"] = dip.loc[idx].values

        # Artifact resolver-state enrichments, if available. These maps are
        # loaded from resolver_state.pkl/domain resolver artifacts by artifacts.load_bundle().
        if "domain" in out.columns and "dest_ip" in out.columns:
            miss_dom2 = _is_missing_text_series(out["domain"])
            if miss_dom2.any():
                mapped = out.loc[miss_dom2, "dest_ip"].map(lambda x: _resolve_map(self.resolver_state, ("ip_to_domain", "reverse_dns", "dest_ip_to_domain"), x))
                out.loc[miss_dom2, "domain"] = out.loc[miss_dom2, "domain"].mask(_is_missing_text_series(out.loc[miss_dom2, "domain"]), mapped)
        if "dest_ip" in out.columns and "domain" in out.columns:
            miss_dip2 = _is_missing_text_series(out["dest_ip"])
            if miss_dip2.any():
                mapped = out.loc[miss_dip2, "domain"].map(lambda x: _resolve_map(self.resolver_state, ("domain_to_ip", "dns", "domain_resolver", "host_to_ip"), x))
                out.loc[miss_dip2, "dest_ip"] = out.loc[miss_dip2, "dest_ip"].mask(_is_missing_text_series(out.loc[miss_dip2, "dest_ip"]), mapped)
        if "workstation" in out.columns and "username" in out.columns:
            miss_ws2 = _is_missing_text_series(out["workstation"])
            if miss_ws2.any():
                mapped = out.loc[miss_ws2, "username"].map(lambda x: _resolve_map(self.resolver_state, ("user_to_workstation", "username_to_workstation", "user_host"), x))
                out.loc[miss_ws2, "workstation"] = out.loc[miss_ws2, "workstation"].mask(_is_missing_text_series(out.loc[miss_ws2, "workstation"]), mapped)

        # Row-context semantic domain/referrer fallback. Top-domain artifacts are
        # used only for clean web/proxy traffic, never for security/OS/DHCP rows.
        miss_dom3 = _is_missing_text_series(out["domain"])
        if miss_dom3.any():
            row_dom = _row_domain_fallbacks(out, self._get_top_domain_values())
            out.loc[miss_dom3, "domain"] = row_dom.loc[miss_dom3]

        if "referrer" in out.columns:
            miss_ref = _is_missing_text_series(out["referrer"])
            if miss_ref.any():
                row_ref = _row_referrer_fallbacks(out, self._get_top_domain_values())
                out.loc[miss_ref, "referrer"] = row_ref.loc[miss_ref]

        # Primary-column repair pass: no unresolved placeholders.
        # Resolver/artifact maps win, then deterministic row-context values.
        miss_cip = _is_missing_text_series(out["client_ip"])
        if miss_cip.any():
            prior = self._artifact_default("client_ip")
            if prior and not _is_missing_text_value(prior):
                out.loc[miss_cip, "client_ip"] = prior
            else:
                out.loc[miss_cip, "client_ip"] = _row_client_ip_fallbacks(out).loc[miss_cip]

        miss_dip3 = _is_missing_text_series(out["dest_ip"])
        if miss_dip3.any():
            prior = self._artifact_default("dest_ip")
            if prior and not _is_missing_text_value(prior):
                out.loc[miss_dip3, "dest_ip"] = prior
            else:
                out.loc[miss_dip3, "dest_ip"] = _row_dest_ip_fallbacks(out, self.resolver_state).loc[miss_dip3]

        miss_method = _is_missing_text_series(out["method"])
        if miss_method.any():
            prior = self._artifact_default("method")
            out.loc[miss_method, "method"] = prior if prior and not _is_missing_text_value(prior) else _row_method_fallbacks(out).loc[miss_method]

        miss_path = _is_missing_text_series(out["url_path"])
        if miss_path.any():
            prior = self._artifact_default("url_path")
            out.loc[miss_path, "url_path"] = prior if prior and not _is_missing_text_value(prior) else _row_url_path_fallbacks(out).loc[miss_path]

        ws_s = out["workstation"].fillna("").astype(str).str.strip()
        miss_ws = _is_missing_text_series(ws_s) | ws_s.str.match(_WEAK_WORKSTATION_RE, na=False)
        if miss_ws.any():
            prior = self._artifact_default("workstation")
            out.loc[miss_ws, "workstation"] = prior if prior and not _is_missing_text_value(prior) else _row_workstation_fallbacks(out, self.resolver_state).loc[miss_ws]

        proc_s = out["process"].fillna("").astype(str).str.strip()
        miss_proc = _is_missing_text_series(proc_s) | proc_s.str.match(r"^(?:event_process|unknown_process|unresolved-process|process)$", flags=re.I, na=False)
        if miss_proc.any():
            prior = self._artifact_default("process")
            out.loc[miss_proc, "process"] = prior if prior and not _is_missing_text_value(prior) else _row_process_fallbacks(out).loc[miss_proc]

        user_s = out["username"].fillna("").astype(str).str.strip()
        miss_user = _is_missing_text_series(user_s) | user_s.str.lower().isin(_GENERIC_USER_TOKENS)
        if miss_user.any():
            prior = self._artifact_default("username")
            out.loc[miss_user, "username"] = prior if prior and not _is_missing_text_value(prior) else _row_username_fallbacks(out, self.resolver_state).loc[miss_user]

        # Rebuild URL after resolver/domain/IP enrichment.
        missing_full2 = out["full_url"].fillna("").astype(str).map(_is_missing_text_value)
        can_build2 = missing_full2 & (~_is_missing_text_series(out["domain"])) & (~_is_missing_text_series(out["url_path"]))
        if can_build2.any():
            path = out.loc[can_build2, "url_path"].astype(str)
            prefix = np.where(path.str.startswith("/"), "", "/")
            out.loc[can_build2, "full_url"] = "http://" + out.loc[can_build2, "domain"].astype(str) + prefix + path

        # Numeric normalization.
        for c in _NUMERIC_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        # Status defaults to training-safe neutral code, but artifact priors win.
        default_status = 200
        if isinstance(self.priors, dict):
            val = self._artifact_default("status") or self.priors.get("default_status", default_status)
            try:
                default_status = int(float(val))
            except Exception:
                default_status = 200
        out["status"] = out["status"].fillna(default_status).clip(lower=0)

        # Bytes defaults: artifact priors first, then deterministic context-based
        # imputation by method/status/content type. Avoid blanket zeroing.
        out = _impute_bytes_contextual(out, self.bytes_priors)

        # Fill remaining text columns from artifacts/column-aware defaults.
        # IMPORTANT: row-dependent defaults must be computed per row. v4 used a
        # scalar command default from the first raw_log in the whole batch, which
        # contaminated unrelated proxy/firewall/windows rows and caused broad
        # false-positive primary overrides.
        explicit_fill = str(fill_text).strip() if fill_text is not None and str(fill_text).strip() else ""
        for c in _TEXT_COLS:
            s = out[c].fillna("").astype(str).str.strip()
            s = s.replace({"nan": "", "None": "", "NULL": "", "null": ""})
            miss = s.map(_is_missing_text_value)
            if miss.any():
                if explicit_fill:
                    s.loc[miss] = explicit_fill
                elif c == "timestamp" and "timestamp_raw" in out.columns:
                    per = out.loc[miss, "timestamp_raw"].fillna("").astype(str).str.strip()
                    per = per.mask(per.map(_is_missing_text_value), self._text_default(c, out))
                    s.loc[miss] = per
                elif c == "command" and "raw_log" in out.columns:
                    per = out.loc[miss, "raw_log"].fillna("").astype(str).str.strip().str.slice(0, 512)
                    per = per.mask(per.map(_is_missing_text_value), self._text_default(c, out))
                    s.loc[miss] = per
                elif c == "raw_log":
                    sig_cols = [cc for cc in ["timestamp", "client_ip", "dest_ip", "method", "domain", "url_path", "process", "command"] if cc in out.columns]
                    if sig_cols:
                        per = out.loc[miss, sig_cols].astype(str).agg(" | ".join, axis=1)
                    else:
                        per = pd.Series(self._text_default(c, out), index=out.index[miss])
                    s.loc[miss] = per
                else:
                    s.loc[miss] = self._text_default(c, out)
            out[c] = s

        # Friendly log_type fallback from artifact/default logic.
        out["log_type"] = out["log_type"].replace("", np.nan).fillna(self._text_default("log_type", out)).astype(str)

        # Keep raw text intact when present; otherwise derive a compact forensic row signature.
        raw_missing = out["raw_log"].fillna("").astype(str).map(_is_missing_text_value)
        if raw_missing.any():
            sig_cols = [c for c in ["timestamp", "client_ip", "dest_ip", "method", "domain", "url_path", "process", "command"] if c in out.columns]
            out.loc[raw_missing, "raw_log"] = out.loc[raw_missing, sig_cols].astype(str).agg(" | ".join, axis=1)

        return out
