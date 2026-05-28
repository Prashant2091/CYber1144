
from __future__ import annotations

import ipaddress
import re
from functools import lru_cache

import numpy as np
import pandas as pd

_TUNNEL_HINT_RE = re.compile(r"(?i)\b(?:teredo|6to4|isatap|nat64|dns64|tunnel|wireguard|openvpn|ipsec|relay|::ffff:|64:ff9b::|2002:|2001:0000:)\b")
_SUSPICIOUS_TEXT_RE = re.compile(r"(?i)\b(?:payload|download|upload|exfil|powershell|pwsh|cmd\.exe|wget|curl|certutil|bitsadmin|bot|crawler|scanner|beacon|heartbeat|callback)\b")
_NAT64_WKP = ipaddress.ip_network("64:ff9b::/96")

@lru_cache(maxsize=200_000)
def _ipv6_flags_one(ip_str: str) -> tuple[int, int, int, int, int, int]:
    """
    Returns:
      present, mapped, six_to_four, teredo, isatap, nat64
    """
    ip_str = (ip_str or "").strip().strip("[]")
    if not ip_str:
        return (0, 0, 0, 0, 0, 0)
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except Exception:
        return (0, 0, 0, 0, 0, 0)

    if isinstance(ip_obj, ipaddress.IPv4Address):
        return (0, 0, 0, 0, 0, 0)

    packed = ip_obj.packed
    present = 1
    mapped = 1 if getattr(ip_obj, "ipv4_mapped", None) is not None else 0
    six_to_four = 1 if packed[0:2] == b"\x20\x02" else 0
    teredo = 1 if packed[0:4] == b"\x20\x01\x00\x00" else 0
    isatap = 1 if (len(packed) == 16 and packed[8:10] == b"\x00\x00" and packed[10:12] == b"\x5e\xfe") else 0
    nat64 = 1 if ip_obj in _NAT64_WKP else 0
    return (present, mapped, six_to_four, teredo, isatap, nat64)

def _apply_flags(series: pd.Series) -> pd.DataFrame:
    s = series.fillna("").astype(str).str.strip()
    codes, uniq = pd.factorize(s, sort=False)
    vals = [_ipv6_flags_one(u) for u in uniq]
    arr = np.asarray(vals, dtype=np.int8)
    return pd.DataFrame({
        "ipv6_present": arr[codes, 0],
        "ipv6_is_mapped": arr[codes, 1],
        "ipv6_is_6to4": arr[codes, 2],
        "ipv6_is_teredo": arr[codes, 3],
        "ipv6_is_isatap": arr[codes, 4],
        "ipv6_is_nat64": arr[codes, 5],
    }, index=series.index)

def add_ipv6_primary_conditions(df: pd.DataFrame, raw_col: str = "raw_log") -> pd.DataFrame:
    out = df.copy()

    cip = out.get("client_ip", pd.Series("", index=out.index)).fillna("").astype(str)
    dip = out.get("dest_ip", pd.Series("", index=out.index)).fillna("").astype(str)
    raw = out.get(raw_col, pd.Series("", index=out.index)).fillna("").astype(str)
    ua = out.get("user_agent", pd.Series("", index=out.index)).fillna("").astype(str)
    cmd = out.get("command", pd.Series("", index=out.index)).fillna("").astype(str)
    proc = out.get("process", pd.Series("", index=out.index)).fillna("").astype(str)
    dom = out.get("domain", pd.Series("", index=out.index)).fillna("").astype(str)
    url = out.get("full_url", pd.Series("", index=out.index)).fillna("").astype(str)

    c = _apply_flags(cip)
    d = _apply_flags(dip)

    out["ipv6_present"] = (c["ipv6_present"] | d["ipv6_present"]).astype(np.int8)
    out["ipv6_is_mapped"] = (c["ipv6_is_mapped"] | d["ipv6_is_mapped"]).astype(np.int8)
    out["ipv6_is_6to4"] = (c["ipv6_is_6to4"] | d["ipv6_is_6to4"]).astype(np.int8)
    out["ipv6_is_teredo"] = (c["ipv6_is_teredo"] | d["ipv6_is_teredo"]).astype(np.int8)
    out["ipv6_is_isatap"] = (c["ipv6_is_isatap"] | d["ipv6_is_isatap"]).astype(np.int8)
    out["ipv6_is_nat64"] = (c["ipv6_is_nat64"] | d["ipv6_is_nat64"]).astype(np.int8)

    out["ipv6_mapped_any"] = out["ipv6_is_mapped"].astype(np.int8)
    out["ipv6_6to4_any"] = out["ipv6_is_6to4"].astype(np.int8)
    out["ipv6_teredo_any"] = out["ipv6_is_teredo"].astype(np.int8)
    out["ipv6_isatap_any"] = out["ipv6_is_isatap"].astype(np.int8)
    out["ipv6_nat64_any"] = out["ipv6_is_nat64"].astype(np.int8)
    out["ipv6_present_any"] = out["ipv6_present"].astype(np.int8)

    raw_blob = (raw + " " + ua + " " + cmd + " " + proc + " " + dom + " " + url)
    tunnel_hint = raw_blob.str.contains(_TUNNEL_HINT_RE, na=False)
    suspicious_text = raw_blob.str.contains(_SUSPICIOUS_TEXT_RE, na=False)
    ip_bad = pd.to_numeric(out.get("ip_bad_truth", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int) > 0

    tunnel_any = (
        (out["ipv6_is_mapped"].astype(int) > 0)
        | (out["ipv6_is_6to4"].astype(int) > 0)
        | (out["ipv6_is_teredo"].astype(int) > 0)
        | (out["ipv6_is_isatap"].astype(int) > 0)
        | (out["ipv6_is_nat64"].astype(int) > 0)
        | tunnel_hint.to_numpy(dtype=bool)
    )

    out["ipv6_tunnel_any"] = tunnel_any.astype(np.int8)

    primary = tunnel_any & (suspicious_text.to_numpy(dtype=bool) | ip_bad.to_numpy(dtype=bool) | tunnel_hint.to_numpy(dtype=bool))
    out["ipv6_primary_flag"] = primary.astype(np.int8)

    return out
