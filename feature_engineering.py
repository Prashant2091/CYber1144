from __future__ import annotations

# Auto-generated from the uploaded training notebook and patched for local inference use.
# Original notebook source: cyber-new22.ipynb (feature engineering cell).

#Feature engineering model training 
# ============================================================
# ✅ SINGLE-CELL: UPDATED FEATURE ENGINEERING (FULL, NO-LOSS)
# ------------------------------------------------------------
# Adds:
# 1) ✅ IPv6 tunneling / masked IPv4 unmasking (mapped/6to4/teredo/isatap/nat64)
# 2) ✅ Odd-hours from RAW wall-time (NOT forced UTC) unless suspicious VPN/tunnel/bot
# 3) ✅ Whitelist domain features (hit + anti-spoof combo)
#
# Keeps:
# - Our full original feature set (UEBA 9 + all baseline features)
# - Our caches (geo db, ua freq, global byte stats)
# - Our output columns
# ============================================================

import os, gc, re, glob, ipaddress, urllib.parse
from functools import lru_cache
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

try:
    from feature_schema import RUNTIME_FEATURE_COLUMNS
except Exception:  # pragma: no cover
    RUNTIME_FEATURE_COLUMNS = []

# ----------------------------
# Global scaler (fit once, then transform)
# ----------------------------
scaler = RobustScaler()

# default geo-db path (Kaggle)
def _pick_first_existing_path(candidates):
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ip_geo_csv_path = _pick_first_existing_path([
    os.getenv("IP_GEO_CSV", ""),
    os.path.join(_THIS_DIR, "data", "ip-geo-country-ipv4.csv"),
    os.path.join(_THIS_DIR, "data", "ip_geo_country_ipv4.csv"),
    "/kaggle/input/defaults/ip-geo-country-ipv4.csv",
])
whitelist_csv = _pick_first_existing_path([
    os.getenv("WHITELIST_CSV", ""),
    os.path.join(_THIS_DIR, "data", "top-1m.csv"),
    os.path.join(_THIS_DIR, "data", "whitelist.csv"),
    "/kaggle/input/white-domain-top-lists/top-1m.csv",
])
# ============================================================
# TIMEZONE / ODD-HOURS (AUTO, wall-time first)
# ============================================================
SUSP_TZ_RE = re.compile(
    r"\b(?:vpn|tor|wireguard|openvpn|proxy|tunnel|teredo|6to4|isatap|ipsec|sock[s]?5|headless|selenium|webdriver|bot)\b",
    re.IGNORECASE
)

TZ_ABBR_OFFSETS_MIN = {
    "UTC": 0, "GMT": 0,
    "IST": 330, "PKT": 300, "BDT": 480, "MYT": 480, "SGT": 480, "HKT": 480,
    "JST": 540, "KST": 540, "ICT": 420,
    "CET": 60, "CEST": 120, "EET": 120, "EEST": 180,
    "BST": 60, "MSK": 180,
    "PST": -480, "PDT": -420,
    "MST": -420, "MDT": -360,
    "CST": -360, "CDT": -300,
    "EST": -300, "EDT": -240,
    "AEST": 600, "AEDT": 660, "ACST": 570, "AWST": 480,
    "NZST": 720, "NZDT": 780,
}
TZ_ABBR_RE = re.compile(r"\b(" + "|".join(sorted(TZ_ABBR_OFFSETS_MIN.keys(), key=len, reverse=True)) + r")\b")

# ============================================================
# IPV6 UNMASKING (TUNNELING)
# Supports: IPv4-mapped, 6to4, Teredo, ISATAP, NAT64(64:ff9b::/96)
# ============================================================
_NAT64_WKP = ipaddress.ip_network("64:ff9b::/96")

@lru_cache(maxsize=200000)
def _unmask_one_ip(ip_str: str):
    ip_str = (ip_str or "").strip().strip("[]")
    if not ip_str:
        return ("", 0, 0, 0, 0, 0, 0)  # v4, mapped, 6to4, teredo, isatap, nat64, is_ipv6
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except Exception:
        return ("", 0, 0, 0, 0, 0, 0)

    if isinstance(ip_obj, ipaddress.IPv4Address):
        return ("", 0, 0, 0, 0, 0, 0)

    packed = ip_obj.packed
    is_ipv6 = 1

    # IPv4-mapped ::ffff:a.b.c.d
    try:
        v4m = getattr(ip_obj, "ipv4_mapped", None)
        if v4m is not None:
            return (str(v4m), 1, 0, 0, 0, 0, is_ipv6)
    except Exception:
        pass

    # 6to4 2002:w.x.y.z::/48
    if packed[0:2] == b"\x20\x02":
        v4 = ipaddress.IPv4Address(packed[2:6])
        return (str(v4), 0, 1, 0, 0, 0, is_ipv6)

    # Teredo 2001:0000::/32 (client IPv4 is inverted)
    if packed[0:4] == b"\x20\x01\x00\x00":
        inv = bytes((~b) & 0xFF for b in packed[12:16])
        try:
            v4 = ipaddress.IPv4Address(inv)
            return (str(v4), 0, 0, 1, 0, 0, is_ipv6)
        except Exception:
            return ("", 0, 0, 1, 0, 0, is_ipv6)

    # ISATAP ...:0:5efe:w.x.y.z
    if packed[8:10] == b"\x00\x00" and packed[10:12] == b"\x5e\xfe":
        try:
            v4 = ipaddress.IPv4Address(packed[12:16])
            return (str(v4), 0, 0, 0, 1, 0, is_ipv6)
        except Exception:
            return ("", 0, 0, 0, 1, 0, is_ipv6)

    # NAT64 well-known prefix 64:ff9b::/96
    try:
        if ip_obj in _NAT64_WKP:
            v4 = ipaddress.IPv4Address(packed[12:16])
            return (str(v4), 0, 0, 0, 0, 1, is_ipv6)
    except Exception:
        pass

    return ("", 0, 0, 0, 0, 0, is_ipv6)

def _unmask_ipv6_series(ip_s: pd.Series) -> pd.DataFrame:
    s = ip_s.fillna("").astype(str).str.strip()
    codes, uniq = pd.factorize(s, sort=False)
    out = [ _unmask_one_ip(u) for u in uniq ]

    v4_u    = np.array([x[0] for x in out], dtype=object)
    mapped  = np.array([x[1] for x in out], dtype=np.int8)
    to4     = np.array([x[2] for x in out], dtype=np.int8)
    ter     = np.array([x[3] for x in out], dtype=np.int8)
    isa     = np.array([x[4] for x in out], dtype=np.int8)
    nat64   = np.array([x[5] for x in out], dtype=np.int8)
    is6     = np.array([x[6] for x in out], dtype=np.int8)

    return pd.DataFrame({
        "unmasked_v4": v4_u[codes],
        "ipv6_is_mapped": mapped[codes],
        "ipv6_is_6to4": to4[codes],
        "ipv6_is_teredo": ter[codes],
        "ipv6_is_isatap": isa[codes],
        "ipv6_is_nat64": nat64[codes],
        "ipv6_present": is6[codes],
    }, index=ip_s.index)

# ============================================================
# WHITELIST DOMAINS (TRAINING FEATURE + ANTI-SPOOF)
# ============================================================
@lru_cache(maxsize=8)
def _load_whitelist_set_from_csv(csv_path: str):
    try:
        if not csv_path or (not os.path.exists(csv_path)):
            return set()
        dfw = pd.read_csv(csv_path)
        if dfw.shape[1] == 0:
            return set()
        col = None
        for c in dfw.columns:
            if str(c).strip().lower() in ("domain","domains","host","hostname"):
                col = c
                break
        if col is None:
            col = dfw.columns[0]
        vals = dfw[col].dropna().astype(str).str.strip().str.lower().tolist()
        vals = [v.replace("http://","").replace("https://","").split("/")[0] for v in vals]
        vals = [v for v in vals if v and v not in ("nan","none","null","-","--","notprovided")]
        return set(vals)
    except Exception:
        return set()

def _whitelist_hit_series(dom: pd.Series, whitelist_set: set) -> pd.Series:
    d = dom.fillna("").astype(str).str.lower().str.strip()
    if not whitelist_set:
        return pd.Series(np.zeros(len(d), dtype=np.int8), index=d.index)

    codes, uniq = pd.factorize(d, sort=False)

    def hit_one(x: str) -> int:
        if not x:
            return 0
        if x in whitelist_set:
            return 1
        parts = x.split(".")
        for i in range(1, len(parts)):
            suf = ".".join(parts[i:])
            if suf in whitelist_set:
                return 1
        return 0

    hit_u = np.fromiter((hit_one(u) for u in uniq), dtype=np.int8, count=len(uniq))
    return pd.Series(hit_u[codes], index=d.index, dtype=np.int8)

# ============================================================
# GEO HELPERS (your original)
# ============================================================
def _is_private_or_special_ipv4_series(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip()
    return (
        x.str.startswith("10.") |
        x.str.startswith("127.") |
        x.str.startswith("192.168.") |
        x.str.startswith("169.254.") |
        x.str.match(r"^172\.(1[6-9]|2\d|3[0-1])\.", na=False) |
        x.str.match(r"^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.", na=False) |
        x.eq("")
    )

def _ipv4_to_int_series(ip_s: pd.Series) -> pd.Series:
    s = ip_s.fillna("").astype(str).str.strip()
    parts = s.str.split(".", n=3, expand=True)
    if parts.shape[1] != 4:
        return pd.Series(np.nan, index=s.index, dtype="float64")

    a = pd.to_numeric(parts[0], errors="coerce")
    b = pd.to_numeric(parts[1], errors="coerce")
    c = pd.to_numeric(parts[2], errors="coerce")
    d = pd.to_numeric(parts[3], errors="coerce")

    valid = (
        a.notna() & b.notna() & c.notna() & d.notna() &
        a.between(0,255) & b.between(0,255) & c.between(0,255) & d.between(0,255)
    )

    out = pd.Series(np.nan, index=s.index, dtype="float64")
    aa = a[valid].astype(np.uint32)
    bb = b[valid].astype(np.uint32)
    cc = c[valid].astype(np.uint32)
    dd = d[valid].astype(np.uint32)
    out.loc[valid] = ((aa<<24) + (bb<<16) + (cc<<8) + dd).astype(np.uint32)
    return out

def _map_ipv4_series_to_country(ip_s: pd.Series, starts, ends, ccodes) -> pd.Series:
    ip_int = _ipv4_to_int_series(ip_s)
    valid = ip_int.notna()

    out = np.full(len(ip_s), "??", dtype=object)
    if valid.any():
        ip_vals = ip_int.loc[valid].astype("uint32").to_numpy()
        idxs = np.searchsorted(starts, ip_vals, side="right") - 1
        idxs = np.clip(idxs, 0, len(starts) - 1)
        in_range = (ip_vals >= starts[idxs]) & (ip_vals <= ends[idxs])
        mapped = np.where(in_range, ccodes[idxs], "??").astype(object)
        out[np.flatnonzero(valid.to_numpy())] = mapped

    return pd.Series(out, index=ip_s.index, dtype="object")

def _ccTLD_country_from_domain(dom: pd.Series) -> pd.Series:
    d = dom.fillna("").astype(str).str.lower().str.strip()
    tld = d.str.rsplit(".", n=1).str[-1]
    cc = tld.where(tld.str.len().eq(2) & tld.str.isalpha(), "")
    return cc.str.upper().replace("", "??")

# =========================
# CACHES (your original)
# =========================
@lru_cache(maxsize=2)
def _resolve_ip_geo_csv_path(explicit: str | None) -> str | None:
    if explicit and os.path.exists(explicit):
        return explicit

    g = globals().get("ip_geo_csv_path", None)
    if isinstance(g, str) and os.path.exists(g):
        return g

    for cand in [globals().get("ip_geo_csv_path", None), os.getenv("IP_GEO_CSV", ""), os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ip-geo-country-ipv4.csv"), "/kaggle/input/defaults/ip-geo-country-ipv4.csv"]:
        if cand and os.path.exists(cand):
            return cand

    hits = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ip-geo-country-ipv4*.csv")) + glob.glob("/kaggle/input/**/ip-geo-country-ipv4*.csv", recursive=True)
    if hits:
        p = sorted(hits)[0]
        if os.path.exists(p):
            return p
    return None

@lru_cache(maxsize=4)
def _load_ip_geo_arrays(csv_path: str):
    ip_geo = pd.read_csv(csv_path, header=None, names=["ip_start", "ip_end", "country"])
    ip_geo["ip_start"] = pd.to_numeric(ip_geo["ip_start"], errors="coerce")
    ip_geo["ip_end"]   = pd.to_numeric(ip_geo["ip_end"], errors="coerce")
    ip_geo = ip_geo.dropna(subset=["ip_start", "ip_end"]).copy()
    ip_geo["ip_start"] = ip_geo["ip_start"].astype("uint32")
    ip_geo["ip_end"]   = ip_geo["ip_end"].astype("uint32")
    ip_geo["country"]  = ip_geo["country"].astype(str)
    ip_geo.sort_values(by="ip_start", inplace=True, ignore_index=True)
    return ip_geo["ip_start"].to_numpy(), ip_geo["ip_end"].to_numpy(), ip_geo["country"].to_numpy()

@lru_cache(maxsize=4)
def _load_global_ip_bytes_stats_map(path: str):
    g = pd.read_csv(path)
    cols = set(g.columns)

    if {"client_ip", "mean_ip", "std_ip"}.issubset(cols):
        tmp = g[["client_ip", "mean_ip", "std_ip"]].copy()
    elif {"client_ip", "mean", "std"}.issubset(cols):
        tmp = g[["client_ip", "mean", "std"]].rename(columns={"mean":"mean_ip","std":"std_ip"}).copy()
    else:
        return None

    tmp["client_ip"] = tmp["client_ip"].astype(str)
    tmp["mean_ip"] = pd.to_numeric(tmp["mean_ip"], errors="coerce")
    tmp["std_ip"]  = pd.to_numeric(tmp["std_ip"], errors="coerce").replace(0, np.nan)
    tmp = tmp.dropna(subset=["client_ip"]).drop_duplicates("client_ip", keep="last")
    return tmp.set_index("client_ip")[["mean_ip","std_ip"]]

@lru_cache(maxsize=4)
def _load_global_ua_freq(path: str):
    return joblib.load(path, mmap_mode="r")

# =========================
# URL / ENTROPY HELPERS (your original)
# =========================
def _shannon_entropy_fast(s: str) -> float:
    if not s:
        return 0.0
    b = s.encode("utf-8", errors="ignore")
    if not b:
        return 0.0
    arr = np.frombuffer(b, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(np.float32)
    counts = counts[counts > 0]
    p = counts / float(arr.size)
    return float(-(p * np.log2(p)).sum())

def _double_unquote_lower(s: str, max_len: int = 4096) -> str:
    try:
        s2 = urllib.parse.unquote(urllib.parse.unquote(s))
    except Exception:
        s2 = s
    s2 = (s2 or "").lower()
    if max_len and len(s2) > max_len:
        s2 = s2[:max_len]
    return s2

# ============================================================
# PRECOMPILED REGEX / CONSTANTS (your original)
# ============================================================
SUSPICIOUS_UA_RE = re.compile(
    r"(?:HeadlessChrome|ChromeHeadless|PhantomJS|GhostDriver|SlimerJS|NightmareJS|ZombieJS|CasperJS|"
    r"selenium|webdriver|automation|bot|spider|crawler|scraper|fetcher|scanner|attack|exploit|headless|"
    r"auto|mass|bulk|test|spam|curl|wget|powershell(?:\.exe)?|microsoft\s*bits|libwww-perl|python-requests|"
    r"httrack|sqlmap|nikto|nmap|go-http-client|cmd\.exe|pwsh(?:\.exe)?|mshta\.exe|rundll32\.exe|regsvr32\.exe|"
    r"wscript\.exe|cscript\.exe|wmic\.exe|certutil\.exe|bitsadmin\.exe|schtasks\.exe|ftp\.exe|psexec\.exe|"
    r"msbuild\.exe|installutil\.exe|reg\.exe|tasklist\.exe|taskkill\.exe|net(?:sh)?\.exe|findstr\.exe|"
    r"attrib\.exe|esentutl\.exe|cmstp\.exe|msiexec\.exe|hh\.exe|odbcconf\.exe|scriptrunner\.exe|"
    r"infdefaultinstall\.exe|syncappvpublishingserver\.exe|mavinject\.exe|mofcomp\.exe|forfiles\.exe|"
    r"xwizard\.exe)",
    re.IGNORECASE
)

SUSPICIOUS_UA_RE2 = re.compile(
    r"(?:sqlmap|nikto|wget|curl|bot|crawler|spider|scanner|exploit|attack|python-requests|java-http-client|^\s*$|null|^-+$)",
    re.IGNORECASE
)

_CLOUD_SERVICES = [
    "dropbox.com","drive.google.com","docs.google.com","onedrive.live.com","sharepoint.com","box.com","icloud.com",
    "mega.nz","mediafire.com","wetransfer.com","pcloud.com","amazonaws.com","sendspace.com","file.io","anonfiles.com",
    "zippyshare.com","transfernow.net","filemail.com","transferxl.com"
]
CLOUD_RE = re.compile("|".join(map(re.escape, _CLOUD_SERVICES)), re.IGNORECASE)

RE_CRIT = re.compile(
    r"(?:union\s+select|xp_cmdshell|gopher|net\s+user.*\/add|"
    r"mimikatz|sekurlsa|procdump|ntdsutil|"
    r"\b(?:powershell|pwsh)\b.*(?:\s|-)enc\b|"
    r"downloadstring|iex\s+|invoke-expression)",
    re.IGNORECASE
)
RE_HIGH = re.compile(r"(?:<script>|javascript:|onerror=|onload=|\b(?:rundll32|regsvr32|mshta|certutil|bitsadmin)\b)", re.IGNORECASE)
RE_MED  = re.compile(r"(?:\.\./|\b(?:cmd\.exe|/bin/sh|/bin/bash)\b)", re.IGNORECASE)
RE_LOW  = re.compile(r"(?:base64)", re.IGNORECASE)
RE_VLOW = re.compile(r"(?:\.(?:chm|hta|vbs|ps1)(?:\b|$))", re.IGNORECASE)

CRIT_TOOLS_PAT = r"(?:mimikatz|sekurlsa|procdump|vaultcmd|klist|tgticket|ntdsutil|dsquery|dsget|csvde|ldifde|nltest|psexec|wmic|winrm|invoke-command|enter-pssession|vssadmin|wbadmin|bcdedit|wevtutil|cipher)"
HIGH_TOOLS_PAT = r"(?:powershell|pwsh|cmd\.exe|bash|sh|zsh|cscript|wscript|net\.exe|net1\.exe|sc\.exe|reg\.exe|schtasks|at\.exe|runas|rundll32|regsvr32|mshta|installutil|msbuild|msxsl|certutil|bitsadmin|hh\.exe|cmstp)"
RECON_TOOLS_PAT= r"(?:whoami|quser|qwinsta|hostname|ipconfig|ifconfig|arp|route|netstat|nslookup|ping|tracert|tasklist|taskkill|systeminfo|driverquery|fsutil|dir|tree|type|findstr)"

# ============================================================
# TIMESTAMP PARSER (wall-time first, cached per unique timestamp)
# ============================================================
_EPOCH_RE = re.compile(r"(?:audit\()?\s*(\d{9,19})(?:\.\d+)?(?:\))?")

@lru_cache(maxsize=1)
def _dateutil_tzinfos():
    try:
        from dateutil.tz import tzoffset  # type: ignore
        return {k: tzoffset(k, int(v) * 60) for k, v in TZ_ABBR_OFFSETS_MIN.items()}
    except Exception:
        return {}

@lru_cache(maxsize=200000)
def _parse_one_ts_wall(ts: str):
    """
    Returns (wall_dt_naive, offset_min, has_tz, ok)
    """
    s = (ts or "").strip().strip("[](){}<>")
    if not s:
        return (None, 0, 0, 0)

    m = _EPOCH_RE.search(s)
    if m:
        ds = m.group(1)
        if ds.isdigit():
            try:
                n = int(ds)
                if len(ds) == 10:  # seconds
                    dt = datetime.fromtimestamp(n, tz=timezone.utc)
                elif len(ds) == 13:  # ms
                    dt = datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
                elif len(ds) == 16:  # us
                    dt = datetime.fromtimestamp(n / 1_000_000.0, tz=timezone.utc)
                elif len(ds) == 19:  # ns
                    dt = datetime.fromtimestamp(n / 1_000_000_000.0, tz=timezone.utc)
                else:
                    dt = None
                if dt is not None:
                    return (dt.replace(tzinfo=None), 0, 1, 1)
            except Exception:
                pass

    # dateutil parse
    try:
        from dateutil import parser as dtparser  # type: ignore
        default_dt = datetime(2026, 1, 1, 0, 0, 0)
        dt = dtparser.parse(s, tzinfos=_dateutil_tzinfos(), fuzzy=True, default=default_dt)
        if getattr(dt, "tzinfo", None) is not None and dt.utcoffset() is not None:
            off = int(dt.utcoffset().total_seconds() // 60)
            return (dt.replace(tzinfo=None), off, 1, 1)
        return (dt.replace(tzinfo=None), 0, 0, 1)
    except Exception:
        pass

    # pandas fallback
    try:
        dtp = pd.to_datetime(s, errors="coerce")
        if pd.notna(dtp):
            return (dtp.to_pydatetime(), 0, 0, 1)
    except Exception:
        pass

    return (None, 0, 0, 0)



def _to_naive_wall_timestamp(x):
    try:
        if x is None or pd.isna(x):
            return pd.NaT
        t = pd.Timestamp(x)
        if pd.isna(t):
            return pd.NaT
        if getattr(t, "tzinfo", None) is not None:
            return t.tz_localize(None)
        return t
    except Exception:
        return pd.NaT

def _parse_wall_datetime_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      ts_used  : datetime64[ns] (wall-time unless suspicious => utc-time)
      off_min  : int32 minutes (best-effort)
      suspicious: bool
    """
    raw_line = df.get("raw", df.get("raw_log", pd.Series("", index=df.index))).fillna("").astype(str)
    ua = df.get("user_agent", pd.Series("", index=df.index)).fillna("").astype(str)
    cmd = df.get("command", pd.Series("", index=df.index)).fillna("").astype(str)

    # Prefer the app's conservative UTC-forcing decision when present.  The
    # older feature cell treated a bare "proxy" token as suspicious, which
    # shifted all hour/weekend features to UTC and silently disagreed with the
    # app's odd-hours gate.  `tz_force_utc` is the single source of truth now.
    if "tz_force_utc" in df.columns:
        suspicious = pd.to_numeric(df["tz_force_utc"], errors="coerce").fillna(0).astype(int).astype(bool)
    else:
        suspicious = (
            raw_line.str.contains(SUSP_TZ_RE, na=False) |
            ua.str.contains(SUSP_TZ_RE, na=False) |
            cmd.str.contains(SUSP_TZ_RE, na=False)
        )

    def _fill_wall_nat(wall: pd.Series) -> pd.Series:
        # Convert any tz-aware values to naive wall-time without shifting clock
        # time.  This preserves local-hour semantics and prevents object-dtype
        # Series from breaking `.dt` operations.
        wall = pd.Series(wall, index=df.index).map(_to_naive_wall_timestamp)
        wall = pd.to_datetime(wall, errors="coerce")
        miss = wall.isna()
        if miss.any():
            seed = df.get("timestamp", pd.Series("", index=df.index)).fillna("").astype(str)
            if "raw_log" in df.columns:
                seed = seed + "|" + df["raw_log"].fillna("").astype(str)
            h = pd.util.hash_pandas_object(seed + "|tsfill", index=False).astype(np.uint64)
            sec = (h % (6 * 365 * 24 * 3600)).astype(np.int64)
            base = pd.Timestamp("2020-01-01 00:00:00")
            wall.loc[miss] = base + pd.to_timedelta(sec.loc[miss].to_numpy(), unit="s")
        return wall

    # prefer precomputed wall_dt / timestamp_local_str if present (from your app)
    if "wall_dt" in df.columns:
        wall = _fill_wall_nat(pd.to_datetime(df["wall_dt"], errors="coerce"))
        off = pd.to_numeric(df.get("tz_offset_min", 0), errors="coerce").fillna(0).astype(np.int32)
        ts_used = wall.copy()
        m = suspicious & wall.notna()
        if m.any():
            ts_used.loc[m] = wall.loc[m] - pd.to_timedelta(off.loc[m].to_numpy(), unit="m")
        return ts_used, off, suspicious

    if "timestamp_local_str" in df.columns:
        wall = _fill_wall_nat(pd.to_datetime(df["timestamp_local_str"], errors="coerce"))
        off = pd.to_numeric(df.get("tz_offset_min", 0), errors="coerce").fillna(0).astype(np.int32)
        ts_used = wall.copy()
        m = suspicious & wall.notna()
        if m.any():
            ts_used.loc[m] = wall.loc[m] - pd.to_timedelta(off.loc[m].to_numpy(), unit="m")
        return ts_used, off, suspicious

    # parse from timestamp strings (cached)
    ts_raw = df.get("timestamp", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    codes, uniq = pd.factorize(ts_raw, sort=False)
    parsed = [_parse_one_ts_wall(u) for u in uniq]

    wall_u = np.array([p[0] for p in parsed], dtype=object)
    off_u  = np.array([p[1] for p in parsed], dtype=np.int32)
    has_tz = np.array([p[2] for p in parsed], dtype=np.int8)
    ok_u   = np.array([p[3] for p in parsed], dtype=np.int8)

    wall = pd.to_datetime(pd.Series(wall_u[codes], index=df.index), errors="coerce")
    off = pd.Series(off_u[codes], index=df.index, dtype=np.int32)

    # abbr fallback only when no tz found in string
    ab = raw_line.str.extract(TZ_ABBR_RE, expand=False)
    ab_ok = ab.notna()
    if ab_ok.any():
        mapped = ab.loc[ab_ok].map(lambda x: TZ_ABBR_OFFSETS_MIN.get(str(x), 0)).astype(np.int32)
        no_tz = pd.Series(has_tz[codes] == 0, index=df.index)
        use = ab_ok & no_tz & off.eq(0)
        if use.any():
            off.loc[use] = mapped.loc[use].to_numpy()

    # deterministic fill for wall NaT
    miss = wall.isna()
    if miss.any():
        h = pd.util.hash_pandas_object(ts_raw.fillna("").astype(str) + "|tsfill", index=False).astype(np.uint64)
        sec = (h % (6 * 365 * 24 * 3600)).astype(np.int64)
        base = pd.Timestamp("2020-01-01 00:00:00")
        wall.loc[miss] = base + pd.to_timedelta(sec.loc[miss].to_numpy(), unit="s")

    # suspicious => UTC time for features
    ts_used = wall.copy()
    m = suspicious & wall.notna()
    if m.any():
        ts_used.loc[m] = wall.loc[m] - pd.to_timedelta(off.loc[m].to_numpy(), unit="m")

    return ts_used, off, suspicious

# ============================================================
# ✅ MAIN FEATURE ENGINEERING FUNCTION (UPDATED)
# ============================================================
def our_custom_feature_engineering_function(
    X,
    debug: bool = False,
    tz: str | None = "auto",                 # kept for compatibility; "auto" = DO NOT force tz conversion
    ip_geo_csv: str | None = None,
    ua_freq_path: str | None = None,
    gstats_path: str | None = None,
    fit_scaler_if_needed: bool = True,
    whitelist_csv: str | None = None,        # NEW
    whitelist_domains: set | None = None,    # NEW
):
    # ---- Resolve optional local artifact paths ----
    if not ua_freq_path:
        ua_freq_path = _pick_first_existing_path([
            os.getenv("UA_FREQ_PATH", ""),
            os.path.join(_THIS_DIR, "data", "global_ua_freq.pkl"),
            os.path.join(_THIS_DIR, "data", "ua_freq.pkl"),
        ])
    if not gstats_path:
        gstats_path = _pick_first_existing_path([
            os.getenv("GLOBAL_IP_STATS_CSV", ""),
            os.path.join(_THIS_DIR, "data", "global_ip_bytes_stats.csv"),
            os.path.join(_THIS_DIR, "data", "ip_bytes_stats.csv"),
        ])
    # ---- Ensure DataFrame ----
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    df = X.copy()

    if "timestamp" not in df.columns:
        raise KeyError("Missing 'timestamp' column after parsing logs!")
    if "client_ip" not in df.columns:
        raise KeyError("Missing 'client_ip' column after parsing logs!")

    # Ensure required columns exist
    defaults = {
        "user_agent": "", "domain": "", "full_url": "", "url_path": "",
        "method": "", "status": np.nan, "bytes_out": 0, "bytes_in": 0,
        "username": "", "workstation": "", "process": "", "command": "",
        "dest_ip": "", "log_type": "", "raw": "", "raw_log": "",
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    # ensure raw exists
    if df["raw"].isna().all() and ("raw_log" in df.columns):
        df["raw"] = df["raw_log"].fillna("").astype(str)

    # ============================================================
    # TIME FEATURES (AUTO, wall-time odd-hours; suspicious => UTC)
    # ============================================================
    ts, tz_off_min, suspicious_tz = _parse_wall_datetime_series(df)

    # optional explicit tz override (only if user demands; otherwise keep auto)
    if tz and isinstance(tz, str) and tz.lower() not in ("auto", "none", ""):
        try:
            # interpret ts as UTC then convert to tz (explicit user override)
            ts_utc = pd.to_datetime(ts, errors="coerce", utc=True)
            ts = ts_utc.dt.tz_convert(tz).dt.tz_localize(None)
        except Exception:
            pass

    df["timestamp"] = ts
    df["tz_offset_min"] = tz_off_min.astype(np.int32)
    df["timestamp_suspicious_tz"] = suspicious_tz.astype(np.int8)

    df["hour"] = ts.dt.hour.fillna(0).astype(np.int8)
    df["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(np.int8)

    hf = df["hour"].astype(np.float32) + ts.dt.minute.fillna(0).astype(np.float32)/60.0
    df["odd_hours"] = ((hf >= 23.0) | (hf < 5.5)).astype(np.int8)

    hour_bucket = ts.dt.floor("h")
    min_bucket = ts.dt.floor("min")
    tenmin_bucket = ts.dt.floor("10min")

    # Normalize numerics
    df["bytes_out"] = pd.to_numeric(df["bytes_out"], errors="coerce").fillna(0).clip(lower=0).astype(np.int64)
    df["bytes_in"]  = pd.to_numeric(df["bytes_in"], errors="coerce").fillna(0).clip(lower=0).astype(np.int64)
    df["status"]    = pd.to_numeric(df["status"], errors="coerce")

    # Common series
    cip = df["client_ip"].fillna("").astype(str).str.strip()
    dip = df["dest_ip"].fillna("").astype(str).str.strip()
    ua  = df["user_agent"].fillna("").astype(str)
    dom = df["domain"].fillna("").astype(str)
    full_url = df["full_url"].fillna("").astype(str)
    url_path = df["url_path"].fillna("").astype(str)
    method = df["method"].fillna("").astype(str).str.upper()

    # ============================================================
    # IPv6 tunneling unmask + tunnel flags (NEW)
    # ============================================================
    c_un = _unmask_ipv6_series(cip)
    d_un = _unmask_ipv6_series(dip)

    df["ipv6_present_any"] = ((c_un["ipv6_present"] > 0) | (d_un["ipv6_present"] > 0)).astype(np.int8)
    df["ipv6_mapped_any"]  = ((c_un["ipv6_is_mapped"] > 0) | (d_un["ipv6_is_mapped"] > 0)).astype(np.int8)
    df["ipv6_6to4_any"]    = ((c_un["ipv6_is_6to4"] > 0) | (d_un["ipv6_is_6to4"] > 0)).astype(np.int8)
    df["ipv6_teredo_any"]  = ((c_un["ipv6_is_teredo"] > 0) | (d_un["ipv6_is_teredo"] > 0)).astype(np.int8)
    df["ipv6_isatap_any"]  = ((c_un["ipv6_is_isatap"] > 0) | (d_un["ipv6_is_isatap"] > 0)).astype(np.int8)
    df["ipv6_nat64_any"]   = ((c_un["ipv6_is_nat64"] > 0) | (d_un["ipv6_is_nat64"] > 0)).astype(np.int8)

    df["ipv6_tunnel_any"] = (
        (df["ipv6_mapped_any"] > 0) |
        (df["ipv6_6to4_any"] > 0) |
        (df["ipv6_teredo_any"] > 0) |
        (df["ipv6_isatap_any"] > 0) |
        (df["ipv6_nat64_any"] > 0)
    ).astype(np.int8)

    # effective IPv4 for ip_rep/geo (prefer unmasked)
    cip_eff = cip.copy()
    m1 = c_un["unmasked_v4"].fillna("").astype(str).str.len().gt(0)
    if m1.any():
        cip_eff.loc[m1] = c_un.loc[m1, "unmasked_v4"].astype(str).values

    dip_eff = dip.copy()
    m2 = d_un["unmasked_v4"].fillna("").astype(str).str.len().gt(0)
    if m2.any():
        dip_eff.loc[m2] = d_un.loc[m2, "unmasked_v4"].astype(str).values

    # ============================================================
    # Feature 1: ip_bad_rep (UPDATED: client OR dest; uses unmasked v4)
    # ============================================================
    bad_ips_local = globals().get("bad_ips", set())
    try:
        bad_ips_local = set(bad_ips_local)
    except Exception:
        bad_ips_local = set()
    df["ip_bad_rep"] = (cip_eff.isin(bad_ips_local) | dip_eff.isin(bad_ips_local)).astype(np.int8)

    # ============================================================
    # Feature 2: suspicious_geo ✅ (UPDATED: uses effective IPs)
    # ============================================================
    suspicious_geo_codes = globals().get("suspicious_geo_country_codes", [])
    susp_geo_set = set(map(str, suspicious_geo_codes)) if suspicious_geo_codes is not None else set()

    df["country"] = "??"
    df["suspicious_geo"] = np.int8(0)

    ip_geo_csv = _resolve_ip_geo_csv_path(ip_geo_csv)

    cip_s = cip_eff.fillna("").astype(str).str.strip()
    dip_s = dip_eff.fillna("").astype(str).str.strip()

    cip_priv = _is_private_or_special_ipv4_series(cip_s)
    dip_priv = _is_private_or_special_ipv4_series(dip_s)

    geo_ip = cip_s.copy()
    use_dest = cip_priv & (~dip_priv)
    if use_dest.any():
        geo_ip.loc[use_dest] = dip_s.loc[use_dest].values

    geo_is_internal = _is_private_or_special_ipv4_series(geo_ip)

    country = pd.Series(
        np.where(geo_is_internal.to_numpy(), "INTERNAL", "??"),
        index=df.index, dtype="object"
    )

    if ip_geo_csv is not None and os.path.exists(ip_geo_csv):
        try:
            starts, ends, ccodes = _load_ip_geo_arrays(ip_geo_csv)
            mapped = _map_ipv4_series_to_country(geo_ip.where(~geo_is_internal, ""), starts, ends, ccodes)
            country = country.where(country.eq("INTERNAL"), mapped.where(mapped.ne("??"), country))
        except Exception:
            pass

    cc = _ccTLD_country_from_domain(dom)
    use_cc = country.eq("??") & cc.ne("??")
    if use_cc.any():
        country.loc[use_cc] = cc.loc[use_cc].values

    public_unknown = country.eq("??") & (~geo_is_internal) & geo_ip.ne("")
    if public_unknown.any():
        country.loc[public_unknown] = "PUBLIC_UNKNOWN"

    df["country"] = country.values

    DEFAULT_SUSP = {"RU", "CN", "KP", "IR", "SY"}
    base_susp = country.isin(susp_geo_set) if susp_geo_set else country.isin(DEFAULT_SUSP)
    unknown_susp = (country.eq("PUBLIC_UNKNOWN") & (df["ip_bad_rep"].astype(int) == 1))
    df["suspicious_geo"] = (base_susp | unknown_susp).astype(np.int8)

    # ============================================================
    # Feature 3: malicious_lolbin_ua + ua_freq_enc
    # ============================================================
    ua_is_mal = ua.str.contains(SUSPICIOUS_UA_RE, na=False).astype(np.int8)

    if ua_freq_path and os.path.exists(ua_freq_path):
        try:
            ua_freq_map = _load_global_ua_freq(ua_freq_path)
            df["ua_freq_enc"] = ua.map(ua_freq_map).fillna(0).astype(np.float32)
        except Exception:
            df["ua_freq_enc"] = np.float32(0.0)
    else:
        df["ua_freq_enc"] = np.float32(0.0)

    ua_cat = np.zeros(len(df), dtype=np.uint8)
    ua_cat[df["ua_freq_enc"].to_numpy() < 0.0005] = 1
    ua_cat[ua_is_mal.to_numpy().astype(bool)] = 2
    df["malicious_lolbin_ua"] = ua_cat.astype(np.uint8)

    # ============================================================
    # Feature 5: high_bytes_out (zscore if global stats exist else threshold)
    # ============================================================
    high_bytes_thresh = 250 * 1024 * 1024
    df["high_bytes_out"] = (df["bytes_out"] > high_bytes_thresh).astype(np.int8)

    if gstats_path and os.path.exists(gstats_path):
        try:
            mstats = _load_global_ip_bytes_stats_map(gstats_path)
            if mstats is not None:
                stats = mstats.reindex(cip_eff.astype(str))
                mean_ip = stats["mean_ip"].to_numpy()
                std_ip  = stats["std_ip"].to_numpy()
                z = (df["bytes_out"].to_numpy(dtype=np.float64) - mean_ip) / std_ip
                df["high_bytes_out"] = (np.nan_to_num(z, nan=-999.0) > 1.5).astype(np.int8)
        except Exception:
            pass

    # ============================================================
    # Domain rarity + domain_freq
    # ============================================================
    domain_freq_norm = dom.value_counts(normalize=True)
    df["domain_freq"] = dom.map(domain_freq_norm).fillna(0).astype(np.float32)
    df["domain_cat_Rare Domain"] = (df["domain_freq"] < 0.0005).astype(np.int8)

    # ============================================================
    # combined_rare_suspicious_ua
    # ============================================================
    ua_counts = ua.value_counts(dropna=False)
    df["rare_ua"] = ua.map(ua_counts).fillna(0).lt(3).astype(np.int8)
    df["explicit_suspicious_ua"] = ua.str.contains(SUSPICIOUS_UA_RE2, na=True).astype(np.int8)
    df["combined_rare_suspicious_ua"] = (df["rare_ua"] | df["explicit_suspicious_ua"]).astype(np.int8)

    # ============================================================
    # first_cloud_use
    # ============================================================
    cloud_upload = full_url.str.contains(CLOUD_RE, na=False)
    df["first_cloud_use"] = ((cloud_upload.groupby(cip_eff, sort=False).cumsum() == 1) & cloud_upload).astype(np.int8)

    # ============================================================
    # suspicious_url (entropy)
    # ============================================================
    df["url_length"] = full_url.str.len().fillna(0).astype(np.int32)
    codes_u, uniques_u = pd.factorize(full_url, sort=False)
    ent = np.fromiter((_shannon_entropy_fast(u) for u in uniques_u), dtype=np.float32, count=len(uniques_u))
    df["url_entropy"] = ent[codes_u].astype(np.float32)
    df["suspicious_url"] = ((df["url_length"] > 100) | (df["url_entropy"] > 4.5)).astype(np.int8)

    # ============================================================
    # weekend / peak / bins
    # ============================================================
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)
    df["peak_hour"] = df["hour"].between(19, 22).astype(np.int8)

    h = df["hour"].astype(np.int16)
    df["hour_of_day_bin"] = np.select([h <= 5, h <= 11, h <= 17], [0, 1, 2], default=3).astype(np.int8)

    dow = df["day_of_week"].astype(np.int16)
    df["day_of_week_bin"] = np.select([dow <= 3, dow == 4, dow >= 5], [0, 1, 2], default=0).astype(np.int8)

    # ============================================================
    # avg_bytes_per_hour_bin
    # ============================================================
    df["avg_bytes_per_hour"] = df.groupby([cip_eff, hour_bucket], sort=False)["bytes_out"].transform("mean").fillna(0).astype(np.float32)
    try:
        df["avg_bytes_per_hour_bin"] = pd.qcut(df["avg_bytes_per_hour"], q=4, duplicates="drop").cat.codes.astype(np.int8)
    except Exception:
        df["avg_bytes_per_hour_bin"] = np.int8(0)

    # ============================================================
    # user_activity_freq_bin / ip_request_freq_bin
    # ============================================================
    ip_counts = cip_eff.value_counts()
    ip_count_per_row = cip_eff.map(ip_counts).fillna(0).astype(np.int32)

    try:
        df["user_activity_freq_bin"] = pd.qcut(ip_count_per_row, q=4, duplicates="drop").cat.codes.astype(np.int8)
    except Exception:
        df["user_activity_freq_bin"] = np.int8(0)

    try:
        df["ip_request_freq_bin"] = pd.qcut(ip_count_per_row, q=4, duplicates="drop").cat.codes.astype(np.int8)
    except Exception:
        df["ip_request_freq_bin"] = np.int8(0)

    # interactions
    df["interaction_ip_geo"] = (df["ip_bad_rep"].astype(bool) & df["suspicious_geo"].astype(bool)).astype(np.int8)
    df["interaction_ua_odd_hours"] = ((df["malicious_lolbin_ua"] > 0) & (df["odd_hours"] > 0)).astype(np.int8)

    # ============================================================
    # burst_activity
    # ============================================================
    ts_min, ts_max = ts.min(), ts.max()
    total_hours = (ts_max - ts_min).total_seconds()/3600.0 if pd.notna(ts_min) and pd.notna(ts_max) else 0.0
    total_hours = max(total_hours, 1e-6)

    df["ip_request_freq"] = (ip_count_per_row.astype(np.float32) / total_hours).astype(np.float32)

    req_per_ip_hr = df.groupby([cip_eff, hour_bucket], sort=False)["client_ip"].transform("size").astype(np.int32)
    df["requests_per_ip_hour"] = req_per_ip_hr

    mu = req_per_ip_hr.mean()
    sd = req_per_ip_hr.std()
    z = (req_per_ip_hr - mu) if (not np.isfinite(sd) or sd == 0) else (req_per_ip_hr - mu)/sd
    df["burst_activity"] = (z > 3.0).astype(np.int8)

    # rare_user_agent
    df["rare_user_agent"] = (ua.map(ua_counts).fillna(0) / max(len(df), 1) < 0.001).astype(np.int8)

    # ip_freq_enc
    df["ip_freq_enc"] = (ip_count_per_row.astype(np.float32) / max(int(ip_counts.max()), 1)).astype(np.float32)

    # rare_suspicious_activity
    df["rare_suspicious_activity"] = (df["suspicious_url"].astype(bool) | df["first_cloud_use"].astype(bool)).astype(np.int8)

    # ua_freq + ua_activity_10min
    df["ua_freq"] = (ua.map(ua_counts).fillna(0).astype(np.float32) / max(len(df), 1)).astype(np.float32)
    df["ua_activity_10min"] = df.groupby([ua, tenmin_bucket], sort=False)["user_agent"].transform("size").astype(np.int32)

    # ip_request_deviation
    df["ip_request_deviation"] = (df["requests_per_ip_hour"].astype(np.float32) - df["requests_per_ip_hour"].mean()).astype(np.float32)

    # ============================================================
    # Scaling
    # ============================================================
    scale_cols = ["ip_freq_enc", "ua_freq", "domain_freq"]
    Xs = df[scale_cols].to_numpy(dtype=np.float32, copy=False)
    try:
        if hasattr(scaler, "center_"):
            df[scale_cols] = scaler.transform(Xs)
        else:
            if fit_scaler_if_needed:
                df[scale_cols] = scaler.fit_transform(Xs)
    except Exception:
        pass

    # critical_events / interactions / spike
    df["critical_events"] = ((df["burst_activity"] > 0) | (df["high_bytes_out"] > 0)).astype(np.int8)
    df["ip_geo_malicious_interaction"] = (df["ip_bad_rep"].astype(bool) & df["suspicious_geo"].astype(bool)).astype(np.int8)
    df["odd_hour_lolbin_ua"] = ((df["odd_hours"] > 0) & (df["malicious_lolbin_ua"] > 0)).astype(np.int8)
    q99 = float(df["ip_request_deviation"].quantile(0.99)) if len(df) else 0.0
    df["extreme_ip_request_spike"] = (df["ip_request_deviation"] > q99).astype(np.int8)

    # ============================================================
    # ✅ UEBA 9 FEATURES
    # ============================================================
    bout = df["bytes_out"].astype(np.float32)
    bin_ = df["bytes_in"].astype(np.float32)
    exfil = (bout / (bin_ + 1024.0)) * np.log1p(bout)
    df["data_exfil_ratio"] = exfil.astype(np.float32)

    tl_int = np.select([exfil >= 20.0, exfil >= 5.0, exfil >= 3.0], [3, 2, 1], default=0).astype(np.int8)
    df["threat_level_int"] = tl_int
    tl_lbl = np.array(["LOW", "MEDIUM", "HIGH", "CRITICAL"], dtype=object)
    df["threat_level"] = tl_lbl[tl_int.astype(int)]

    risky_methods = {"POST", "PUT", "DELETE", "PATCH"}
    st = pd.to_numeric(df["status"], errors="coerce")
    df["critical_mod_success"] = (method.isin(risky_methods) & st.between(200, 299)).astype(np.int8)

    ws = df["workstation"].fillna("").astype(str).str.upper()
    proc_l = df["process"].fillna("").astype(str).str.lower()
    cmd_l = df["command"].fillna("").astype(str).str.lower()
    activity = (proc_l + " " + cmd_l)

    is_dc  = ws.str.contains(r"(?:^DC\b)|(?:\-DC\b)|(?:\bDC\-)", regex=True, na=False)
    is_srv = ws.str.contains(r"\bSRV\b|SERVER", regex=True, na=False)
    is_paw = ws.str.contains(r"\bADM\b|PAW", regex=True, na=False)
    asset_score = np.select([is_dc, is_srv, is_paw], [1.0, 0.9, 0.8], default=0.0).astype(np.float32)

    tool_score = np.select(
        [
            activity.str.contains(CRIT_TOOLS_PAT, regex=True, na=False),
            activity.str.contains(HIGH_TOOLS_PAT, regex=True, na=False),
            activity.str.contains(RECON_TOOLS_PAT, regex=True, na=False),
        ],
        [1.0, 0.8, 0.5],
        default=0.0
    ).astype(np.float32)

    df["username_risk_score"] = np.maximum(asset_score, tool_score).astype(np.float32)

    payload = df["command"].fillna("").astype(str)
    codes_p, uniq_p = pd.factorize(payload, sort=False)

    def _score_payload(s: str) -> float:
        if not s or len(s) < 3:
            return 0.0
        dec = _double_unquote_lower(s, max_len=4096)
        if RE_CRIT.search(dec): return 1.0
        if RE_HIGH.search(dec): return 0.9
        if RE_MED.search(dec):  return 0.8
        if RE_LOW.search(dec):  return 0.7
        if RE_VLOW.search(dec): return 0.6
        return 0.0

    scores = np.fromiter((_score_payload(u) for u in uniq_p), dtype=np.float32, count=len(uniq_p))
    df["command_risk_score"] = scores[codes_p].astype(np.float32)

    df["workstation_risk_score"] = np.select([is_dc, is_srv, is_paw], [1.0, 0.9, 0.8], default=0.2).astype(np.float32)

    # location anomaly
    if df["username"].astype(str).str.len().gt(0).any():
        user_total = df.groupby(df["username"], sort=False)["client_ip"].transform("count")
        user_ip_count = df.groupby([df["username"], cip_eff], sort=False)["client_ip"].transform("count")
        df["location_anomaly_score"] = (1.0 - (user_ip_count / user_total.replace(0, np.nan))).fillna(0.0).clip(0.0, 1.0).astype(np.float32)
    else:
        df["location_anomaly_score"] = np.float32(0.0)

    # session intensity
    df["session_intensity"] = df.groupby([cip_eff, min_bucket], sort=False)["client_ip"].transform("size").astype(np.int32)

    # behavioral consistency
    if df["username"].astype(str).str.len().gt(0).any():
        user_total_activity = df.groupby(df["username"], sort=False)["username"].transform("size").astype(np.float32)
        fp = df.groupby([df["username"], df["workstation"], cip_eff], sort=False)["username"].transform("size").astype(np.float32)
        df["behavioral_consistency"] = (fp / (user_total_activity + 5.0)).clip(0.0, 1.0).astype(np.float32)
    else:
        df["behavioral_consistency"] = np.float32(0.0)

    # session_multi_device_risk
    df["session_multi_device_risk"] = np.float32(0.0)
    if df["username"].astype(str).str.len().gt(0).any():
        tmp = df[["username", "timestamp"]].copy()
        tmp["_idx"] = df.index
        tmp["client_ip"] = cip_eff.values
        tmp.sort_values(["username", "timestamp"], inplace=True, kind="mergesort")

        prev_ip = tmp.groupby("username", sort=False)["client_ip"].shift(1)
        prev_ts = tmp.groupby("username", sort=False)["timestamp"].shift(1)
        td = (tmp["timestamp"] - prev_ts).dt.total_seconds().div(60.0).fillna(np.inf).clip(lower=0)

        ip_changed = prev_ip.notna() & (tmp["client_ip"] != prev_ip)
        risk = np.select([ip_changed & (td < 5.0), ip_changed & (td < 60.0)], [1.0, 0.5], default=0.0).astype(np.float32)
        df.loc[tmp["_idx"].to_numpy(), "session_multi_device_risk"] = risk

    # ============================================================
    # ✅ WHITELIST FEATURES (NEW)
    # ============================================================
    wl_set = set()
    if whitelist_domains is not None:
        try:
            wl_set = set([str(w).strip().lower() for w in whitelist_domains if str(w).strip()])
        except Exception:
            wl_set = set()

    if (not wl_set) and isinstance(whitelist_csv, str) and whitelist_csv and os.path.exists(whitelist_csv):
        wl_set = _load_whitelist_set_from_csv(whitelist_csv)

    df["whitelist_hit"] = _whitelist_hit_series(dom, wl_set).astype(np.int8)

    df["whitelist_suspicious_combo"] = (
        (df["whitelist_hit"] > 0) &
        (
            (df["ip_bad_rep"] > 0) |
            (df["suspicious_geo"] > 0) |
            (df["malicious_lolbin_ua"] > 0) |
            (df["odd_hours"] > 0) |
            (df["suspicious_url"] > 0) |
            (df["ipv6_tunnel_any"] > 0)
        )
    ).astype(np.int8)

    # ============================================================
    # OUTPUT (kept + added new features)
    # ============================================================
    feature_cols = [
        "ip_bad_rep","suspicious_geo","malicious_lolbin_ua","odd_hours",
        "high_bytes_out","domain_cat_Rare Domain","combined_rare_suspicious_ua",
        "first_cloud_use","suspicious_url","is_weekend","peak_hour","hour_of_day_bin",
        "day_of_week_bin","avg_bytes_per_hour_bin","user_activity_freq_bin",
        "ip_request_freq_bin","interaction_ip_geo","interaction_ua_odd_hours",
        "burst_activity","rare_user_agent","requests_per_ip_hour","ip_freq_enc",
        "rare_suspicious_activity","ua_freq","domain_freq","ua_activity_10min",
        "ip_request_deviation","critical_events","ip_geo_malicious_interaction",
        "odd_hour_lolbin_ua","extreme_ip_request_spike", 'timestamp_suspicious_tz',

        "data_exfil_ratio","threat_level","threat_level_int",
        "critical_mod_success","username_risk_score","command_risk_score",
        "workstation_risk_score","location_anomaly_score","session_intensity",
        "behavioral_consistency","session_multi_device_risk",

        # NEW (ipv6 tunneling)
        "ipv6_tunnel_any","ipv6_present_any","ipv6_mapped_any","ipv6_6to4_any","ipv6_teredo_any","ipv6_isatap_any","ipv6_nat64_any",

        # NEW (whitelist)
        "whitelist_hit","whitelist_suspicious_combo",

        # keep your original passthrough cols
        "user_agent","domain","timestamp","client_ip","method","url_path",
        "bytes_out","bytes_in","status","log_type","raw"
    ]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    # Runtime output should be numeric model/rule features only.  The original
    # notebook emitted audit/pass-through strings (threat_level, user_agent,
    # domain, raw, etc.), which later leaked into feature_columns.pkl and caused
    # schemas like 0..41 to include `threat_level` instead of the trained q-list.
    runtime_cols = list(RUNTIME_FEATURE_COLUMNS) if RUNTIME_FEATURE_COLUMNS else [
        c for c in feature_cols
        if c not in {"threat_level", "user_agent", "domain", "timestamp", "client_ip", "method", "url_path", "bytes_out", "bytes_in", "status", "log_type", "raw"}
    ]
    for c in runtime_cols:
        if c not in df.columns:
            df[c] = 0
    out = df[runtime_cols].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    if not debug:
        gc.collect()

    return out
	
	