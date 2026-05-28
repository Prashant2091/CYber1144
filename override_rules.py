from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Family-aware primary override engine.
# Goal: force only high-confidence malicious cases, while leaving normal DHCP,
# proxy, browser, firewall ALLOW, clean email/EDR, and routine OS/network logs
# as model-scored / suspicious rather than automatic malicious.

OVERRIDE_REASON_LEGEND: Dict[int, str] = {
    0: "none",
    1: "ioc_bad_ip_or_domain",
    2: "security_verdict_malicious",
    3: "exploit_or_waf_block",
    4: "lolbin_or_os_abuse",
    5: "network_c2_or_exfil",
    6: "proxy_web_scanner_or_payload",
    7: "identity_or_access_abuse",
    8: "cloud_saas_data_control_abuse",
    9: "container_k8s_secret_abuse",
    10: "email_dlp_malware_or_exfil",
    11: "ipv6_tunnel_combo",
    12: "moderate_keyword_plus_strong_signal",
}

OBS_IOC_DOMAINS: Set[str] = {
    "e-files.download",
    "hkdust.github.wiki",
    "cdn-update-check.net",
    "itamaraty-gov.com",
    "7xq2k9d1k3.biz",
}
OBS_IOC_IPS: Set[str] = {
    "185.112.83.116",
    "61.177.56.27",
    "94.190.43.52",
    "208.81.37.55",
    "134.122.188.249",
    "185.227.70.204",
    "43.131.69.98",
    "77.73.133.73",
}

BUILTIN_TRUSTED_DOMAINS: Set[str] = {
    "google.com", "github.com", "microsoft.com", "amazon.com", "aws.amazon.com",
    "stackoverflow.com", "wikipedia.org", "login.microsoftonline.com", "api.github.com",
    "office.com", "office365.com", "microsoftonline.com", "cloudflare.com",
}

# Conservative patterns: these only auto-primary when paired with family context
# or a hard signal. Generic automation UAs are NOT automatically malicious.
SCANNER_RE = re.compile(
    r"(?<![a-z0-9])(?:sqlmap|nikto|nmap|masscan|zmap|zgrab|naabu|nuclei|wpscan|acunetix|netsparker|openvas|nessus|dirbuster|dirsearch|gobuster|ffuf|feroxbuster|wfuzz|netspider|msiecrawler|femtosearchbot|zmeu|morfeus|fyrebot|pavuk|httrack|winhttrack)(?![a-z0-9])",
    re.I,
)
AUTOMATION_UA_RE = re.compile(
    r"(?<![a-z0-9])(?:python-requests|go-http-client|aws-sdk-go|aiohttp|okhttp|java/|curl/?|wget/?|axios|postmanruntime|selenium|headlesschrome|phantomjs|puppeteer|playwright)(?![a-z0-9])",
    re.I,
)
BROWSER_UA_RE = re.compile(r"\b(?:mozilla|chrome|firefox|safari|edge|trident|msie|opera|samsungbrowser)\b", re.I)
LOLBIN_ABUSE_RE = re.compile(
    r"(?:\bmsxsl(?:\.exe)?\b[^\n\r]{0,180}https?://|"
    r"\bcertutil(?:\.exe)?\b[^\n\r]{0,160}-urlcache\s+-split\s+-f\s+https?://|"
    r"\bdesktopimgdownldr(?:\.exe)?\b[^\n\r]{0,180}/lockscreenurl:\s*https?://|"
    r"\bms-appinstaller://\?source=https?://|"
    r"\bwevtutil(?:\.exe)?\b\s+cl\s+Security\b|"
    r"\blsass[_-]?dump(?:\.exe)?\b|\bprocdump(?:64)?(?:\.exe)?\b[^\n\r]{0,120}\blsass\b|"
    r"\b(?:powershell|pwsh)(?:\.exe)?\b[^\n\r]{0,120}\s-(?:enc|encodedcommand)\b|"
    r"\b(?:curl|wget)\b[^\n\r]{0,140}\|\s*(?:sh|bash)\b|"
    r"\bcat\s+/etc/shadow\b|\bnc\b\s+\d{1,3}(?:\.\d{1,3}){3}\s+4444\b)",
    re.I,
)
EXPLOIT_PATH_RE = re.compile(
    r"(?:/\.env\b|/wp-admin/?\b|/wp-login\.php\b|/xmlrpc\.php\b|/phpmyadmin/?\b|/server-status\b|"
    r"\.\./|%2e%2e%2f|union\s+select|'\s*or\s*1\s*=\s*1|<script|javascript:|/etc/passwd|cmd=|exec=|payload=|rule=(?:SQLI|RCE|PATH_TRAVERSAL)|\b(?:sqli|sql\s*injection|rce|xss|lfi|rfi|ssti|path[_-]?traversal)\b)",
    re.I,
)
SECURITY_MAL_RE = re.compile(
    r"(?:\bverdict\s*[:=]\s*(?:malicious|phish|malware|spoof)\b|\bET\s+(?:MALWARE|TROJAN)\b|\bPossible\s+C2\b|\bGPL\s+ATTACK\b|\bNmap\s+Scripting\s+Engine\b|\bSuspicious\s+TLS\s+SNI\b|\baction\s*[:=]\s*(?:BLOCK|DROP|QUARANTINE|REVOKE)\b|\bdecision\s*[:=]\s*(?:BLOCK|QUARANTINE|ALERT|REVOKE)\b)",
    re.I,
)
SECURITY_CLEAN_RE = re.compile(
    r"(?:\bverdict\s*[:=]\s*(?:benign|clean|legit|safe)\b|\baction\s*[:=]\s*DELIVER\b|\bdecision\s*[:=]\s*ALLOW\b|\bfalse\s*positive\b|\bnon[-_ ]?malicious\b|\bnot[-_ ]?malicious\b)",
    re.I,
)
C2_EXFIL_RE = re.compile(r"\b(?:c2|beacon|heartbeat|callback|exfil|dns[-_ ]?tunnel|dns[-_ ]?exfil|dnscat2|iodine|meterpreter|cobalt\s*strike|sliver|ransomware|xmrig)\b", re.I)
RISK_DOMAIN_RE = re.compile(
    r"(?:^|\.)(?:[a-z0-9-]{6,24}\d[a-z0-9-]*\.(?:biz|download|top|xyz|icu|pw|tk|link|click|zip|cfd|sbs)|"
    r"(?:login|signin|sso|auth|verify|update|security|download|cdn|files|drive)[-_.][a-z0-9-]+\.(?:biz|download|top|xyz|icu|pw|tk|link|click|zip|cfd|sbs)|"
    r"xn--[a-z0-9-]+)",
    re.I,
)
IDENTITY_FAIL_RE = re.compile(r"\b(?:FAIL|REJECT|DENY|DENIED|TIMEOUT|NOT_SENT|UNEXPECTED|BAD_PASSWORD|NEW_GEO|RARE_GEO|risk\s*[:=]\s*HIGH|risk\s*[:=]\s*CRITICAL)\b", re.I)
IDENTITY_SUCCESS_LOW_RE = re.compile(r"\b(?:SUCCESS|PASS|APPROVE|ACCEPT|LOGON)\b", re.I)
CLOUD_CONTROL_RE = re.compile(r"\b(?:StopLogging|DisableMailboxAudit|ExportWorkspaceData|bypass_dlp|CreateTransportRule|disable_security_checks|repo\.disable_branch_protection|org\.oauth_app_authorize|CreateAccessKey|PutBucketPolicy|AddForwarding|CreateAccessKey)\b", re.I)
K8S_CONTAINER_RE = re.compile(r"\b(?:cluster-admin|clusterrolebindings|privileged_container|privileged=true|Created\s+privileged\s+pod|verb\s*[:=]\s*exec|resource\s*[:=]\s*secrets|curl\s+https?://[^\s]+\s*\|\s*(?:sh|bash))\b", re.I)
DATA_EXFIL_RE = re.compile(r"\b(?:LARGE_EXPORT|COPY_OUT|/export\b|customers\.csv|db\.tgz|secrets\.txt|personal_drive|USB|RESTRICTED|CONFIDENTIAL|PII|FINANCE)\b", re.I)
DHCP_OK_RE = re.compile(r"\b(?:Renew|Assign|Release|DHCPACK|DHCPREQUEST|DHCPOFFER|DHCPDISCOVER)\b", re.I)


def _series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series(default, index=df.index if df is not None else None, dtype="object")
    return df[col].fillna(default).astype(str)


def _num(df: Optional[pd.DataFrame], col: str, n: int, default: float = 0.0) -> np.ndarray:
    if df is None or col not in df.columns:
        return np.full(n, default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)


def _normalize_domain_value(x: Any) -> str:
    s = "" if x is None else str(x).strip().lower()
    if not s or s in {"none", "nan", "null", "-", "--", "unknown", "notprovided", "unknown.local", "local.invalid"}:
        return ""
    s = re.sub(r"^https?://", "", s)
    s = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    s = s.split("@")[-1].split(":")[0].strip("[] .")
    return s


def _domain_suffix_hit(dom: pd.Series, values: Iterable[str]) -> np.ndarray:
    vals = {_normalize_domain_value(v) for v in (values or []) if _normalize_domain_value(v)}
    if not vals or len(dom) == 0:
        return np.zeros(len(dom), dtype=bool)
    d = dom.map(_normalize_domain_value)
    codes, uniq = pd.factorize(d, sort=False)
    def hit_one(x: str) -> bool:
        if not x:
            return False
        if x in vals:
            return True
        parts = x.split(".")
        for k in range(2, min(5, len(parts)) + 1):
            if ".".join(parts[-k:]) in vals:
                return True
        return False
    out_u = np.fromiter((hit_one(u) for u in uniq), dtype=bool, count=len(uniq)) if len(uniq) else np.array([], dtype=bool)
    return out_u[codes] if len(codes) else np.zeros(len(dom), dtype=bool)


def _ip_hit(ip_s: pd.Series, bad_ips: Iterable[str]) -> np.ndarray:
    vals = {str(x).strip() for x in (bad_ips or []) if str(x).strip()}
    if not vals:
        return np.zeros(len(ip_s), dtype=bool)
    return ip_s.fillna("").astype(str).str.strip().isin(vals).to_numpy(dtype=bool)


def _is_public_ip_one(x: str) -> bool:
    try:
        ip = ipaddress.ip_address(str(x).strip().strip("[]"))
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified)
    except Exception:
        return False


def _contains(series: pd.Series, rx: re.Pattern) -> np.ndarray:
    return series.str.contains(rx, na=False).to_numpy(dtype=bool)


def compute_override_signals(
    df: pd.DataFrame,
    X: Optional[pd.DataFrame] = None,
    odd_used: Optional[np.ndarray] = None,
    bad_ips: Optional[Iterable[str]] = None,
    bad_domains: Optional[Iterable[str]] = None,
    top_domains: Optional[Iterable[str]] = None,
    whitelist_domains: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return primary override flags, reason codes, and debug masks.

    This function is intentionally conservative: strong family-specific evidence
    or hard IOCs trigger automatic malicious; softer automation/bot/path signals
    become suspicious tags unless combined with risk signals.
    """
    if df is None or len(df) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int8), {}

    n = len(df)
    bad_ips_all = set(bad_ips or set()) | OBS_IOC_IPS
    bad_domains_all = {str(d).lower().strip() for d in (bad_domains or set())} | OBS_IOC_DOMAINS
    trusted_domains = BUILTIN_TRUSTED_DOMAINS | {str(d).lower().strip() for d in (top_domains or set())} | {str(d).lower().strip() for d in (whitelist_domains or set())}

    raw = _series(df, "raw_log")
    ua = _series(df, "user_agent")
    dom = _series(df, "domain")
    url = _series(df, "full_url")
    path = _series(df, "url_path")
    ref = _series(df, "referrer")
    cmd = _series(df, "command")
    proc = _series(df, "process")
    outc = _series(df, "outcome")
    proto = _series(df, "proto")
    log_type = (_series(df, "log_type") + " " + _series(df, "source_log_type")).str.lower()
    method = _series(df, "method").str.upper()

    blob = raw + " " + ua + " " + dom + " " + url + " " + path + " " + ref + " " + cmd + " " + proc + " " + outc + " " + proto + " " + log_type
    blob_l = blob.str.lower()

    client_ip = _series(df, "client_ip")
    dest_ip = _series(df, "dest_ip")
    ip_bad = _ip_hit(client_ip, bad_ips_all) | _ip_hit(dest_ip, bad_ips_all)
    dom_bad = _domain_suffix_hit(dom, bad_domains_all) | _domain_suffix_hit(url.map(_normalize_domain_value), bad_domains_all) | _domain_suffix_hit(ref.map(_normalize_domain_value), bad_domains_all)
    hard_ioc = ip_bad | dom_bad

    trusted_hit = _domain_suffix_hit(dom, trusted_domains) | _domain_suffix_hit(ref.map(_normalize_domain_value), trusted_domains)
    public_dest = dest_ip.map(_is_public_ip_one).to_numpy(dtype=bool)

    status = _num(df, "status", n, 200).astype(int)
    bytes_out = _num(df, "bytes_out", n, 0.0)
    bytes_in = _num(df, "bytes_in", n, 0.0)
    high_bytes = (bytes_out >= 250_000) | ((_num(X, "high_bytes_out", n, 0) > 0) if X is not None else False)
    exfil_ratio = bytes_out / (bytes_in + 1024.0)

    odd = (np.asarray(odd_used).astype(int) > 0) if odd_used is not None else (_num(X, "odd_hours", n, 0) > 0)
    suspicious_geo = _num(X, "suspicious_geo", n, 0) > 0
    suspicious_url = (_num(X, "suspicious_url", n, 0) > 0) | _contains(blob, EXPLOIT_PATH_RE)
    ipv6_tunnel = _num(X, "ipv6_tunnel_any", n, 0) > 0
    if "ipv6_tunnel_any" in df.columns:
        ipv6_tunnel = ipv6_tunnel | (_num(df, "ipv6_tunnel_any", n, 0) > 0)

    scanner = _contains(blob, SCANNER_RE)
    automation = _contains(blob, AUTOMATION_UA_RE)
    browser = _contains(ua, BROWSER_UA_RE)
    lolbin_abuse = _contains(blob, LOLBIN_ABUSE_RE)
    exploit_path = _contains(blob, EXPLOIT_PATH_RE)
    security_mal = _contains(blob, SECURITY_MAL_RE)
    clean_text = _contains(blob, SECURITY_CLEAN_RE)
    c2_exfil_text = _contains(blob, C2_EXFIL_RE)
    risky_domain = _contains(dom.map(_normalize_domain_value), RISK_DOMAIN_RE) | _contains(blob, RISK_DOMAIN_RE)
    identity_risk_text = _contains(blob, IDENTITY_FAIL_RE)
    cloud_control = _contains(blob, CLOUD_CONTROL_RE)
    k8s_container = _contains(blob, K8S_CONTAINER_RE)
    data_exfil_text = _contains(blob, DATA_EXFIL_RE)

    is_dhcp = log_type.str.contains(r"\bdhcp\b|type7_asset_csv", regex=True, na=False).to_numpy(dtype=bool) | _contains(blob, DHCP_OK_RE)
    is_proxy_web = log_type.str.contains(r"proxy|apache|web|edge|waf|lb|apigw|envoy|type1_space|type4_web_csv|type5_proxy_csv", regex=True, na=False).to_numpy(dtype=bool)
    is_security = log_type.str.contains(r"ids|edr|email_sec|dlp|casb|waf", regex=True, na=False).to_numpy(dtype=bool)
    is_network = log_type.str.contains(r"flow|zeek|tls|dns|firewall|cloud_flow|type8_firewall_csv|lb|edge", regex=True, na=False).to_numpy(dtype=bool)
    is_identity = log_type.str.contains(r"idp|ad_auth|mfa|aaa|pam|type6_event_csv|type9_dynamic_csv|linux_auth", regex=True, na=False).to_numpy(dtype=bool)
    is_os = log_type.str.contains(r"linux_auth|auditd|mac_es|fim|edr|type3_proc_csv|type6_event_csv", regex=True, na=False).to_numpy(dtype=bool)
    is_cloud_saas = log_type.str.contains(r"cloud_audit|saas_admin|gh_audit|cicd|db_audit|objstore|secrets|kms|dlp|casb", regex=True, na=False).to_numpy(dtype=bool)
    is_container = log_type.str.contains(r"k8s|container|docker|container_runtime|k8s_event|k8s_audit", regex=True, na=False).to_numpy(dtype=bool)

    # Family-specific hard rules.
    detection_rule_mal = (
        blob_l.str.contains(r"detection\s+rule", regex=True, na=False).to_numpy(dtype=bool)
        & blob_l.str.contains(r"(?:malware|exploitation\s+tool|bots?\s*&?\s*vulnerability\s+scanner|suricata\s+malware|bad[-_ ]?bot)", regex=True, na=False).to_numpy(dtype=bool)
        & ~clean_text
    )
    security_verdict = (
        detection_rule_mal
        | (is_security & security_mal & ~clean_text)
        | (log_type.str.contains("ids", na=False).to_numpy(dtype=bool) & (security_mal | c2_exfil_text | scanner))
        | (log_type.str.contains("edr", na=False).to_numpy(dtype=bool) & (blob_l.str.contains(r"verdict\s*[:=]\s*malicious", regex=True, na=False).to_numpy(dtype=bool) | lolbin_abuse))
        | (log_type.str.contains("email_sec", na=False).to_numpy(dtype=bool) & blob_l.str.contains(r"verdict\s*[:=]\s*(?:phish|malware|spoof)|action\s*[:=]\s*(?:drop|quarantine)", regex=True, na=False).to_numpy(dtype=bool))
    )

    waf_block = (
        is_proxy_web
        & blob_l.str.contains(r"(?:action\s*[:=]\s*(?:block|deny|drop)|waf\s*[:=]\s*block)", regex=True, na=False).to_numpy(dtype=bool)
        & (exploit_path | scanner | (pd.to_numeric(df.get("severity", pd.Series(0, index=df.index)), errors="coerce").fillna(0).to_numpy(dtype=float) >= 70))
    )
    apache_method_attack = (
        log_type.str.contains("apache|type1_space|type4_web_csv", na=False).to_numpy(dtype=bool)
        & method.isin(["PUT", "DELETE", "PATCH"]).to_numpy(dtype=bool)
        & ((status >= 400) | exploit_path | scanner | risky_domain)
    )
    exploit_or_waf = waf_block | apache_method_attack | (exploit_path & (scanner | hard_ioc | (status >= 400)))

    os_abuse = (
        is_os & (
            lolbin_abuse
            | blob_l.str.contains(r"failed\s+password\s+for\s+root|too\s+many\s+authentication\s+failures|invalid\s+user|/etc/shadow", regex=True, na=False).to_numpy(dtype=bool)
        )
    )

    network_c2_exfil = (
        is_network
        & (
            hard_ioc
            | c2_exfil_text
            | (high_bytes & public_dest & (risky_domain | suspicious_geo | scanner | automation | odd))
            | (exfil_ratio >= 40.0)
            | blob_l.str.contains(r"\b(?:dst|resp|dest_ip)[=:]?\s*[^\s]+:4444\b|\bnc\b[^\n\r]+\s4444\b", regex=True, na=False).to_numpy(dtype=bool)
            | (log_type.str.contains("dns", na=False).to_numpy(dtype=bool) & (risky_domain | blob_l.str.contains(r"qtype\s*[:=]\s*(?:txt|aaaa)|nxdomain", regex=True, na=False).to_numpy(dtype=bool) & risky_domain))
        )
    )

    proxy_web_payload = (
        is_proxy_web
        & ~is_dhcp
        & (
            hard_ioc
            | (scanner & (risky_domain | exploit_path | high_bytes | status >= 400 | odd))
            | (automation & (exploit_path | risky_domain | high_bytes | hard_ioc | status >= 400))
            | (method.isin(["PUT", "DELETE", "PATCH"]).to_numpy(dtype=bool) & (status >= 400) & ~trusted_hit)
        )
    )

    identity_abuse = (
        is_identity
        & ~is_dhcp
        & (
            hard_ioc
            | (identity_risk_text & (suspicious_geo | hard_ioc | odd | blob_l.str.contains(r"risk\s*[:=]\s*(?:high|critical)|new_geo|unexpected|bad_password|fail|reject|deny", regex=True, na=False).to_numpy(dtype=bool)))
            | blob_l.str.contains(r"wevtutil\s+cl\s+Security|Domain\s+Admins|adminlogin\s+outcome\s*[:=]\s*fail", regex=True, na=False).to_numpy(dtype=bool)
        )
    )

    cloud_saas_abuse = (
        is_cloud_saas
        & (
            hard_ioc
            | cloud_control
            | (data_exfil_text & (high_bytes | public_dest | hard_ioc | blob_l.str.contains(r"decision\s*[:=]\s*(?:block|quarantine|alert|revoke)|severity\s*[:=]\s*(?:high|critical)|anomaly\s*[:=]\s*rare", regex=True, na=False).to_numpy(dtype=bool)))
            | blob_l.str.contains(r"grant\s*[:=]\s*SUPERUSER|COPY_OUT|LARGE_EXPORT|prod/master-key|SecretGet.*admin", regex=True, na=False).to_numpy(dtype=bool)
        )
    )

    container_abuse = (
        is_container
        & (
            k8s_container
            | lolbin_abuse
            | hard_ioc
            | blob_l.str.contains(r"resource\s*[:=]\s*secrets.*decision\s*[:=]\s*allow|Created\s+privileged\s+pod", regex=True, na=False).to_numpy(dtype=bool)
        )
    )

    email_dlp = (
        log_type.str.contains(r"email_sec|dlp|casb", regex=True, na=False).to_numpy(dtype=bool)
        & (
            security_verdict
            | data_exfil_text
            | blob_l.str.contains(r"decision\s*[:=]\s*(?:block|quarantine|alert|revoke)|verdict\s*[:=]\s*(?:phish|malware|spoof)|attach\s*=.*\.exe", regex=True, na=False).to_numpy(dtype=bool)
        )
        & ~clean_text
    )

    ipv6_combo = ipv6_tunnel & (hard_ioc | suspicious_geo | suspicious_url | automation | scanner | odd | high_bytes)

    moderate_combo = (
        (scanner | automation | risky_domain | c2_exfil_text)
        & (hard_ioc | suspicious_geo | suspicious_url | high_bytes | odd | (status >= 400) | ipv6_tunnel)
        & ~trusted_hit
        & ~is_dhcp
    )

    # Strong IOCs are enough, except for normal DHCP/clean logs where an IP may
    # appear as assigned/leased asset data rather than adversary infrastructure.
    hard_ioc_primary = hard_ioc & ~is_dhcp

    base = (
        hard_ioc_primary
        | security_verdict
        | exploit_or_waf
        | os_abuse
        | network_c2_exfil
        | proxy_web_payload
        | identity_abuse
        | cloud_saas_abuse
        | container_abuse
        | email_dlp
        | ipv6_combo
        | moderate_combo
    )

    # Benign/safe contexts suppress only weak/regex conditions, never hard family
    # evidence or IOCs.
    benign_dhcp = is_dhcp & ~hard_ioc & ~lolbin_abuse & ~security_mal
    benign_clean_security = clean_text & ~hard_ioc & ~lolbin_abuse & ~exploit_path & ~data_exfil_text
    benign_firewall_allow = (
        log_type.str.contains(r"firewall|type8_firewall_csv", regex=True, na=False).to_numpy(dtype=bool)
        & blob_l.str.contains(r"\bALLOW\b|action\s*[:=]\s*ACCEPT", regex=True, na=False).to_numpy(dtype=bool)
        & ~hard_ioc & (bytes_out < 250_000)
    )
    benign_proxy_top = is_proxy_web & trusted_hit & browser & ~hard_ioc & ~exploit_path & ~scanner & ~lolbin_abuse
    benign_identity_success = is_identity & _contains(blob, IDENTITY_SUCCESS_LOW_RE) & ~identity_risk_text & ~hard_ioc & ~lolbin_abuse
    benign_app_ok = log_type.str.contains(r"\bapp\b|cicd", regex=True, na=False).to_numpy(dtype=bool) & blob_l.str.contains(r"status\s*[:=]\s*ok|result\s*[:=]\s*OK", regex=True, na=False).to_numpy(dtype=bool) & ~cloud_control & ~hard_ioc

    suppress = benign_dhcp | benign_clean_security | benign_firewall_allow | benign_proxy_top | benign_identity_success | benign_app_ok
    strong = hard_ioc_primary | security_verdict | exploit_or_waf | lolbin_abuse | network_c2_exfil | identity_abuse | cloud_saas_abuse | container_abuse | email_dlp | ipv6_combo
    conf = base & (~suppress | strong)

    conds = [
        hard_ioc_primary,
        security_verdict,
        exploit_or_waf,
        os_abuse,
        network_c2_exfil,
        proxy_web_payload,
        identity_abuse,
        cloud_saas_abuse,
        container_abuse,
        email_dlp,
        ipv6_combo,
        moderate_combo,
    ]
    codes = np.select(conds, list(range(1, 13)), default=0).astype(np.int8)
    codes = np.where(conf, codes, 0).astype(np.int8)

    tag_scanner = scanner | (is_proxy_web & (status >= 400) & (automation | scanner | exploit_path))
    tag_threat = hard_ioc | security_verdict | lolbin_abuse | c2_exfil_text | exploit_or_waf | network_c2_exfil | identity_abuse | cloud_saas_abuse | container_abuse | email_dlp | ipv6_combo
    tag_suspicious = tag_threat | tag_scanner | moderate_combo | (automation & ~trusted_hit & ~is_dhcp) | risky_domain | data_exfil_text
    tag_suspicious = tag_suspicious & ~(benign_dhcp | (benign_clean_security & ~tag_threat) | (benign_proxy_top & ~tag_threat) | (benign_identity_success & ~tag_threat))

    debug: Dict[str, np.ndarray] = {
        "conf_hit_explicit": hard_ioc_primary.astype(bool),
        "conf_hit_critical": security_verdict.astype(bool),
        "conf_hit_ops": os_abuse.astype(bool),
        "conf_hit_ipv6_combo": ipv6_combo.astype(bool),
        "conf_hit_post_put": apache_method_attack.astype(bool),
        "conf_hit_404": ((status == 404) & (scanner | exploit_path | risky_domain)).astype(bool),
        "conf_hit_wl_combo": (trusted_hit & tag_threat).astype(bool),
        "conf_hit_moderate_combo": moderate_combo.astype(bool),
        "conf_benign_asserted": suppress.astype(bool),
        "conf_tag_scanner": tag_scanner.astype(bool),
        "conf_tag_threat": tag_threat.astype(bool),
        "conf_tag_suspicious": tag_suspicious.astype(bool),
        "conf_family_dhcp_suppressed": benign_dhcp.astype(bool),
        "conf_family_clean_suppressed": benign_clean_security.astype(bool),
        "conf_family_trusted_web_suppressed": benign_proxy_top.astype(bool),
    }
    return conf.astype(bool), codes.astype(np.int8), debug
