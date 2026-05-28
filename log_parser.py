
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
_TS_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\s+[+-]\d{4})?$")
_TS_SHORT_MDY_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$")
_IPV4_RE = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")
_APACHE_RE = re.compile(r'^(?P<client_ip>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+HTTP/\d(?:\.\d+)?"\s+(?P<status>\d{3})\s+(?P<bytes_out>\d+|-)\s+"(?P<referrer>[^"]*)"\s+"(?P<ua>[^"]*)"')
_SPACE_DET_RE = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\s+(?P<client_ip>(?:(?:\d{1,3}\.){3}\d{1,3}))\s+(?P<method>GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(?P<path>/\S*)\s+(?P<status>\d{3})\s+(?P<bytes_out>\d+)\s+(?P<bytes_in>\d+)\s+(?P<rest>.+)$')
_SYSLOG_RE = re.compile(r'^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<proc>[^\[:]+)(?:\[\d+\])?:\s+(?P<msg>.+)$')
_AUDITD_MSG_RE = re.compile(r'msg=audit\((?P<epoch>\d+(?:\.\d+)?):(?P<eid>\d+)\)')
_AUDITD_A0_RE = re.compile(r'\ba0="([^"]+)"')
_KEYVAL_PAIR_RE = re.compile(r'([A-Za-z0-9_./-]+)=(".*?"|\'.*?\'|\S+)')
_REQ_RE = re.compile(r'^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(\S+)$', re.I)
_URL_RE = re.compile(r'https?://[^\s"\'<>]+', re.I)
_HOST_END_RE = re.compile(r'([a-z0-9.-]+\.[a-z]{2,63})$', re.I)

FAMILY_ALIAS = {
    "dns": "dns", "flow": "flow", "ids": "ids", "zeek_conn": "zeek_conn", "tls": "tls",
    "idp": "idp", "ad_auth": "ad_auth", "pam": "pam", "aaa": "aaa", "mfa": "mfa",
    "edr": "edr", "mac_es": "mac_es", "fim": "fim", "cloud_audit": "cloud_audit", "cloud_flow": "cloud_flow",
    "k8s_audit": "k8s_audit", "containerd": "container_runtime", "docker": "container_runtime",
    "secrets": "secrets_kms", "kms": "secrets_kms", "waf": "waf", "apigw": "apigw", "lb": "lb", "app": "app",
    "envoy": "envoy", "db_audit": "db_audit", "objstore": "objstore", "dlp": "dlp_casb", "casb": "dlp_casb",
    "saas_admin": "saas_admin", "email_sec": "email_sec", "gh_audit": "gh_audit", "cicd": "cicd",
    "k8s_event": "k8s_event", "ics": "ics", "edge": "edge",
    "dhcp": "dhcp", "firewall": "firewall", "proxy": "proxy", "apache": "apache", "security": "security"
}


_JSON_CANON_RENAMES = {
    "src_ip": "client_ip", "source_ip": "client_ip", "sourceip": "client_ip", "clientip": "client_ip",
    "ip": "client_ip", "ip_address": "client_ip", "ip_addr": "client_ip", "source_address": "client_ip",
    "src": "client_ip", "client": "client_ip", "caller_ip": "client_ip", "requester_ip": "client_ip",
    "dst_ip": "dest_ip", "destination_ip": "dest_ip", "destinationip": "dest_ip", "destip": "dest_ip",
    "dest": "dest_ip", "dst": "dest_ip", "remote_ip": "dest_ip", "server_ip": "dest_ip",
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
    "user": "username", "account": "username", "principal": "username", "login": "username", "user_name": "username", "actor": "username", "requester": "username", "caller": "username", "user_id": "username",
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

def _norm_json_key(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _record(raw_line: str, log_type: str = "futuristic_unknown") -> Dict[str, Any]:
    return {
        "timestamp": None, "hour": None, "client_ip": None, "dest_ip": None, "src_port": None, "dst_port": None,
        "proto": None, "method": None, "url_path": None, "status": None, "bytes_out": None, "bytes_in": None,
        "domain": None, "full_url": None, "referrer": None, "user_agent": None, "username": None,
        "process": None, "workstation": None, "command": None, "event_id": None, "logon_id": None,
        "outcome": None, "severity": None, "log_type": log_type, "source_log_type": log_type, "raw_log": raw_line,
    }


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip().strip('"').strip("'")
        if s in {"", "-"}:
            return None
        if "." in s and s.replace(".", "", 1).isdigit():
            return int(float(s))
        return int(s)
    except Exception:
        return None


def _strip_quotes(x: Any) -> str:
    s = "" if x is None else str(x).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1]
    return s.strip()


def _is_ipv4(s: str) -> bool:
    return bool(_IPV4_RE.match((s or "").strip()))


def _host_from_url(url: str) -> Optional[str]:
    try:
        p = urlparse(url)
        host = (p.netloc or "").split("@")[-1].split(":")[0].strip("[]").strip()
        return host or None
    except Exception:
        return None


def _path_from_url(url: str) -> Optional[str]:
    try:
        p = urlparse(url)
        path = p.path or "/"
        return path if path.startswith("/") else "/" + path
    except Exception:
        return None


def _build_full_url(domain: Optional[str], path: Optional[str], https: bool = False) -> Optional[str]:
    d = (domain or "").strip().strip("/")
    if not d:
        return None
    p = (path or "/").strip()
    if p == "":
        p = "/"
    if not p.startswith("/"):
        p = "/" + p
    return ("https://" if https else "http://") + d + p


def _basename(text: str) -> str:
    s = (text or "").strip().replace("\\", "/")
    return s.split("/")[-1] if s else ""


def _extract_url(text: str) -> Optional[str]:
    m = _URL_RE.search(text or "")
    return m.group(0) if m else None


def _extract_host_at_end(text: str) -> Tuple[Optional[str], str]:
    s = (text or "").strip()
    m = _HOST_END_RE.search(s)
    if not m:
        return None, s
    host = m.group(1).rstrip(".,;)]}\"'")
    before = s[:m.start()].rstrip(" ,")
    return host.lower(), before


def _normalize_ts_syslog(mon: str, day: str, hhmmss: str) -> str:
    return f"{datetime.now().year} {mon} {int(day):02d} {hhmmss}"


def _epoch_to_utc(epoch_s: str) -> str:
    try:
        dt = datetime.fromtimestamp(float(epoch_s), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S +0000")
    except Exception:
        return ""


def _parse_endpoint(val: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    s = _strip_quotes(val)
    if not s:
        return None, None
    if s.count(":") == 1 and _is_ipv4(s.split(":")[0]):
        host, port = s.split(":", 1)
        return host, _safe_int(port)
    return (s if _is_ipv4(s) else None), None


def _parse_kv_pairs(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in _KEYVAL_PAIR_RE.findall(body or ""):
        out[k] = _strip_quotes(v)
    return out


def _family(name: str) -> str:
    low = (name or "").strip().lower()
    return FAMILY_ALIAS.get(low, low or "futuristic_unknown")


def _parse_prefixed(line: str) -> Tuple[Optional[str], str]:
    if "\t" not in line:
        return None, line
    prefix, rest = line.split("\t", 1)
    p = prefix.strip().lower()
    if re.match(r"^[a-z_][a-z0-9_.-]{1,48}$", p):
        return p, rest
    return None, line


def _parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()
    if not s.startswith("{"):
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    # Generic JSON/table-row parser: preserve uploaded CSV/JSON headers by
    # canonicalizing aliases into the model's semantic schema instead of
    # collapsing everything into Column 1/2/3 style positional fields.
    rec = _record(str(obj.get("raw_log") or obj.get("raw") or line), "futuristic_unknown")
    rec["command"] = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    for key, val in obj.items():
        nk = _norm_json_key(key)
        target = _JSON_CANON_RENAMES.get(nk, nk)
        if target in rec:
            rec[target] = val
        elif target in {"source_row_id", "source_log_type"}:
            rec[target] = val

    # SaaS/anomaly JSON compatibility from the previous parser.
    if rec.get("username") in (None, ""):
        rec["username"] = obj.get("user_id") or obj.get("user") or obj.get("username")
    if obj.get("login_failures") is not None and rec.get("severity") in (None, ""):
        rec["severity"] = obj.get("login_failures")
    if (obj.get("data_download_gb") is not None or obj.get("off_hours_access") is not None) and not rec.get("outcome"):
        rec["outcome"] = f"download_gb={obj.get('data_download_gb')} off_hours_access={obj.get('off_hours_access')}"

    return rec


def _parse_apache(line: str) -> Optional[Dict[str, Any]]:
    m = _APACHE_RE.match(line)
    if not m:
        return None
    rec = _record(line, "apache")
    rec["timestamp"] = m.group("ts").strip()
    rec["client_ip"] = m.group("client_ip")
    rec["method"] = m.group("method")
    rec["url_path"] = m.group("path")
    rec["status"] = _safe_int(m.group("status"))
    rec["bytes_out"] = _safe_int(m.group("bytes_out"))
    ref = m.group("referrer")
    rec["referrer"] = None if ref == "-" else ref
    ua = m.group("ua")
    rec["user_agent"] = None if ua == "-" else ua
    return rec


def _parse_space_detection(line: str) -> Optional[Dict[str, Any]]:
    m = _SPACE_DET_RE.match(line)
    if not m:
        return None
    rec = _record(line, "type1_space")
    rec["timestamp"] = m.group("ts")
    rec["client_ip"] = m.group("client_ip")
    rec["method"] = m.group("method")
    rec["url_path"] = m.group("path")
    rec["status"] = _safe_int(m.group("status"))
    rec["bytes_out"] = _safe_int(m.group("bytes_out"))
    rec["bytes_in"] = _safe_int(m.group("bytes_in"))
    rest = m.group("rest").strip()
    domain, before = _extract_host_at_end(rest)
    if domain:
        rec["domain"] = domain
        rec["full_url"] = _build_full_url(domain, rec["url_path"], https=True)
    parts = [p.strip() for p in before.split(",")] if before else []
    if parts:
        rec["user_agent"] = parts[0] or None
        if len(parts) > 1:
            rec["command"] = ",".join([p for p in parts[1:] if p]) or None
        if len(parts) > 3:
            rec["outcome"] = parts[3] or None
        if len(parts) > 7:
            rec["severity"] = parts[7] or None
    return rec


def _parse_csv_like(line: str) -> Optional[Dict[str, Any]]:
    try:
        vals = [v.strip() for v in next(csv.reader([line]))]
    except Exception:
        return None
    if len(vals) < 4:
        return None

    if len(vals) >= 10 and _TS_ISO_RE.match(vals[0] or "") and _is_ipv4(vals[1]) and _is_ipv4(vals[2]) and vals[3].upper() in HTTP_METHODS:
        rec = _record(line, "type4_web_csv")
        rec["timestamp"] = vals[0]; rec["client_ip"] = vals[1]; rec["dest_ip"] = vals[2]
        rec["method"] = vals[3].upper(); rec["url_path"] = vals[4] or "/"; rec["referrer"] = None if vals[5] in {"", "-", "N/A", "n/a"} else vals[5]
        rec["status"] = _safe_int(vals[6]); rec["bytes_out"] = _safe_int(vals[7]); rec["bytes_in"] = _safe_int(vals[8]); rec["user_agent"] = vals[9] or None
        if len(vals) > 10: rec["username"] = vals[10] or None
        return rec

    if len(vals) >= 6 and _TS_ISO_RE.match(vals[0] or "") and not _is_ipv4(vals[1]) and not _is_ipv4(vals[2]) and (vals[5] == "" or _is_ipv4(vals[5])):
        rec = _record(line, "type3_proc_csv")
        rec["timestamp"] = vals[0]; rec["username"] = vals[1] or None; rec["workstation"] = vals[2] or None; rec["process"] = vals[3] or None; rec["command"] = vals[4] or None; rec["dest_ip"] = vals[5] or None
        url = _extract_url(vals[4] or "")
        if url:
            rec["full_url"] = url; rec["domain"] = _host_from_url(url); rec["url_path"] = _path_from_url(url)
        return rec

    if len(vals) >= 7 and _TS_ISO_RE.match(vals[1] or "") and _is_ipv4(vals[2]) and not _is_ipv4(vals[3]):
        rec = _record(line, "proxy")
        rec["source_log_type"] = "type5_proxy_csv"
        rec["source_row_id"] = vals[0] or None; rec["timestamp"] = vals[1]; rec["client_ip"] = vals[2]; rec["domain"] = vals[3].lower() if vals[3] else None; rec["referrer"] = vals[4] or None; rec["user_agent"] = vals[5] or None; rec["bytes_out"] = _safe_int(vals[6]); rec["full_url"] = _build_full_url(rec["domain"], "/", https=True)
        return rec

    if len(vals) >= 6 and _TS_ISO_RE.match(vals[1] or "") and re.fullmatch(r"\d{3,5}", vals[3] or ""):
        rec = _record(line, "windows_event")
        rec["source_log_type"] = "type6_event_csv"
        rec["source_row_id"] = vals[0] or None; rec["timestamp"] = vals[1]; rec["workstation"] = vals[2] or None; rec["event_id"] = _safe_int(vals[3]); rec["outcome"] = vals[4] or None; rec["command"] = vals[5] or None
        eid = rec["event_id"] or -1
        if eid in (4624, 4634, 4647):
            rec["process"] = "winlogon.exe"
        elif eid in (4688, 4689):
            rec["process"] = _basename(vals[5] or "") or "eventlog.exe"
            rec["referrer"] = "explorer.exe" if eid == 4688 else "eventlog"
        return rec

    if len(vals) >= 7 and _TS_SHORT_MDY_RE.match(vals[1] or "") and re.fullmatch(r"\d{2}:\d{2}:\d{2}", vals[2] or "") and _is_ipv4(vals[4]):
        rec = _record(line, "dhcp")
        rec["source_log_type"] = "type7_asset_csv"
        rec["source_row_id"] = vals[0] or None; rec["timestamp"] = f"{vals[1]} {vals[2]}"; rec["outcome"] = vals[3] or None; rec["method"] = (vals[3] or "DHCP").upper(); rec["client_ip"] = vals[4] or None; rec["workstation"] = vals[5] or None; rec["command"] = f"mac={vals[6]} action={vals[3]}" if vals[6] else vals[3] or None; rec["proto"] = "DHCP"; rec["status"] = 200
        return rec

    if len(vals) >= 7 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", vals[1] or "") and re.fullmatch(r"\d{2}:\d{2}:\d{2}", vals[2] or "") and _is_ipv4(vals[3]) and _is_ipv4(vals[4]) and vals[5].upper() in {"ALLOW","DENY","BLOCK"}:
        rec = _record(line, "firewall")
        rec["source_log_type"] = "type8_firewall_csv"
        rec["source_row_id"] = vals[0] or None; rec["timestamp"] = f"{vals[1]} {vals[2]}"; rec["client_ip"] = vals[3] or None; rec["dest_ip"] = vals[4] or None; rec["outcome"] = vals[5].upper(); rec["method"] = vals[5].upper(); rec["bytes_out"] = _safe_int(vals[6]); rec["proto"] = "NET"; rec["status"] = 200 if vals[5].upper() == "ALLOW" else 403
        return rec

    if len(vals) >= 8 and _TS_ISO_RE.match(vals[1] or "") and re.fullmatch(r"\d{3,5}", vals[5] or ""):
        rec = _record(line, "windows_logon")
        rec["source_log_type"] = "type9_dynamic_csv"
        rec["source_row_id"] = vals[0] or None; rec["timestamp"] = vals[1]; rec["workstation"] = vals[2] or None
        rec["username"] = vals[3] if (vals[3] and "." in vals[3]) else None
        if rec["username"] is None and vals[3]:
            rec["workstation"] = rec["workstation"] or vals[3]
        rec["dest_ip"] = vals[4] if _is_ipv4(vals[4]) else None; rec["event_id"] = _safe_int(vals[5]); rec["logon_id"] = vals[6] or None; rec["outcome"] = vals[7] or None
        return rec

    return None


def _parse_req(req: str) -> Tuple[Optional[str], Optional[str]]:
    m = _REQ_RE.match((req or "").strip())
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2)


def _parse_keyvalue_family(line: str, forced_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    m = re.match(r'^(?P<ts>\d{4}-\d{2}-\d{2}T[^\s]+)\s+(?P<fam>[A-Za-z0-9_]+)\s+(?P<body>.+)$', line)
    if not m:
        return None
    ts = m.group("ts"); fam = _family(forced_type or m.group("fam")); body = m.group("body"); kv = _parse_kv_pairs(body); rec = _record(line, fam); rec["timestamp"] = ts

    def gv(*names: str) -> Optional[str]:
        for name in names:
            if name in kv and kv[name] not in {"", "-"}:
                return kv[name]
        return None

    if fam == "dns":
        rec["client_ip"] = gv("client", "client_ip", "src", "src_ip"); rec["domain"] = gv("qname", "domain", "host"); rec["dest_ip"] = gv("answer") if _is_ipv4(gv("answer") or "") else None; qtype = gv("qtype") or "QUERY"; rec["method"] = "GET"; rec["url_path"] = f"/dns/{qtype.lower()}"; rec["full_url"] = f"dns://{rec['domain']}/{qtype.lower()}" if rec["domain"] else None; rec["proto"] = "DNS"; rec["outcome"] = gv("rcode"); rec["command"] = body; return rec

    if fam in {"flow", "cloud_flow", "zeek_conn"}:
        cip, sport = _parse_endpoint(gv("src", "orig")); dip, dport = _parse_endpoint(gv("dst", "resp"))
        rec["client_ip"], rec["src_port"] = cip, sport; rec["dest_ip"], rec["dst_port"] = dip, dport; rec["proto"] = gv("proto") or "NET"; rec["bytes_out"] = _safe_int(gv("bytes_out", "orig_bytes", "bytes")); rec["bytes_in"] = _safe_int(gv("bytes_in", "resp_bytes")); rec["outcome"] = gv("action", "state"); rec["command"] = body; return rec

    if fam == "tls":
        rec["client_ip"], _ = _parse_endpoint(gv("src")); rec["dest_ip"], _ = _parse_endpoint(gv("dst")); rec["proto"] = "TLS"; rec["domain"] = gv("sni", "host"); rec["url_path"] = "/tls/handshake"; rec["full_url"] = _build_full_url(rec["domain"], "/", https=True) if rec["domain"] else None; rec["command"] = body; return rec

    if fam == "ids":
        rec["client_ip"], _ = _parse_endpoint(gv("src")); rec["dest_ip"], _ = _parse_endpoint(gv("dst")); rec["proto"] = gv("proto") or "IDS"; rec["outcome"] = gv("alert", "rule"); rec["severity"] = _safe_int(gv("severity")); rec["method"] = "POST"; rec["url_path"] = "/ids/alert"; rec["command"] = body; return rec

    if fam in {"idp", "ad_auth", "pam", "aaa", "mfa"}:
        rec["client_ip"] = gv("ip", "src_ip", "src", "client_ip"); rec["username"] = gv("user", "principal", "requester", "actor", "account", "caller"); rec["workstation"] = gv("device", "host", "workstation"); rec["proto"] = fam.upper(); rec["outcome"] = gv("result", "action", "decision"); rec["command"] = body; app = gv("app", "device", "target"); rec["url_path"] = f"/{fam}/{re.sub(r'[^A-Za-z0-9._/-]+', '-', app)}" if app else f"/{fam}"; return rec

    if fam == "edr":
        rec["workstation"] = gv("host", "device"); rec["username"] = gv("user", "account"); rec["outcome"] = gv("verdict", "tactic", "technique"); rec["process"] = gv("proc", "process") or _basename(gv("path") or ""); rec["command"] = gv("path", "args") or body; rec["proto"] = "EDR"; return rec

    if fam == "mac_es":
        rec["workstation"] = gv("host"); rec["username"] = gv("user"); rec["outcome"] = gv("event", "result"); rec["process"] = _basename(gv("proc") or gv("path") or ""); rec["command"] = gv("args") or gv("path") or body; rec["dest_ip"], rec["dst_port"] = _parse_endpoint(gv("dst")); rec["proto"] = "HOST"; return rec

    if fam == "fim":
        rec["workstation"] = gv("host"); rec["username"] = gv("user"); rec["outcome"] = gv("action", "anomaly"); rec["command"] = gv("path") or body; rec["url_path"] = gv("path"); rec["proto"] = "FS"; return rec

    if fam in {"cloud_audit", "db_audit", "gh_audit", "cicd", "saas_admin", "email_sec", "objstore", "dlp_casb"}:
        rec["client_ip"] = gv("src_ip", "ip"); rec["username"] = gv("principal", "actor", "requester", "user", "to"); rec["outcome"] = gv("outcome", "verdict", "decision", "result"); rec["command"] = body; rec["proto"] = fam.upper()
        if fam == "objstore":
            bucket = gv("bucket") or "bucket"; key = gv("key") or ""; rec["method"] = gv("op"); rec["url_path"] = f"/{bucket}/{key}".replace("//", "/"); rec["domain"] = "objstore.local"; rec["full_url"] = _build_full_url("objstore.local", rec["url_path"], https=True); rec["bytes_out"] = _safe_int(gv("bytes"))
        elif fam == "db_audit":
            rec["method"] = gv("action"); table = gv("table") or gv("role") or gv("db") or "event"; rec["url_path"] = f"/db/{table}"; rec["bytes_out"] = _safe_int(gv("bytes_out"))
        elif fam == "email_sec":
            sender = gv("from"); rec["method"] = "POST"; rec["url_path"] = "/email/security"; rec["domain"] = sender.split("@", 1)[1].lower() if sender and "@" in sender else None
        elif fam == "gh_audit":
            rec["method"] = "POST"; repo = gv("repo") or gv("target") or gv("app") or "event"; rec["url_path"] = "/" + repo.strip("/")
        elif fam == "cicd":
            rec["method"] = "POST"; pipe = gv("pipeline") or "pipeline"; stage = gv("stage") or "stage"; rec["url_path"] = f"/{pipe}/{stage}"
        elif fam == "saas_admin":
            rec["method"] = "POST"; target = gv("target") or gv("dest") or gv("system") or "admin"; rec["url_path"] = "/" + str(target).strip("/")
        elif fam == "dlp_casb":
            rec["method"] = "POST"; dest = gv("dest") or gv("app") or "policy"; rec["url_path"] = "/" + str(dest).strip("/")
        return rec

    if fam in {"k8s_audit", "k8s_event", "container_runtime", "secrets_kms", "ics", "edge", "waf", "apigw", "lb", "app", "envoy"}:
        rec["command"] = body; rec["proto"] = fam.upper()
        if fam == "k8s_audit":
            rec["username"] = gv("user"); rec["method"] = gv("verb"); ns = gv("ns") or "default"; resource = gv("resource") or "resource"; obj = gv("obj") or "obj"; rec["url_path"] = f"/k8s/{ns}/{resource}/{obj}"; rec["outcome"] = gv("decision", "reason"); return rec
        if fam == "k8s_event":
            rec["workstation"] = gv("pod"); rec["outcome"] = gv("reason"); rec["method"] = "POST"; ns = gv("ns") or "default"; rec["url_path"] = f"/k8s/{ns}/event"; return rec
        if fam == "container_runtime":
            rec["workstation"] = gv("host"); rec["outcome"] = gv("event", "reason", "result"); rec["process"] = _basename(gv("cmd") or gv("image") or fam); rec["command"] = gv("cmd") or gv("image") or body; rec["method"] = "POST"; rec["url_path"] = "/container/runtime"; return rec
        if fam == "secrets_kms":
            rec["client_ip"] = gv("src_ip", "ip"); rec["username"] = gv("principal"); rec["method"] = gv("action") or "POST"; secret = gv("secret") or gv("key") or "resource"; rec["url_path"] = f"/secrets/{secret}"; rec["outcome"] = gv("outcome"); return rec
        if fam == "ics":
            rec["client_ip"] = gv("src"); rec["workstation"] = gv("device"); rec["method"] = gv("action") or "POST"; rec["outcome"] = gv("result", "severity"); proto = gv("proto"); rec["proto"] = proto or rec["proto"]; rec["url_path"] = f"/ics/{gv('device') or 'device'}"; return rec
        if fam == "edge":
            rec["client_ip"] = gv("client_ip"); rec["domain"] = gv("host"); rec["url_path"] = gv("uri") or "/"; rec["status"] = _safe_int(gv("status")); rec["bytes_out"] = _safe_int(gv("bytes_out")); rec["outcome"] = gv("waf", "bot", "ratelimit", "cache", "anomaly"); rec["method"] = "GET"; rec["full_url"] = _build_full_url(rec["domain"], rec["url_path"], https=True); return rec
        if fam == "waf":
            rec["client_ip"] = gv("client_ip"); rec["method"] = gv("method"); rec["url_path"] = gv("uri") or "/"; rec["status"] = _safe_int(gv("status")) or (403 if (gv("action") or "").upper() in {"BLOCK", "DENY"} else None); rec["outcome"] = gv("action", "rule"); rec["severity"] = _safe_int(gv("score")); rec["full_url"] = _build_full_url(gv("host") or "waf.edge.local", rec["url_path"], https=True); rec["domain"] = _host_from_url(rec["full_url"] or ""); return rec
        if fam == "apigw":
            rec["client_ip"] = gv("client_ip"); rec["method"] = gv("method"); rec["url_path"] = gv("path") or "/"; rec["status"] = _safe_int(gv("status")); rec["bytes_out"] = _safe_int(gv("bytes_out")); rec["outcome"] = gv("rate_limited", "key_id"); rec["domain"] = "api.gateway.local"; rec["full_url"] = _build_full_url("api.gateway.local", rec["url_path"], https=True); return rec
        if fam == "lb":
            cip, sport = _parse_endpoint(gv("client")); dip, dport = _parse_endpoint(gv("target")); rec["client_ip"], rec["src_port"] = cip, sport; rec["dest_ip"], rec["dst_port"] = dip, dport; meth, path = _parse_req(gv("req") or ""); rec["method"] = meth; rec["url_path"] = path; rec["status"] = _safe_int(gv("status")); rec["bytes_out"] = _safe_int(gv("sent")); rec["user_agent"] = gv("ua"); return rec
        if fam == "app":
            rec["client_ip"] = gv("ip"); rec["username"] = gv("user"); rec["method"] = "POST"; svc = gv("service") or "app"; action = gv("action") or "event"; rec["url_path"] = f"/{svc}/{action}"; rec["status"] = 200 if (gv("status") or "").upper() == "OK" else None; rec["bytes_out"] = _safe_int(gv("bytes")); rec["outcome"] = gv("reason", "anomaly", "status"); return rec
        if fam == "envoy":
            cip, _ = _parse_endpoint(gv("downstream")); rec["client_ip"] = cip; meth, path = _parse_req(gv("req") or ""); rec["method"] = meth; rec["url_path"] = path; rec["status"] = _safe_int(gv("code")); rec["bytes_out"] = _safe_int(gv("bytes_out")); cluster = gv("upstream_cluster") or "service"; rec["domain"] = f"{cluster}.svc.cluster.local"; rec["full_url"] = _build_full_url(rec["domain"], rec["url_path"], https=False); return rec

    rec["client_ip"] = gv("client_ip", "src_ip", "src", "client", "ip", "orig"); rec["dest_ip"] = gv("dest_ip", "dst_ip", "dst", "resp", "server_ip"); rec["username"] = gv("user", "username", "account", "principal", "actor", "requester"); rec["workstation"] = gv("host", "device", "workstation", "node"); rec["method"] = gv("method", "verb", "action"); rec["status"] = _safe_int(gv("status", "code")); rec["bytes_out"] = _safe_int(gv("bytes_out", "orig_bytes", "bytes")); rec["bytes_in"] = _safe_int(gv("bytes_in", "resp_bytes")); rec["proto"] = gv("proto"); rec["domain"] = gv("domain", "host", "qname", "sni"); rec["url_path"] = gv("path", "uri", "resource"); rec["full_url"] = gv("url"); rec["command"] = body; rec["outcome"] = gv("outcome", "result", "decision", "verdict"); rec["severity"] = _safe_int(gv("severity", "score"))
    return rec


def _parse_syslog_linux(line: str) -> Optional[Dict[str, Any]]:
    m = _SYSLOG_RE.match(line)
    if not m:
        return None
    msg = m.group("msg"); proc = m.group("proc"); rec = _record(line, "linux_auth"); rec["timestamp"] = _normalize_ts_syslog(m.group("mon"), m.group("day"), m.group("time")); rec["workstation"] = m.group("host"); rec["process"] = proc; rec["command"] = msg
    ipm = re.search(r'\bfrom\s+((?:(?:\d{1,3}\.){3}\d{1,3}))\b', msg)
    if ipm: rec["client_ip"] = ipm.group(1)
    userm = re.search(r'\bfor\s+([A-Za-z0-9_.-]+)\b', msg)
    if userm: rec["username"] = userm.group(1)
    low = msg.lower()
    if "failed password" in low or "authentication failure" in low: rec["outcome"] = "FAIL"; rec["status"] = 401
    elif "accepted publickey" in low or "accepted password" in low: rec["outcome"] = "SUCCESS"; rec["status"] = 200
    elif "disconnecting" in low: rec["outcome"] = "DISCONNECT"; rec["status"] = 429
    return rec


def _parse_auditd(line: str) -> Optional[Dict[str, Any]]:
    if "msg=audit(" not in line and not line.startswith("type="):
        return None
    rec = _record(line, "auditd")
    mm = _AUDITD_MSG_RE.search(line)
    if mm: rec["timestamp"] = _epoch_to_utc(mm.group("epoch")); rec["event_id"] = _safe_int(mm.group("eid"))
    a0 = _AUDITD_A0_RE.search(line)
    if a0:
        base = _basename(a0.group(1)); rec["process"] = base or a0.group(1)
    rec["command"] = line
    url = _extract_url(line)
    if url: rec["full_url"] = url; rec["domain"] = _host_from_url(url); rec["url_path"] = _path_from_url(url)
    rec["proto"] = "HOST"
    return rec


def _parse_generic_web(line: str) -> Optional[Dict[str, Any]]:
    url = _extract_url(line); meth = re.search(r'\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b', line); ip = re.search(r'\b((?:(?:\d{1,3}\.){3}\d{1,3}))\b', line)
    if not any((url, meth, ip)):
        return None
    rec = _record(line, "futuristic_unknown")
    if ip: rec["client_ip"] = ip.group(1)
    if meth: rec["method"] = meth.group(1)
    if url: rec["full_url"] = url; rec["domain"] = _host_from_url(url); rec["url_path"] = _path_from_url(url)
    return rec


def _finalize(rec: Dict[str, Any], forced_type: Optional[str] = None) -> Dict[str, Any]:
    if forced_type and rec.get("log_type") in {None, "", "futuristic_unknown"}:
        rec["log_type"] = _family(forced_type)
    if forced_type:
        rec["source_log_type"] = forced_type
    if isinstance(rec.get("timestamp"), str):
        rec["timestamp"] = rec["timestamp"].strip()
    if isinstance(rec.get("domain"), str) and rec["domain"]:
        rec["domain"] = rec["domain"].strip().strip("[]").rstrip(".,;)]}\"'").lower()
    if not rec.get("full_url") and rec.get("domain") and rec.get("url_path"):
        rec["full_url"] = _build_full_url(rec.get("domain"), rec.get("url_path"), https=str(rec.get("proto") or "").upper() == "TLS")
    if rec.get("full_url") and not rec.get("domain"):
        rec["domain"] = _host_from_url(rec["full_url"])
    if rec.get("full_url") and not rec.get("url_path"):
        rec["url_path"] = _path_from_url(rec["full_url"])
    if rec.get("process") and not rec.get("command"):
        rec["command"] = rec["process"]
    if rec.get("process"):
        proc = str(rec["process"]).strip()
        if " " in proc and "." not in proc:
            base = _basename(proc)
            if base:
                rec["process"] = base
    if rec.get("method"):
        rec["method"] = str(rec["method"]).strip().upper()
    for k in ("status", "bytes_out", "bytes_in", "src_port", "dst_port", "severity", "event_id"):
        if rec.get(k) is not None:
            rec[k] = _safe_int(rec[k])
    if not rec.get("source_log_type"):
        rec["source_log_type"] = rec.get("log_type", "futuristic_unknown")
    return rec


def parse_log_line_universal(raw_line: str, log_source: Optional[str] = None) -> Dict[str, Any]:
    line = (raw_line or "").rstrip("\n")
    forced_type, body = _parse_prefixed(line)
    for parser in (_parse_json_line, _parse_apache, _parse_space_detection, _parse_auditd, _parse_syslog_linux):
        rec = parser(body)
        if rec is not None:
            return _finalize(rec, forced_type=forced_type)
    rec = _parse_keyvalue_family(body, forced_type=forced_type)
    if rec is not None:
        return _finalize(rec, forced_type=forced_type)
    rec = _parse_csv_like(body)
    if rec is not None:
        return _finalize(rec, forced_type=forced_type)
    rec = _parse_generic_web(body)
    if rec is not None:
        return _finalize(rec, forced_type=forced_type)
    return _finalize(_record(line, _family(forced_type or "futuristic_unknown")), forced_type=forced_type)


def parse_log_universal(raw_line: str, log_source: Optional[str] = None) -> Dict[str, Any]:
    return parse_log_line_universal(raw_line, log_source=log_source)


def parse_universal_line(raw_line: str, log_source: Optional[str] = None) -> Dict[str, Any]:
    return parse_log_line_universal(raw_line, log_source=log_source)


__all__ = ["parse_log_line_universal", "parse_log_universal", "parse_universal_line"]
