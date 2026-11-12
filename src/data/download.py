"""Download NOAA USCRN hourly02 station files.

Discovers file names from the official NOAA directory listing (no guessed
URLs), downloads each yearly file with retries and resumable writes,
verifies size and content, and writes a download manifest.

Usage:
    python -m src.data.download
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.utils import get_logger, load_config, project_path

log = get_logger("download", "download.log")

HREF_RE = re.compile(r'href="([^"]+)"')
# Expected record layout (README section 5): line starts with 5-digit WBANNO,
# then UTC_DATE YYYYMMDD and UTC_TIME HHmm.
RECORD_RE = re.compile(r"^\d{5} \d{8} \d{4} ")


def list_directory(session: requests.Session, url: str, timeout: int) -> list[str]:
    """Return href targets found in an Apache-style NOAA directory listing."""
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return HREF_RE.findall(r.text)


def remote_size(session: requests.Session, url: str, timeout: int) -> int | None:
    r = session.head(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    cl = r.headers.get("Content-Length")
    return int(cl) if cl is not None else None


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    expected_size: int | None,
    max_retries: int,
    backoff: float,
    timeout: int,
) -> tuple[str, int, int]:
    """Download `url` to `dest` with retries and HTTP Range resume.

    Returns (status, bytes_on_disk, attempts).
    """
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        return "already_present", dest.stat().st_size, 0

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 416:  # range not satisfiable -> restart
                    part.unlink(missing_ok=True)
                    continue
                r.raise_for_status()
                mode = "ab" if (offset and r.status_code == 206) else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            size = part.stat().st_size
            if expected_size is not None and size != expected_size:
                raise IOError(f"size mismatch: got {size}, expected {expected_size}")
            part.replace(dest)
            return "downloaded", size, attempt
        except (requests.RequestException, IOError) as e:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, url, e)
            time.sleep(backoff * attempt)
    return "failed", part.stat().st_size if part.exists() else 0, max_retries


def verify_content(path: Path, year: int) -> dict:
    """Check that every line matches the documented record layout and year."""
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    bad = [i for i, ln in enumerate(lines) if not RECORD_RE.match(ln)]
    lst_years = {ln[20:24] for ln in lines if len(ln) > 24}
    widths = {len(ln) for ln in lines}
    return {
        "n_lines": len(lines),
        "n_malformed_lines": len(bad),
        "lst_years_in_file": sorted(lst_years),
        "line_widths": sorted(widths),
        "year_matches": lst_years <= {str(year), str(year - 1), str(year + 1)},
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg = load_config()
    dcfg = cfg["data"]
    station = dcfg["station"]
    base = dcfg["base_url"]
    raw_dir = project_path(dcfg["raw_dir"])
    meta_dir = project_path(dcfg["metadata_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    opts = dcfg["download"]

    session = requests.Session()
    session.headers["User-Agent"] = "precision-irrigation-research/1.0 (academic project)"

    # Documentation files (kept for provenance)
    for name, url in [("README_hourly02.txt", dcfg["readme_url"]), ("HEADERS.txt", dcfg["headers_url"])]:
        status, size, _ = download_file(session, url, meta_dir / name, None,
                                        opts["max_retries"], opts["backoff_seconds"], opts["timeout_seconds"])
        log.info("Documentation %s: %s (%d bytes)", name, status, size)

    root_listing = list_directory(session, base, opts["timeout_seconds"])
    available_years = {int(h.strip("/")) for h in root_listing if re.fullmatch(r"\d{4}/?", h)}
    log.info("NOAA hourly02 year directories available: %d-%d",
             min(available_years), max(available_years))

    manifest = []
    for year in range(dcfg["start_year"], dcfg["end_year"] + 1):
        rec = {"year": year, "station": station, "filename": None, "url": None,
               "remote_size_bytes": None, "local_size_bytes": None, "status": None,
               "sha256": None, "n_lines": None, "n_malformed_lines": None,
               "downloaded_at_utc": None, "note": ""}
        if year not in available_years:
            rec.update(status="year_not_in_archive")
            manifest.append(rec)
            log.error("Year %d not present in NOAA archive", year)
            continue
        year_url = f"{base}{year}/"
        listing = list_directory(session, year_url, opts["timeout_seconds"])
        # Match documented filename convention CRNH02TT-YYYY-${name}.txt exactly.
        pat = re.compile(rf"^CRNH02\d\d-{year}-{re.escape(station)}\.txt$")
        matches = [h for h in listing if pat.match(h)]
        if len(matches) != 1:
            rec.update(status="station_file_not_found",
                       note=f"{len(matches)} matching files in listing")
            manifest.append(rec)
            log.error("Year %d: station file not found (matches=%s)", year, matches)
            continue
        fname = matches[0]
        url = year_url + fname
        dest = raw_dir / fname
        size_remote = remote_size(session, url, opts["timeout_seconds"])
        status, size_local, attempts = download_file(
            session, url, dest, size_remote,
            opts["max_retries"], opts["backoff_seconds"], opts["timeout_seconds"])
        rec.update(filename=fname, url=url, remote_size_bytes=size_remote,
                   local_size_bytes=size_local, status=status,
                   downloaded_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if status in ("downloaded", "already_present"):
            v = verify_content(dest, year)
            rec.update(sha256=sha256(dest), n_lines=v["n_lines"],
                       n_malformed_lines=v["n_malformed_lines"])
            if v["n_malformed_lines"] or not v["year_matches"]:
                rec["status"] = "content_verification_failed"
                rec["note"] = json.dumps(v)
            log.info("Year %d: %s %s (%s bytes, %d lines, %d malformed, attempts=%d)",
                     year, rec["status"], fname, size_local, v["n_lines"],
                     v["n_malformed_lines"], attempts)
        else:
            log.error("Year %d: download failed for %s", year, url)
        manifest.append(rec)

    mdf = pd.DataFrame(manifest)
    mdf.to_csv(meta_dir / "download_manifest.csv", index=False)
    with open(meta_dir / "download_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    ok = mdf["status"].isin(["downloaded", "already_present"])
    log.info("Files OK: %d / %d; years OK: %s", ok.sum(), len(mdf),
             mdf.loc[ok, "year"].tolist())
    if not ok.all():
        log.error("Some years failed: %s", mdf.loc[~ok, ["year", "status", "note"]].to_dict("records"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
