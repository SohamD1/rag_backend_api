from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


class UserFacingError(Exception):
    pass


@dataclass(frozen=True)
class SourceItem:
    index: int
    source_url: str
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class FileItem:
    index: int
    path: Path


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UserFacingError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UserFacingError(f"Invalid JSON in {path}: {exc}") from exc


def _coerce_sources(raw: Any) -> list[SourceItem]:
    if isinstance(raw, dict):
        for key in ("sources", "items", "data"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise UserFacingError("sources JSON must be a list (or an object with a 'sources' list).")

    items: list[SourceItem] = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            source_url = entry.strip()
            if not source_url:
                raise UserFacingError(f"sources[{i}] is an empty string.")
            items.append(SourceItem(index=i, source_url=source_url, raw=None))
            continue

        if isinstance(entry, dict):
            source_url = ""
            for key in ("source_url", "sourceUrl", "url", "link", "source"):
                v = entry.get(key)
                if isinstance(v, str) and v.strip():
                    source_url = v.strip()
                    break
            if not source_url:
                raise UserFacingError(
                    f"sources[{i}] is missing a source url (expected one of: "
                    "source_url, sourceUrl, url, link, source)."
                )
            items.append(SourceItem(index=i, source_url=source_url, raw=entry))
            continue

        raise UserFacingError(f"sources[{i}] must be a string or object; got {type(entry).__name__}.")

    return items


def _iter_files(docs_dir: Path, patterns: list[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in patterns:
        for p in docs_dir.glob(pattern):
            if not p.is_file():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            yield rp


def _sort_files(files: list[Path], order: str) -> list[Path]:
    if order == "name":
        return sorted(files, key=lambda p: p.name.lower())
    if order == "creation":
        return sorted(files, key=lambda p: (p.stat().st_ctime, p.name.lower()))
    if order == "modified":
        return sorted(files, key=lambda p: (p.stat().st_mtime, p.name.lower()))
    if order == "none":
        return list(files)
    raise UserFacingError(f"Unknown order: {order}")


def _format_preview_row(idx: int, file_path: Path, source_url: str) -> str:
    name = file_path.name
    url = source_url
    if len(url) > 120:
        url = url[:117] + "..."
    return f"{idx:>4}  {name:<55}  {url}"


def _build_matches(
    *,
    sources: list[SourceItem],
    files: list[Path],
    allow_mismatch: bool,
) -> list[dict[str, Any]]:
    if len(sources) != len(files) and not allow_mismatch:
        raise UserFacingError(
            f"Count mismatch: sources={len(sources)} files={len(files)}. "
            "Fix the folder/patterns or pass --allow-mismatch to pair by the shorter length."
        )

    pair_count = min(len(sources), len(files))
    matches: list[dict[str, Any]] = []
    for i in range(pair_count):
        matches.append(
            {
                "index": i,
                "file_path": str(files[i]),
                "file_name": files[i].name,
                "source_index": sources[i].index,
                "source_url": sources[i].source_url,
            }
        )
    return matches


def _post_document(
    *,
    client: httpx.Client,
    dashboard_token: str,
    pdf_path: Path,
    source_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    with pdf_path.open("rb") as f:
        resp = client.post(
            "/api/admin/documents",
            headers={"Authorization": f"Bearer {dashboard_token}"},
            data={"source_url": source_url},
            files={"file": (pdf_path.name, f, "application/pdf")},
            timeout=timeout_seconds,
        )

    if resp.status_code >= 400:
        detail = ""
        try:
            body = resp.json()
            detail = body.get("detail") or ""
        except Exception:
            detail = resp.text[:500]
        raise UserFacingError(
            f"Upload failed ({resp.status_code}) file={pdf_path.name} source_url={source_url} detail={detail}"
        )
    try:
        return resp.json()
    except Exception as exc:
        raise UserFacingError(f"Upload returned non-JSON response: {resp.text[:500]}") from exc


def _login_dashboard(*, client: httpx.Client, code: str, timeout_seconds: float) -> str:
    resp = client.post(
        "/api/admin/login",
        json={"code": code},
        timeout=timeout_seconds,
    )
    if resp.status_code >= 400:
        detail = ""
        try:
            body = resp.json()
            detail = body.get("detail") or ""
        except Exception:
            detail = resp.text[:500]
        raise UserFacingError(f"Dashboard login failed ({resp.status_code}) detail={detail}")
    try:
        token = str(resp.json().get("token") or "").strip()
    except Exception as exc:
        raise UserFacingError(f"Dashboard login returned non-JSON response: {resp.text[:500]}") from exc
    if not token:
        raise UserFacingError("Dashboard login did not return a token.")
    return token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview (and optionally upload) a folder of PDFs matched to an ordered sources.json.",
    )
    parser.add_argument("--docs-dir", required=True, help="Folder containing PDFs to ingest.")
    parser.add_argument("--sources-json", required=True, help="Path to sources JSON file.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern(s) to select files (default: *.pdf). Can be repeated.",
    )
    parser.add_argument(
        "--order",
        choices=["creation", "modified", "name", "none"],
        default="creation",
        help="How to order files before index-matching (default: creation).",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow sources/files count mismatch (pairs by shorter length).",
    )
    parser.add_argument(
        "--out",
        default="ingest_matches.preview.json",
        help="Where to write the preview mapping JSON (default: ingest_matches.preview.json).",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("KB_API_BASE_URL", "http://localhost:8000"),
        help="API base URL (default: env KB_API_BASE_URL or http://localhost:8000).",
    )
    parser.add_argument(
        "--dashboard-code",
        default=None,
        help="Current dashboard TOTP code (overrides env DASHBOARD_TOTP_CODE).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually upload documents after preview (otherwise preview-only).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (only with --apply).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue uploading after an error (only with --apply).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between uploads to avoid rate limits (default: 0).",
    )

    args = parser.parse_args(argv)

    docs_dir = Path(args.docs_dir).expanduser()
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise UserFacingError(f"--docs-dir is not a folder: {docs_dir}")

    patterns = args.pattern or ["*.pdf"]

    sources_path = Path(args.sources_json).expanduser()
    sources = _coerce_sources(_read_json(sources_path))

    files = list(_iter_files(docs_dir, patterns))
    files = _sort_files(files, args.order)

    out_path = Path(args.out).expanduser()
    matches = _build_matches(sources=sources, files=files, allow_mismatch=bool(args.allow_mismatch))
    out_path.write_text(json.dumps(matches, indent=2), encoding="utf-8")

    print(f"Sources: {len(sources)}")
    print(f"Files:   {len(files)} (patterns={patterns}, order={args.order})")
    print(f"Pairs:   {len(matches)}")
    print(f"Preview written: {out_path.resolve()}")
    print("")
    print(f"{'IDX':>4}  {'FILE':<55}  SOURCE_URL")
    print("-" * 140)
    preview_count = min(30, len(matches))
    for i in range(preview_count):
        print(_format_preview_row(i, Path(matches[i]["file_name"]), str(matches[i]["source_url"])))
    if len(matches) > preview_count:
        print(f"... ({len(matches) - preview_count} more; see {out_path.name})")

    if not args.apply:
        return 0

    if not args.yes:
        resp = input("Proceed to upload these matches? Type 'yes' to continue: ").strip().lower()
        if resp != "yes":
            _eprint("Aborted.")
            return 1

    dashboard_code = (args.dashboard_code or os.getenv("DASHBOARD_TOTP_CODE") or "").strip()
    api_base = str(args.api_base or "").strip().rstrip("/")
    if not api_base:
        raise UserFacingError("--api-base is empty.")
    if not dashboard_code:
        dashboard_code = input("Dashboard verification code: ").strip()

    with httpx.Client(base_url=api_base) as client:
        dashboard_token = _login_dashboard(
            client=client,
            code=dashboard_code,
            timeout_seconds=float(args.timeout),
        )
        failures = 0
        for m in matches:
            pdf_path = Path(m["file_path"])
            source_url = str(m["source_url"])
            try:
                result = _post_document(
                    client=client,
                    dashboard_token=dashboard_token,
                    pdf_path=pdf_path,
                    source_url=source_url,
                    timeout_seconds=float(args.timeout),
                )
                doc_id = result.get("doc_id", "")
                print(f"OK  file={pdf_path.name} doc_id={doc_id} source_url={source_url}")
            except Exception as exc:
                failures += 1
                _eprint(f"ERR file={pdf_path.name} source_url={source_url} error={exc}")
                if not args.continue_on_error:
                    break
            if args.sleep_seconds and args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))

    if failures:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        _eprint(str(exc))
        raise SystemExit(2)
