#!/usr/bin/env python3
"""
List every file in a public Google Drive folder via Drive API v3 (API-key auth).

Requires the folder to be shared as 'Anyone with the link can view'.
Recurses into subfolders by default. Writes {filename: file_id} to JSON.

One-time setup:
  1. https://console.cloud.google.com  ->  create/select a project
  2. Enable 'Google Drive API' for that project
  3. APIs & Services -> Credentials -> Create credentials -> API key
  4. (recommended) restrict the key to 'Google Drive API' only
  5. Save the key string to  config/key_google_api.txt

Usage:
  python HTR_pipeline/list_drive_folder.py \\
      --folder-id 1U7jIuhzGyfyoShMGdJ6oTRTkx9GqCv2o \\
      --api-key-file config/key_google_api.txt \\
      --out config/drive_file_ids.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

API_BASE = "https://www.googleapis.com/drive/v3/files"


def list_folder(folder_id: str, api_key: str, recurse: bool = True) -> dict[str, str]:
    """Return {filename: file_id} for everything (non-folder) under `folder_id`."""
    out: dict[str, str] = {}
    todo: list[str] = [folder_id]
    seen: set[str] = set()
    while todo:
        fid = todo.pop()
        if fid in seen:
            continue
        seen.add(fid)
        page_token: str | None = None
        while True:
            params = {
                "q": f"'{fid}' in parents and trashed=false",
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": 1000,
                "key": api_key,
            }
            if page_token:
                params["pageToken"] = page_token
            r = requests.get(API_BASE, params=params, timeout=60)
            if r.status_code != 200:
                raise SystemExit(
                    f"Drive API error {r.status_code}: {r.text[:500]}\n"
                    "Common causes: folder not shared 'Anyone with the link', "
                    "Drive API not enabled for the project, or wrong key."
                )
            data = r.json()
            for f in data.get("files", []):
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    if recurse:
                        todo.append(f["id"])
                else:
                    name = f["name"]
                    if name in out and out[name] != f["id"]:
                        print(f"  warning: duplicate filename {name!r}; keeping first id",
                              file=sys.stderr)
                    else:
                        out[name] = f["id"]
            page_token = data.get("nextPageToken")
            if not page_token:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder-id",    required=True)
    ap.add_argument("--api-key-file", type=Path, required=True)
    ap.add_argument("--out",          type=Path, required=True)
    ap.add_argument("--no-recurse",   action="store_true")
    args = ap.parse_args()

    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit(f"Empty API key file: {args.api_key_file}")

    files = list_folder(args.folder_id, api_key, recurse=not args.no_recurse)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(files, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}  ({len(files)} files)")


if __name__ == "__main__":
    main()
