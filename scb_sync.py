#!/usr/bin/env python3
"""
scb_sync.py – Ladda ner SCB-bulkfilen och synka alla svenska AB till appen.

Körs månadsvis av cloud-agenten (eller manuellt):
  python3 scb_sync.py --api-url https://scan.zeroworks.se --token $SCB_IMPORT_TOKEN

Flöde:
  1. Laddar ner scb_bulkfil.zip (~67 MB) från Bolagsverket
  2. Parsar alla aktiva AB (JurForm=49, FtgStat=1)
  3. POSTar i batchar om 500 bolag till /api/scb/import
  4. Avslutar med /api/scb/finalize för att markera borttagna bolag som inaktiva
"""

import argparse
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone

import requests

SCB_URL = "https://vardefulla-datamangder.bolagsverket.se/scb/scb_bulkfil.zip"
BATCH_SIZE = 500


def download_scb(dest: str):
    print(f"Laddar ner SCB-fil från {SCB_URL}...")
    resp = requests.get(SCB_URL, stream=True, timeout=600)
    resp.raise_for_status()
    total = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            total += len(chunk)
    print(f"Nedladdad: {total / 1024 / 1024:.1f} MB")


def parse_scb(zip_path: str) -> list:
    companies = []
    with zipfile.ZipFile(zip_path) as z:
        fname = z.namelist()[0]
        with z.open(fname) as f:
            header = f.readline().decode("latin-1", errors="replace").rstrip().split("\t")
            try:
                jurform_idx = header.index("JurForm")
                orgnr_idx   = header.index("PeOrgNr")
                namn_idx    = header.index("Namn")
                ftgstat_idx = header.index("FtgStat")
                ng1_idx     = header.index("Ng1")
            except ValueError as e:
                raise RuntimeError(f"Kolumn saknas i SCB-fil: {e}")

            ng_extra_idx = [header.index(f"Ng{i}") for i in range(2, 6) if f"Ng{i}" in header]

            for line in f:
                row = line.decode("latin-1", errors="replace").rstrip().split("\t")
                if len(row) <= ng1_idx:
                    continue
                if row[jurform_idx] != "49":
                    continue
                if row[ftgstat_idx] != "1":
                    continue

                raw = row[orgnr_idx]
                if len(raw) == 12 and raw.startswith("16"):
                    orgnr = raw[2:]
                elif len(raw) == 10 and raw.isdigit():
                    orgnr = raw
                else:
                    continue

                primary_sni = row[ng1_idx]
                sni_codes = [primary_sni]
                for idx in ng_extra_idx:
                    if idx < len(row) and row[idx] and row[idx] != primary_sni:
                        sni_codes.append(row[idx])

                companies.append({
                    "orgnr":        orgnr,
                    "company_name": row[namn_idx].strip(),
                    "primary_sni":  primary_sni,
                    "sni_codes":    sni_codes,
                })

    print(f"Parsade {len(companies):,} aktiva AB")
    return companies


def post_batch(api_url: str, token: str, companies: list, sync_at: str, sess: requests.Session):
    resp = sess.post(
        f"{api_url}/api/scb/import",
        json={"companies": companies, "sync_at": sync_at},
        headers={"X-Import-Token": token},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("imported", 0)


def finalize(api_url: str, token: str, sync_at: str, sess: requests.Session):
    resp = sess.post(
        f"{api_url}/api/scb/finalize",
        json={"sync_at": sync_at},
        headers={"X-Import-Token": token},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("deactivated", 0)


def main():
    parser = argparse.ArgumentParser(description="Synka SCB-bolag till appen")
    parser.add_argument("--api-url", default="https://scan.zeroworks.se")
    parser.add_argument("--token",   default=os.environ.get("SCB_IMPORT_TOKEN", ""),
                        help="Import-token (eller sätt SCB_IMPORT_TOKEN i env)")
    parser.add_argument("--scb-file", default=None,
                        help="Använd lokal zip-fil istället för att ladda ner")
    parser.add_argument("--no-finalize", action="store_true",
                        help="Skippa finalize-steget (markera ej bort borttagna bolag)")
    args = parser.parse_args()

    if not args.token:
        print("FEL: --token eller SCB_IMPORT_TOKEN krävs", file=sys.stderr)
        sys.exit(1)

    sync_at = datetime.now(timezone.utc).isoformat()
    print(f"Sync startar: {sync_at}")

    # Ladda ner eller använd lokal fil
    zip_path = args.scb_file
    tmp_file = None
    if not zip_path:
        tmp_file = f"/tmp/scb_bulkfil_{datetime.now().strftime('%Y%m')}.zip"
        if os.path.exists(tmp_file):
            print(f"Använder cachad fil: {tmp_file}")
        else:
            download_scb(tmp_file)
        zip_path = tmp_file

    companies = parse_scb(zip_path)

    sess = requests.Session()
    total_imported = 0
    batches = [companies[i:i+BATCH_SIZE] for i in range(0, len(companies), BATCH_SIZE)]
    print(f"Skickar {len(batches)} batchar à {BATCH_SIZE} bolag...")

    for i, batch in enumerate(batches, 1):
        try:
            n = post_batch(args.api_url, args.token, batch, sync_at, sess)
            total_imported += n
            if i % 20 == 0 or i == len(batches):
                pct = i / len(batches) * 100
                print(f"  Batch {i}/{len(batches)} ({pct:.0f}%) — {total_imported:,} importerade")
        except Exception as e:
            print(f"  FEL batch {i}: {e}", file=sys.stderr)
            time.sleep(5)
            continue

    print(f"Import klar: {total_imported:,} bolag")

    if not args.no_finalize:
        deactivated = finalize(args.api_url, args.token, sync_at, sess)
        print(f"Finalize: {deactivated} bolag markerade inaktiva")

    print("Sync klar!")


if __name__ == "__main__":
    main()
