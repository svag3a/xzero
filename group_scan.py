#!/usr/bin/env python3
"""
group_scan.py – Koncernscan: OCR + Tavily per bolag → en samlad Bedrock-analys.

Filstruktur i S3 (ett subfolder per bolag, max 5 PDF:er per bolag):
  s3://xzero-scans/koncernnamn/
    Moderbolaget AB/
      arsred_2023.pdf
      arsred_2022.pdf
      ...
    Dotterbolag 1 AB/
      arsred_2023.pdf
      ...

Flöde:
  1. Listar subfoldrar → ett bolag per subfolder
  2. OCR:ar upp till 5 PDF:er per bolag (nyast filnamn sist → sorterat fallande)
  3. Tavily-sökning per bolag → lägger till som extra textpost
  4. Skickar allt som ett paket till /admin/scan-from-texts → en Bedrock-analys

Usage:
  python3 group_scan.py \\
    --bucket xzero-scans \\
    --prefix koncernnamn/ \\
    --orgnr 5561234567 \\
    --company "Koncernnamn AB" \\
    --contact-name "Anna Svensson" \\
    --contact-email anna@koncernen.se \\
    --api-url http://13.48.24.83

Miljövariabler:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  TAVILY_API_KEY   (för webinfo per bolag)
  ADMIN_TOKEN      (om satt i appen)
"""

import argparse
import os
import sys
import time
import json
import requests
import boto3

MAX_REPORTS_PER_COMPANY = 5
MAX_CHARS_PER_DOC       = 80_000
MAX_WAIT_SECS           = 600
POLL_INTERVAL           = 10


# ── Textract ─────────────────────────────────────────────────────────────────

def start_textract_job(textract, bucket: str, key: str) -> str:
    resp = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    return resp["JobId"]


def wait_for_textract(textract, job_id: str, label: str) -> str:
    elapsed = 0
    while elapsed < MAX_WAIT_SECS:
        resp = textract.get_document_text_detection(JobId=job_id)
        status = resp["JobStatus"]

        if status == "SUCCEEDED":
            pages = [resp]
            while "NextToken" in resp:
                resp = textract.get_document_text_detection(
                    JobId=job_id, NextToken=resp["NextToken"]
                )
                pages.append(resp)
            lines = [
                b["Text"]
                for page in pages
                for b in page.get("Blocks", [])
                if b["BlockType"] == "LINE"
            ]
            text = "\n".join(lines)
            if len(text) > MAX_CHARS_PER_DOC:
                text = text[:MAX_CHARS_PER_DOC]
            print(f"      ✓ {len(text):,} tecken")
            return text

        if status == "FAILED":
            raise RuntimeError(f"Textract misslyckades: {resp.get('StatusMessage','')}")

        print(f"      status={status}, väntar {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Textract-jobb tog för lång tid (>{MAX_WAIT_SECS}s)")


# ── Tavily ───────────────────────────────────────────────────────────────────

def fetch_tavily(company_name: str) -> str | None:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return None
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key":     key,
                "query":       f"{company_name} verksamhet erbjudande kunder",
                "max_results": 5,
            },
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        parts = []
        for r in results:
            parts.append(f"## {r.get('title','')}\n{r.get('content','')}")
        return "\n\n".join(parts)[:40_000]
    except Exception as e:
        print(f"      Tavily-fel: {e}")
        return None


# ── S3-helpers ───────────────────────────────────────────────────────────────

def list_company_folders(s3, bucket: str, prefix: str) -> list[str]:
    """Returnerar unika direkta subfoldrar under prefix."""
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    paginator = s3.get_paginator("list_objects_v2")
    folders = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            folders.add(cp["Prefix"])
    return sorted(folders)


def list_pdfs_in_folder(s3, bucket: str, folder: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=folder):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                keys.append(obj["Key"])
    # Sortera fallande på filnamn (nyast årsredovisning antas ha högst år i namnet)
    keys.sort(reverse=True)
    return keys[:MAX_REPORTS_PER_COMPANY]


# ── Huvudflöde ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Koncernscan: OCR + Tavily → xZero")
    parser.add_argument("--bucket",        required=True)
    parser.add_argument("--prefix",        default="",    help="Rot-prefix i bucketen, t.ex. 'koncernnamn/'")
    parser.add_argument("--orgnr",         required=True, help="Moderbolagets org.nr")
    parser.add_argument("--company",       default="",    help="Koncernens namn")
    parser.add_argument("--contact-name",  default="")
    parser.add_argument("--contact-email", default="")
    parser.add_argument("--contact-phone", default="")
    parser.add_argument("--api-url",       default="http://13.48.24.83")
    parser.add_argument("--region",        default="")
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    region      = args.region or os.environ.get("AWS_DEFAULT_REGION") or "eu-north-1"
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    s3          = boto3.client("s3",       region_name=region)
    textract    = boto3.client("textract", region_name=region)

    # ── 1. Hitta bolag (subfoldrar) ──────────────────────────────────────────
    print(f"Listar bolag i s3://{args.bucket}/{args.prefix}...")
    folders = list_company_folders(s3, args.bucket, args.prefix)

    if not folders:
        print("Inga subfoldrar hittades. Skapa en subfolder per bolag.")
        sys.exit(1)

    company_names = [f.rstrip("/").split("/")[-1] for f in folders]
    print(f"  {len(folders)} bolag: {', '.join(company_names)}\n")

    # ── 2. OCR per bolag ─────────────────────────────────────────────────────
    all_texts = []

    for folder, company_name in zip(folders, company_names):
        print(f"── {company_name} ──")

        pdf_keys = list_pdfs_in_folder(s3, args.bucket, folder)
        if not pdf_keys:
            print("  Inga PDF:er, hoppar över")
            continue

        print(f"  {len(pdf_keys)} årsredovisningar – startar Textract-jobb...")
        jobs = []
        for key in pdf_keys:
            filename = key.split("/")[-1].replace(".pdf","").replace(".PDF","")
            job_id = start_textract_job(textract, args.bucket, key)
            jobs.append({"key": key, "filename": filename, "job_id": job_id})
            print(f"    → {filename}")

        print()
        for job in jobs:
            label = f"{company_name} – {job['filename']}"
            print(f"  Hämtar: {job['filename']}")
            try:
                text = wait_for_textract(textract, job["job_id"], label)
                all_texts.append({"label": label, "text": text})
            except Exception as e:
                print(f"      FEL: {e}")

        # ── 3. Tavily per bolag ──────────────────────────────────────────────
        print(f"  Hämtar webbinfo för {company_name}...")
        web_text = fetch_tavily(company_name)
        if web_text:
            all_texts.append({
                "label": f"{company_name} – Webbinfo",
                "text":  web_text,
            })
            print(f"      ✓ {len(web_text):,} tecken")
        else:
            print(f"      Ingen webbinfo (TAVILY_API_KEY saknas eller inga resultat)")
        print()

    # ── Sammanfattning ───────────────────────────────────────────────────────
    print(f"{'═'*50}")
    print(f"  Totalt {len(all_texts)} textposter klara:")
    for t in all_texts:
        print(f"    {t['label'][:60]:<60}  {len(t['text']):>8,} tecken")
    print(f"{'═'*50}\n")

    if not all_texts:
        print("Inga texter att analysera. Avslutar.")
        sys.exit(1)

    if args.dry_run:
        print("[dry-run] Skickar inte till appen.")
        sys.exit(0)

    # ── 4. Skicka till appen ─────────────────────────────────────────────────
    print(f"Skickar till {args.api_url}/admin/scan-from-texts...")
    headers = {"Content-Type": "application/json"}
    if admin_token:
        headers["X-Admin-Token"] = admin_token

    try:
        resp = requests.post(
            f"{args.api_url}/admin/scan-from-texts",
            headers=headers,
            json={
                "orgnr":         args.orgnr,
                "company_name":  args.company,
                "contact_name":  args.contact_name,
                "contact_email": args.contact_email,
                "contact_phone": args.contact_phone,
                "texts":         all_texts,
            },
            timeout=30,
        )
        resp.raise_for_status()
        print(f"  OK: {resp.json()}")
    except requests.HTTPError as e:
        print(f"  FEL {e.response.status_code}: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"  FEL: {e}")
        sys.exit(1)

    print()
    print("Bedrock-analysen körs nu i bakgrunden på servern (~5–15 min för en hel koncern).")
    print("Resultatet sparas i CRM:et och skickas till kontaktpersonen när det är klart.")


if __name__ == "__main__":
    main()
