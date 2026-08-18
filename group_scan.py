#!/usr/bin/env python3
"""
group_scan.py – OCR + gruppscan för skannade årsredovisningar i S3.

Flöde:
  1. Listar PDF-filer i S3-bucketen (eller ett prefix)
  2. Kör AWS Textract async på varje fil
  3. Väntar på alla jobb, hämtar extraherad text
  4. Skickar alla texter till /admin/scan-from-texts → Bedrock kör analysen

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
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION (eller AWS_REGION)
  ADMIN_TOKEN  (om satt i appen)

Tips – skapa bucket och ladda upp:
  aws s3 mb s3://xzero-scans --region eu-north-1
  aws s3 cp ./arsredovisningar/ s3://xzero-scans/koncernnamn/ --recursive
"""

import argparse
import os
import sys
import time
import json
import boto3
import requests

MAX_WAIT_SECS = 600   # max 10 min per Textract-jobb
POLL_INTERVAL = 10    # sekunder mellan statuspolling


def start_textract_job(client, bucket: str, key: str) -> str:
    resp = client.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    return resp["JobId"]


def wait_for_textract(client, job_id: str, label: str) -> str:
    """Väntar på Textract-jobb och returnerar all extraherad text."""
    elapsed = 0
    while elapsed < MAX_WAIT_SECS:
        resp = client.get_document_text_detection(JobId=job_id)
        status = resp["JobStatus"]

        if status == "SUCCEEDED":
            pages = [resp]
            # Hämta eventuella extra sidor
            while "NextToken" in resp:
                resp = client.get_document_text_detection(
                    JobId=job_id, NextToken=resp["NextToken"]
                )
                pages.append(resp)

            lines = []
            for page in pages:
                for block in page.get("Blocks", []):
                    if block["BlockType"] == "LINE":
                        lines.append(block["Text"])
            text = "\n".join(lines)
            print(f"    {label}: {len(text):,} tecken extraherade")
            return text

        if status == "FAILED":
            raise RuntimeError(f"Textract misslyckades för {label}: {resp.get('StatusMessage','')}")

        print(f"    {label}: status={status}, väntar {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Textract-jobb {job_id} tog för lång tid ({MAX_WAIT_SECS}s)")


def submit_group_scan(
    api_url: str,
    admin_token: str,
    orgnr: str,
    company_name: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str,
    texts: list,
) -> dict:
    headers = {"Content-Type": "application/json"}
    if admin_token:
        headers["X-Admin-Token"] = admin_token

    resp = requests.post(
        f"{api_url}/admin/scan-from-texts",
        headers=headers,
        json={
            "orgnr":         orgnr,
            "company_name":  company_name,
            "contact_name":  contact_name,
            "contact_email": contact_email,
            "contact_phone": contact_phone,
            "texts":         texts,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="OCR + gruppscan via Textract → xZero")
    parser.add_argument("--bucket",        required=True,  help="S3-bucket med PDF:erna")
    parser.add_argument("--prefix",        default="",     help="Prefix/mapp i bucketen (t.ex. 'kund/')")
    parser.add_argument("--orgnr",         required=True,  help="Org.nr för moderbolaget")
    parser.add_argument("--company",       default="",     help="Koncernnamn (visas i CRM)")
    parser.add_argument("--contact-name",  default="",     help="Kontaktpersonens namn")
    parser.add_argument("--contact-email", default="",     help="Kontaktpersonens e-post")
    parser.add_argument("--contact-phone", default="",     help="Kontaktpersonens telefon")
    parser.add_argument("--api-url",       default="http://13.48.24.83", help="App-URL")
    parser.add_argument("--region",        default="",     help="AWS-region (default: eu-north-1)")
    parser.add_argument("--dry-run",       action="store_true", help="Kör Textract men skicka inte till appen")
    args = parser.parse_args()

    region = args.region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "eu-north-1"
    admin_token = os.environ.get("ADMIN_TOKEN", "")

    s3       = boto3.client("s3",       region_name=region)
    textract = boto3.client("textract", region_name=region)

    # Lista PDF:er i bucketen
    print(f"Listar PDF:er i s3://{args.bucket}/{args.prefix}...")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                keys.append(key)

    if not keys:
        print("Inga PDF-filer hittades. Kontrollera bucket och prefix.")
        sys.exit(1)

    print(f"  Hittade {len(keys)} PDF-filer:")
    for k in keys:
        print(f"    {k}")
    print()

    # Starta Textract-jobb för alla
    print("Startar Textract-jobb...")
    jobs = []
    for key in keys:
        label = key.split("/")[-1].replace(".pdf", "").replace(".PDF", "")
        job_id = start_textract_job(textract, args.bucket, key)
        jobs.append({"key": key, "label": label, "job_id": job_id})
        print(f"  {label} → job_id={job_id}")

    print()

    # Vänta på och hämta resultat
    print("Väntar på Textract-resultat...")
    texts = []
    errors = []
    for job in jobs:
        print(f"  [{job['label']}]")
        try:
            text = wait_for_textract(textract, job["job_id"], job["label"])
            # Trunkera till 80 000 tecken per dokument för att hålla tokenbudget
            MAX_CHARS = 80_000
            if len(text) > MAX_CHARS:
                print(f"    → trunkerar till {MAX_CHARS:,} tecken")
                text = text[:MAX_CHARS]
            texts.append({"label": job["label"], "text": text})
        except Exception as e:
            print(f"    FEL: {e}")
            errors.append(job["label"])

    print()
    print(f"OCR klar: {len(texts)} dokument lyckades, {len(errors)} misslyckades")
    if errors:
        print(f"  Misslyckades: {', '.join(errors)}")

    if not texts:
        print("Inga texter att analysera. Avslutar.")
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] Skulle skickat följande till /admin/scan-from-texts:")
        for t in texts:
            print(f"  {t['label']}: {len(t['text']):,} tecken")
        sys.exit(0)

    # Skicka till appen
    print(f"\nSkickar till {args.api_url}/admin/scan-from-texts...")
    try:
        result = submit_group_scan(
            api_url=args.api_url,
            admin_token=admin_token,
            orgnr=args.orgnr,
            company_name=args.company,
            contact_name=args.contact_name,
            contact_email=args.contact_email,
            contact_phone=args.contact_phone,
            texts=texts,
        )
        print(f"  OK: {result}")
        print()
        print("Bedrock-analysen körs nu i bakgrunden på servern.")
        print("Resultatet sparas automatiskt i CRM:et och skickas till kontaktpersonen när det är klart.")
    except requests.HTTPError as e:
        print(f"  FEL {e.response.status_code}: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"  FEL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
