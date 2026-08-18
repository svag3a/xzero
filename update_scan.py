#!/usr/bin/env python3
"""
update_scan.py – Uppdatera finansiella nyckeltal för en befintlig scan.

Usage:
  python3 update_scan.py --scan-id 21 --revenue 245 --ebit 18.5
  python3 update_scan.py --scan-id 21 --revenue 245 --ebit 18.5 --api-url http://13.48.24.83

Miljövariabler:
  ADMIN_TOKEN  (om satt i appen)
"""

import argparse
import os
import sys
import requests


def main():
    parser = argparse.ArgumentParser(description="Uppdatera finansiella nyckeltal för en scan")
    parser.add_argument("--scan-id",  required=True, type=int, help="Scan-id att uppdatera")
    parser.add_argument("--revenue",  required=True, type=float, help="Nettoomsättning i MSEK")
    parser.add_argument("--ebit",     required=True, type=float, help="EBIT (rörelseresultat) i MSEK")
    parser.add_argument("--industry", default=None, help="Bransch (valfri)")
    parser.add_argument("--api-url",  default="http://13.48.24.83")
    args = parser.parse_args()

    ebit_margin = round(args.ebit / args.revenue * 100, 1) if args.revenue else 0.0

    admin_token = os.environ.get("ADMIN_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if admin_token:
        headers["X-Admin-Token"] = admin_token

    payload = {
        "scan_id":        args.scan_id,
        "revenue_msek":   args.revenue,
        "ebit_msek":      args.ebit,
        "ebit_margin_pct": ebit_margin,
    }
    if args.industry:
        payload["industry"] = args.industry

    print(f"Uppdaterar scan {args.scan_id}:")
    print(f"  Omsättning: {args.revenue} MSEK")
    print(f"  EBIT:       {args.ebit} MSEK  ({ebit_margin}%)")
    if args.industry:
        print(f"  Bransch:    {args.industry}")

    try:
        resp = requests.post(
            f"{args.api_url}/admin/update-scan-financials",
            headers=headers,
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        print(f"  OK: {resp.json()}")
    except requests.HTTPError as e:
        print(f"  FEL {e.response.status_code}: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"  FEL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
