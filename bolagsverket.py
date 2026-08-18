"""
Bolagsverket Värdefulla datamängder API – iXBRL-hämtning.

Dokumentation:
  https://bolagsverket.se/apierochoppnadata/hamtaforetagsinformation/vardefulladatamangder/

Registrering (gratis, inget avtal):
  https://bolagsverket.se/...kundanmalantillapiforvardefulladatamangder...

Env vars:
  BOLAGSVERKET_CLIENT_ID
  BOLAGSVERKET_CLIENT_SECRET
  BOLAGSVERKET_TOKEN_URL   (valfri, default = produktion)
  BOLAGSVERKET_BASE_URL    (valfri, default = produktion)
"""

import io
import os
import zipfile
import logging

import requests
from bs4 import BeautifulSoup

_TOKEN_URL_PROD = "https://portal.api.bolagsverket.se/oauth2/token"
_BASE_URL_PROD  = "https://gw.api.bolagsverket.se/vardefulla-datamangder/v1"
SCOPE           = "vardefulla-datamangder:read"


def _token_url() -> str:
    return os.environ.get("BOLAGSVERKET_TOKEN_URL") or _TOKEN_URL_PROD

def _base_url() -> str:
    return os.environ.get("BOLAGSVERKET_BASE_URL") or _BASE_URL_PROD


def _get_token() -> str:
    client_id     = os.environ["BOLAGSVERKET_CLIENT_ID"]
    client_secret = os.environ["BOLAGSVERKET_CLIENT_SECRET"]
    resp = requests.post(
        _token_url(),
        auth=(client_id, client_secret),   # Basic Auth – vanligaste OAuth2-formatet
        data={"grant_type": "client_credentials", "scope": SCOPE},
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(f"Token-fel {resp.status_code}: {resp.text}")
    return resp.json()["access_token"]


def _fetch_document_list(orgnr: str, token: str) -> list[dict]:
    resp = requests.post(
        f"{_base_url()}/dokumentlista",
        headers={"Authorization": f"Bearer {token}"},
        json={"identitetsbeteckning": orgnr},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("dokument", [])


def _fetch_ixbrl_text(doc_id: str, token: str) -> str:
    resp = requests.get(
        f"{_base_url()}/dokument/{doc_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/zip"},
        timeout=120,
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        xhtml_files = [n for n in z.namelist() if n.lower().endswith(".xhtml")]
        if not xhtml_files:
            raise ValueError(f"Ingen .xhtml i ZIP för dokument {doc_id}")
        content = z.read(xhtml_files[0]).decode("utf-8", errors="replace")
    soup = BeautifulSoup(content, "lxml")
    text = soup.get_text(separator="\n", strip=True)
    # iXBRL-dokument kan vara mycket stora — trunkera för att hålla tokenbudget
    MAX_CHARS = 80_000
    if len(text) > MAX_CHARS:
        logging.warning(f"[bolagsverket] {doc_id}: {len(text):,} tecken, trunkerar till {MAX_CHARS:,}")
        text = text[:MAX_CHARS]
    return text


def get_annual_report_texts(orgnr: str, max_years: int = 3) -> list[tuple[str, str]]:
    """
    Hämtar upp till max_years årsredovisningar för org.nr från Bolagsverket.
    Returnerar [(label, text), ...] sorterat nyast först.

    Kräver BOLAGSVERKET_CLIENT_ID och BOLAGSVERKET_CLIENT_SECRET i env.
    Kräver Python-paketen: requests, beautifulsoup4, lxml.

    Bara digitalt inlämnade årsredovisningar från 2020+ finns tillgängliga.
    """
    if not os.environ.get("BOLAGSVERKET_CLIENT_ID"):
        raise RuntimeError("BOLAGSVERKET_CLIENT_ID ej konfigurerad")

    token = _get_token()

    docs = _fetch_document_list(orgnr, token)
    if not docs:
        raise RuntimeError(
            f"Inga årsredovisningar hittades för org.nr {orgnr}. "
            "Kontrollera att bolaget lämnat digitala årsredovisningar (iXBRL) från 2020+."
        )

    docs.sort(key=lambda d: d.get("rapporteringsperiodTom", ""), reverse=True)

    result: list[tuple[str, str]] = []
    for doc in docs[:max_years]:
        doc_id   = doc.get("dokumentId", "")
        period   = doc.get("rapporteringsperiodTom", "")
        year     = period[:4] if period else "okänt år"
        label    = f"Årsredovisning {year}"
        try:
            logging.info(f"[bolagsverket] hämtar {label} (id={doc_id})")
            text = _fetch_ixbrl_text(doc_id, token)
            logging.info(f"[bolagsverket] {label}: {len(text):,} tecken")
            result.append((label, text))
        except Exception as exc:
            logging.warning(f"[bolagsverket] fel för {label} (id={doc_id}): {exc}")

    if not result:
        raise RuntimeError(f"Kunde inte ladda ner årsredovisningar för {orgnr}")

    return result
