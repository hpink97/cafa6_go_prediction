"""Thin, rate-limited client for the Ensembl REST API endpoints used for promoter extraction."""

from __future__ import annotations

import logging
import threading
import time

import requests

LOGGER = logging.getLogger("cafa6.promoters.ensembl")
BASE_URL = "https://rest.ensembl.org"
# Keep xref resolution below the nominal service limit; concurrent xref requests
# otherwise produce sustained 5xx responses even when the aggregate rate is legal.
MAX_REQUESTS_PER_SECOND = 5.0


class RateLimiter:
    """Fixed-interval limiter shared across threads."""

    def __init__(self, max_per_second: float):
        self._min_interval = 1.0 / max_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            remaining = self._min_interval - (time.monotonic() - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()


_RATE_LIMITER = RateLimiter(MAX_REQUESTS_PER_SECOND)


def _request(method: str, path: str, session: requests.Session, max_retries: int = 8, timeout: float = 30, **kwargs) -> requests.Response:
    last_response = None
    for attempt in range(max_retries):
        _RATE_LIMITER.wait()
        try:
            response = session.request(method, f"{BASE_URL}{path}", timeout=timeout, **kwargs)
        except requests.exceptions.RequestException as exc:
            # Network-level failures (timeouts, connection resets) raise instead of returning a Response,
            # so they need their own retry path alongside the HTTP-status-based one below.
            backoff = min(60.0, 3.0 * (2**attempt))
            LOGGER.warning("network error on %s (attempt %d/%d): %s; sleeping %.0fs", path, attempt + 1, max_retries, exc, backoff)
            time.sleep(backoff)
            continue
        last_response = response
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 1.0))
            LOGGER.warning("rate limited on %s, sleeping %.1fs", path, retry_after)
            time.sleep(retry_after)
            continue
        if response.status_code >= 500:
            # Longer, capped holdoff: 3, 6, 12, 24, 48, 60, 60, 60s -- gives a sustained outage room to clear
            # instead of giving up on the ID after a few seconds.
            backoff = min(60.0, 3.0 * (2**attempt))
            LOGGER.warning("server error %d on %s (attempt %d/%d), sleeping %.0fs", response.status_code, path, attempt + 1, max_retries, backoff)
            time.sleep(backoff)
            continue
        return response
    return last_response



def resolve_gene_id(protein_id: str, species: str, session: requests.Session) -> str | None:
    """Resolve a UniProt accession to an Ensembl gene ID via xrefs/symbol."""
    response = _request("GET", f"/xrefs/symbol/{species}/{protein_id}", session, params={"content-type": "application/json"})
    if response is None or response.status_code != 200:
        return None
    for entry in response.json():
        if entry.get("type") == "gene":
            return entry["id"]
    return None


def fetch_gene_coordinates(gene_ids: list[str], species: str, session: requests.Session) -> dict[str, dict]:
    """Bulk-fetch gene coordinates (start, end, strand, seq_region_name) via POST lookup/id."""
    coordinates: dict[str, dict] = {}
    batch_size = 300  # smaller batches are less likely to trip Ensembl's 500/timeout under load
    for start in range(0, len(gene_ids), batch_size):
        batch = gene_ids[start : start + batch_size]
        response = _request(
            "POST",
            "/lookup/id",
            session,
            params={"content-type": "application/json", "species": species},
            json={"ids": batch},
            timeout=60,
        )
        if response is None or response.status_code != 200:
            LOGGER.warning("lookup/id batch failed for %s (status=%s)", species, getattr(response, "status_code", None))
            continue
        for gene_id, record in response.json().items():
            if record:
                coordinates[gene_id] = record
    return coordinates


def fetch_all_genes(species: str, session: requests.Session) -> dict[str, dict]:
    """Fetch every protein-coding gene's coordinates genome-wide, for small genomes (e.g. bacteria).

    Used for operon-leader resolution, which needs every gene's neighbors, not just the
    genes belonging to proteins we happen to be extracting promoters for.
    """
    response = _request("GET", f"/info/assembly/{species}", session, params={"content-type": "application/json"})
    if response is None or response.status_code != 200:
        raise RuntimeError(f"Could not fetch assembly info for {species}")
    regions = response.json().get("top_level_region", [])

    genes: dict[str, dict] = {}
    max_slice_length = 5_000_000  # Ensembl REST hard cap on overlap/region span
    for region in regions:
        # Complete bacterial/archaeal replicons are labeled "chromosome" in some assemblies
        # (e.g. E. coli, B. subtilis) and "primary_assembly" in others (e.g. P. aeruginosa).
        if region.get("coord_system") not in ("chromosome", "primary_assembly"):
            continue
        seq_region_name = region["name"]
        length = region["length"]
        for start in range(1, length + 1, max_slice_length):
            end = min(start + max_slice_length - 1, length)
            response = _request(
                "GET",
                f"/overlap/region/{species}/{seq_region_name}:{start}-{end}",
                session,
                params={"feature": "gene", "content-type": "application/json"},
                timeout=120,
            )
            if response is None or response.status_code != 200:
                LOGGER.warning(
                    "overlap/region failed for %s:%s:%d-%d (status=%s)",
                    species,
                    seq_region_name,
                    start,
                    end,
                    getattr(response, "status_code", None),
                )
                continue
            for record in response.json():
                if record.get("biotype") == "protein_coding":
                    genes[record["id"]] = record
    return genes


def fetch_promoter_sequences(regions: list[str], species: str, session: requests.Session) -> dict[str, str]:
    """Bulk-fetch genomic sequences for a list of 'seq_region:start-end:strand' region strings."""
    sequences: dict[str, str] = {}
    batch_size = 20  # smaller batches are less likely to trip Ensembl's 500/timeout under load
    for start in range(0, len(regions), batch_size):
        batch = regions[start : start + batch_size]
        response = _request(
            "POST",
            f"/sequence/region/{species}",
            session,
            params={"content-type": "application/json"},
            json={"regions": batch},
            timeout=60,
        )
        if response is None or response.status_code != 200:
            LOGGER.warning("sequence/region batch failed for %s (status=%s)", species, getattr(response, "status_code", None))
            continue
        for record in response.json():
            if "seq" in record:
                sequences[record["query"]] = record["seq"]
    return sequences
