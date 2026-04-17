"""
Academic PDF Search & Download Engine
=====================================
Multi-source pipeline for discovering and downloading open-access academic
PDFs. Searches OpenAlex, Semantic Scholar, CrossRef, and Unpaywall.

Sources (priority order):
  1. OpenAlex        -- largest open scholarly graph, free, fast
  2. Semantic Scholar -- good OA PDF links, citation graph
  3. Unpaywall       -- DOI-to-OA PDF resolution
  4. CrossRef        -- metadata + reference chaining

Usage:
  export POLITE_EMAIL=your@email.com   # required for API access

  python pdf_search_engine.py                  # full search (all topics)
  python pdf_search_engine.py --test           # test mode (2 topics, 5 results each)
  python pdf_search_engine.py --topic "scour"  # single topic search
  python pdf_search_engine.py --query "suction caisson lateral load"  # custom query
  python pdf_search_engine.py --no-s2          # skip Semantic Scholar (rate-limited)
  python pdf_search_engine.py --download-dir ./pdfs --output-dir ./markdown
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import hashlib
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

# ── Configuration ──────────────────────────────────────────────────────────
# Paths configurable via CLI args or environment variables
DOWNLOAD_DIR = Path(os.environ.get("PDF_DOWNLOAD_DIR", Path(__file__).parent / "downloads"))
LIT_REVIEW_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", Path(__file__).parent / "markdown"))
LOG_FILE = DOWNLOAD_DIR / "download_log.json"
DEDUP_FILE = DOWNLOAD_DIR / "known_papers.json"

# Email for polite API pools (required by OpenAlex, Unpaywall, CrossRef)
EMAIL = os.environ.get("POLITE_EMAIL", "")
MAX_RESULTS_PER_QUERY = 50
MIN_YEAR = 2018
REQUEST_DELAY = 1.0
SKIP_S2 = False

log = logging.getLogger(__name__)

# ── Research Topics ────────────────────────────────────────────────────────
# Each topic has: name, queries (sent to APIs), and optional filters
RESEARCH_TOPICS = [
    {
        "name": "suction_bucket_foundation",
        "queries": [
            "suction bucket foundation offshore wind",
            "suction caisson offshore wind turbine",
            "suction anchor capacity sand clay",
            "tripod suction bucket foundation cyclic loading",
            "suction pile installation pressure",
        ],
    },
    {
        "name": "scour_offshore_wind",
        "queries": [
            "scour offshore wind turbine foundation",
            "scour detection natural frequency vibration",
            "scour monitoring bridge foundation",
            "scour depth prediction offshore monopile",
            "local scour protection offshore structure",
            "scour effect dynamic response wind turbine",
        ],
    },
    {
        "name": "structural_health_monitoring",
        "queries": [
            "structural health monitoring offshore wind turbine",
            "vibration based damage detection wind turbine",
            "operational modal analysis offshore structure",
            "natural frequency monitoring environmental effects",
            "SHM wind turbine foundation scour",
            "ambient vibration monitoring offshore",
        ],
    },
    {
        "name": "soil_structure_interaction",
        "queries": [
            "soil structure interaction offshore wind turbine",
            "p-y curve offshore pile lateral loading",
            "dynamic impedance foundation frequency",
            "Winkler spring model offshore foundation",
            "SSI natural frequency wind turbine",
        ],
    },
    {
        "name": "geotechnical_reliability",
        "queries": [
            "reliability analysis geotechnical offshore foundation",
            "probabilistic bearing capacity shallow foundation",
            "Monte Carlo simulation geotechnical uncertainty",
            "LRFD offshore foundation design",
            "spatial variability soil properties foundation",
        ],
    },
    {
        "name": "centrifuge_modelling",
        "queries": [
            "centrifuge modelling offshore foundation",
            "centrifuge test suction caisson bucket",
            "physical modelling similitude offshore wind",
            "1g model test offshore foundation",
        ],
    },
    {
        "name": "dimensional_analysis_geotechnics",
        "queries": [
            "Buckingham Pi theorem geotechnical",
            "dimensional analysis foundation engineering",
            "similitude scaling laws offshore wind model",
            "dimensionless groups soil mechanics",
        ],
    },
    {
        "name": "numerical_modelling_foundation",
        "queries": [
            "finite element limit analysis foundation",
            "3D bearing capacity suction caisson",
            "OptumG2 geotechnical numerical",
            "OpenSeesPy offshore wind turbine",
            "coupled aero-hydro-servo-elastic simulation wind turbine",
            "OpenFAST offshore wind simulation",
        ],
    },
    {
        "name": "machine_learning_geotechnics",
        "queries": [
            "machine learning geotechnical engineering prediction",
            "deep learning soil property prediction",
            "autoencoder geotechnical data",
            "transfer learning geotechnical cross-site",
            "neural network foundation bearing capacity",
        ],
    },
    {
        "name": "offshore_wind_design_standards",
        "queries": [
            "offshore wind turbine foundation design review",
            "DNV offshore wind standard foundation",
            "IEC 61400 offshore wind turbine",
            "lifetime assessment offshore wind structure",
        ],
    },
]

# ── Deduplication ──────────────────────────────────────────────────────────

def load_known_papers():
    """Load set of known paper identifiers (DOIs + title hashes)."""
    if DEDUP_FILE.exists():
        return set(json.loads(DEDUP_FILE.read_text(encoding="utf-8")))
    # Build from existing literature_review filenames
    known = set()
    if LIT_REVIEW_DIR.exists():
        for f in LIT_REVIEW_DIR.iterdir():
            if f.suffix == ".md":
                # Hash the filename as a rough dedup key
                known.add(hashlib.md5(f.stem.lower().encode()).hexdigest()[:16])
    return known


def save_known_papers(known):
    DEDUP_FILE.write_text(json.dumps(sorted(known), indent=2), encoding="utf-8")


def title_hash(title):
    """Normalize and hash a title for dedup."""
    clean = re.sub(r"[^a-z0-9 ]", "", title.lower().strip())
    clean = re.sub(r"\s+", " ", clean)
    return hashlib.md5(clean.encode()).hexdigest()[:16]


def is_known(doi, title, known_set):
    if doi and doi.lower() in known_set:
        return True
    if title and title_hash(title) in known_set:
        return True
    return False


def mark_known(doi, title, known_set):
    if doi:
        known_set.add(doi.lower())
    if title:
        known_set.add(title_hash(title))


# ── Relevance Filter ──────────────────────────────────────────────────────

# Core domain keywords — at least one must appear in the title
DOMAIN_KEYWORDS = {
    "offshore", "wind", "turbine", "foundation", "suction", "bucket", "caisson",
    "pile", "monopile", "tripod", "jacket", "scour", "erosion", "sediment",
    "geotechnical", "soil", "clay", "sand", "bearing capacity", "lateral",
    "cyclic", "monotonic", "vibration", "frequency", "modal", "SHM",
    "monitoring", "structural health", "damage detection", "centrifuge",
    "model test", "similitude", "dimensional analysis", "buckingham",
    "reliability", "probabilistic", "monte carlo", "LRFD", "uncertainty",
    "finite element", "FEM", "limit analysis", "numerical", "OpenSees",
    "OptumG", "OpenFAST", "p-y curve", "Winkler", "impedance",
    "machine learning", "deep learning", "neural network", "autoencoder",
    "transfer learning", "encoder", "prediction",
    "anchor", "mooring", "floating", "spar", "semi-submersible",
    "fatigue", "dynamic", "eigenvalue", "natural frequency",
    "CPT", "cone penetration", "shear strength", "undrained",
    "wave", "current", "hydrodynamic", "load", "capacity",
}


def is_relevant(title):
    """Check if a paper title contains at least one domain keyword."""
    if not title:
        return False
    title_lower = title.lower()
    return any(kw in title_lower for kw in DOMAIN_KEYWORDS)


# ── Download Log ───────────────────────────────────────────────────────────

def load_log():
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text(encoding="utf-8"))
    return []


def save_log(entries):
    LOG_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


# ── API Clients ────────────────────────────────────────────────────────────

class OpenAlexClient:
    """Search OpenAlex for papers with open-access PDFs."""
    BASE = "https://api.openalex.org/works"

    def search(self, query, max_results=MAX_RESULTS_PER_QUERY, min_year=MIN_YEAR):
        params = {
            "search": query,
            "filter": f"from_publication_date:{min_year}-01-01,has_oa_accepted_or_published_version:true",
            "select": "id,doi,title,publication_year,open_access,authorships,primary_location,cited_by_count",
            "sort": "relevance_score:desc",
            "per_page": min(max_results, 50),
            "mailto": EMAIL,
        }
        try:
            resp = requests.get(self.BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for work in data.get("results", []):
                oa = work.get("open_access", {})
                pdf_url = oa.get("oa_url")
                doi = work.get("doi", "")
                if doi and doi.startswith("https://doi.org/"):
                    doi = doi.replace("https://doi.org/", "")
                title = work.get("title", "")
                authors = []
                for a in work.get("authorships", [])[:3]:
                    name = a.get("author", {}).get("display_name", "")
                    if name:
                        authors.append(name)
                results.append({
                    "title": title,
                    "doi": doi,
                    "year": work.get("publication_year"),
                    "pdf_url": pdf_url,
                    "cited_by": work.get("cited_by_count", 0),
                    "authors": authors,
                    "source": "openalex",
                })
            return results
        except Exception as e:
            log.warning(f"OpenAlex search failed for '{query}': {e}")
            return []


class SemanticScholarClient:
    """Search Semantic Scholar for papers with open-access PDFs."""
    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, query, max_results=MAX_RESULTS_PER_QUERY, min_year=MIN_YEAR):
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,authors,year,externalIds,openAccessPdf,citationCount",
            "year": f"{min_year}-",
            "openAccessPdf": "",
        }
        try:
            # Exponential backoff for rate limits
            for attempt in range(3):
                resp = requests.get(self.BASE, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 10 * (2 ** attempt)
                    log.info(f"  S2 rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            else:
                log.warning(f"S2 exhausted retries for '{query}'")
                return []
            data = resp.json()
            results = []
            for paper in data.get("data", []):
                oa_pdf = paper.get("openAccessPdf")
                pdf_url = oa_pdf.get("url") if oa_pdf else None
                ext_ids = paper.get("externalIds", {})
                doi = ext_ids.get("DOI", "")
                authors = [a.get("name", "") for a in paper.get("authors", [])[:3]]
                results.append({
                    "title": paper.get("title", ""),
                    "doi": doi,
                    "year": paper.get("year"),
                    "pdf_url": pdf_url,
                    "cited_by": paper.get("citationCount", 0),
                    "authors": authors,
                    "source": "semantic_scholar",
                })
            return results
        except Exception as e:
            log.warning(f"Semantic Scholar search failed for '{query}': {e}")
            return []


class UnpaywallClient:
    """Resolve DOI to open-access PDF via Unpaywall."""

    def get_pdf_url(self, doi):
        if not doi:
            return None
        url = f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                best = data.get("best_oa_location")
                if best:
                    return best.get("url_for_pdf") or best.get("url")
            return None
        except Exception:
            return None


class CrossRefClient:
    """Search CrossRef for metadata and DOIs."""
    BASE = "https://api.crossref.org/works"

    def search(self, query, max_results=20, min_year=MIN_YEAR):
        params = {
            "query": query,
            "rows": max_results,
            "filter": f"from-pub-date:{min_year}",
            "sort": "relevance",
            "select": "DOI,title,author,published-print,is-referenced-by-count",
            "mailto": EMAIL,
        }
        try:
            resp = requests.get(self.BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("message", {}).get("items", []):
                doi = item.get("DOI", "")
                titles = item.get("title", [])
                title = titles[0] if titles else ""
                pub_date = item.get("published-print", {})
                parts = pub_date.get("date-parts", [[None]])[0]
                year = parts[0] if parts else None
                authors = []
                for a in item.get("author", [])[:3]:
                    name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                    if name:
                        authors.append(name)
                results.append({
                    "title": title,
                    "doi": doi,
                    "year": year,
                    "pdf_url": None,  # CrossRef doesn't provide PDFs directly
                    "cited_by": item.get("is-referenced-by-count", 0),
                    "authors": authors,
                    "source": "crossref",
                })
            return results
        except Exception as e:
            log.warning(f"CrossRef search failed for '{query}': {e}")
            return []


# ── PDF Downloader ─────────────────────────────────────────────────────────

def sanitize_filename(title, doi, year, authors):
    """Create a clean filename from paper metadata."""
    # Format: (Year Author1) Title.pdf
    author = authors[0].split()[-1] if authors else "Unknown"
    # Clean title
    clean_title = re.sub(r'[<>:"/\\|?*]', '', title or "untitled")
    clean_title = clean_title[:120]  # limit length
    year_str = str(year) if year else "XXXX"
    return f"({year_str} {author}) {clean_title}.pdf"


def download_pdf(url, filepath, timeout=60):
    """Download a PDF file from URL. Returns True on success."""
    try:
        headers = {
            "User-Agent": f"Mozilla/5.0 (academic-research; mailto:{EMAIL})" if EMAIL else "Mozilla/5.0",
            "Accept": "application/pdf,*/*",
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")

        # Stream download — check magic bytes from the first chunk
        with open(filepath, "wb") as f:
            first_chunk = True
            for chunk in resp.iter_content(chunk_size=8192):
                if first_chunk:
                    first_chunk = False
                    is_pdf_type = "pdf" in content_type.lower() or "octet-stream" in content_type.lower()
                    if not is_pdf_type and not chunk.startswith(b"%PDF"):
                        log.debug(f"Not a PDF: {url} (type: {content_type})")
                        return False
                f.write(chunk)

        # Validate: must be >10KB and start with %PDF
        if filepath.stat().st_size < 10000:
            filepath.unlink(missing_ok=True)
            return False
        with open(filepath, "rb") as f:
            if not f.read(5).startswith(b"%PDF"):
                filepath.unlink(missing_ok=True)
                return False
        return True
    except Exception as e:
        log.debug(f"Download failed {url}: {e}")
        if filepath.exists():
            filepath.unlink(missing_ok=True)
        return False


# ── Main Pipeline ──────────────────────────────────────────────────────────

def search_topic(topic, clients, known_set, max_results=MAX_RESULTS_PER_QUERY):
    """Search all APIs for a topic, merge and deduplicate results."""
    openalex, semantic, unpaywall, crossref = clients
    all_papers = {}  # doi/title_hash -> paper dict

    for query in topic["queries"]:
        log.info(f"  Searching: '{query}'")

        # OpenAlex (primary — best for OA papers)
        results = openalex.search(query, max_results=max_results)
        time.sleep(REQUEST_DELAY)

        # Semantic Scholar (secondary — good citation data)
        if not SKIP_S2:
            results += semantic.search(query, max_results=max_results)
            time.sleep(REQUEST_DELAY * 3)  # S2 is more rate-limited

        # CrossRef (tertiary — for DOI discovery)
        results += crossref.search(query, max_results=20)
        time.sleep(REQUEST_DELAY)

        for paper in results:
            doi = paper.get("doi", "")
            title = paper.get("title", "")
            if not title:
                continue
            if is_known(doi, title, known_set):
                continue

            key = doi.lower() if doi else title_hash(title)
            if key in all_papers:
                # Merge: keep the one with a PDF URL
                existing = all_papers[key]
                if not existing["pdf_url"] and paper["pdf_url"]:
                    existing["pdf_url"] = paper["pdf_url"]
                if paper.get("cited_by", 0) > existing.get("cited_by", 0):
                    existing["cited_by"] = paper["cited_by"]
            else:
                all_papers[key] = paper

    # For papers without PDF URLs, try Unpaywall
    no_pdf = [p for p in all_papers.values() if not p["pdf_url"] and p["doi"]]
    log.info(f"  Trying Unpaywall for {len(no_pdf)} papers without PDF links...")
    for paper in no_pdf:
        url = unpaywall.get_pdf_url(paper["doi"])
        if url:
            paper["pdf_url"] = url
        time.sleep(REQUEST_DELAY * 0.5)

    # Sort by citation count (most cited first)
    papers = sorted(all_papers.values(), key=lambda p: p.get("cited_by", 0), reverse=True)
    return papers


def run_pipeline(topics=None, test_mode=False):
    """Main execution: search topics, download PDFs."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    known_set = load_known_papers()
    download_log = load_log()

    clients = (
        OpenAlexClient(),
        SemanticScholarClient(),
        UnpaywallClient(),
        CrossRefClient(),
    )

    if topics is None:
        topics = RESEARCH_TOPICS

    if test_mode:
        topics = topics[:2]
        for t in topics:
            t["queries"] = t["queries"][:2]

    total_downloaded = 0
    total_found = 0
    total_skipped_known = 0

    summary = {}

    for topic in topics:
        topic_name = topic["name"]
        log.info(f"\n{'='*60}")
        log.info(f"TOPIC: {topic_name}")
        log.info(f"{'='*60}")

        papers = search_topic(topic, clients, known_set,
                              max_results=10 if test_mode else MAX_RESULTS_PER_QUERY)

        downloadable = [p for p in papers if p["pdf_url"]]
        log.info(f"  Found {len(papers)} unique papers, {len(downloadable)} with PDF links")
        total_found += len(papers)

        topic_downloaded = 0
        for paper in downloadable:
            title = paper["title"]
            doi = paper["doi"]

            if is_known(doi, title, known_set):
                total_skipped_known += 1
                continue

            if not is_relevant(title):
                log.debug(f"  Skipped (irrelevant): {title[:80]}")
                continue

            filename = sanitize_filename(title, doi, paper["year"], paper["authors"])
            filepath = DOWNLOAD_DIR / filename

            if filepath.exists():
                mark_known(doi, title, known_set)
                continue

            log.info(f"  Downloading: {title[:80]}...")
            success = download_pdf(paper["pdf_url"], filepath)

            if success:
                topic_downloaded += 1
                total_downloaded += 1
                mark_known(doi, title, known_set)
                download_log.append({
                    "title": title,
                    "doi": doi,
                    "year": paper["year"],
                    "authors": paper["authors"],
                    "pdf_url": paper["pdf_url"],
                    "filename": filename,
                    "source": paper["source"],
                    "cited_by": paper.get("cited_by", 0),
                    "topic": topic_name,
                    "downloaded_at": datetime.now().isoformat(),
                })
                log.info(f"    ✓ Saved: {filename}")
            else:
                log.debug(f"    ✗ Failed: {paper['pdf_url']}")

            time.sleep(REQUEST_DELAY)

        summary[topic_name] = {
            "found": len(papers),
            "downloadable": len(downloadable),
            "downloaded": topic_downloaded,
        }
        log.info(f"  Topic complete: {topic_downloaded} new PDFs downloaded")

    # Save state
    save_known_papers(known_set)
    save_log(download_log)

    # Print summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    for topic_name, stats in summary.items():
        log.info(f"  {topic_name}: {stats['downloaded']}/{stats['downloadable']} downloaded "
                 f"({stats['found']} found)")
    log.info(f"\n  TOTAL: {total_downloaded} new PDFs downloaded")
    log.info(f"  Total unique papers found: {total_found}")
    log.info(f"  Skipped (already known): {total_skipped_known}")
    log.info(f"  Download directory: {DOWNLOAD_DIR}")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic PDF Search & Download Engine")
    parser.add_argument("--test", action="store_true", help="Test mode (2 topics, 5 results)")
    parser.add_argument("--topic", type=str, help="Search a single topic by name")
    parser.add_argument("--query", type=str, help="Run a custom search query")
    parser.add_argument("--min-year", type=int, default=MIN_YEAR, help=f"Minimum year (default: {MIN_YEAR})")
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS_PER_QUERY)
    parser.add_argument("--no-s2", action="store_true", help="Skip Semantic Scholar (rate-limited)")
    parser.add_argument("--download-dir", type=str, help="PDF download directory")
    parser.add_argument("--output-dir", type=str, help="Markdown output directory")
    args = parser.parse_args()

    # Apply CLI overrides
    global MIN_YEAR, MAX_RESULTS_PER_QUERY, SKIP_S2, DOWNLOAD_DIR, LIT_REVIEW_DIR, LOG_FILE, DEDUP_FILE
    MIN_YEAR = args.min_year
    MAX_RESULTS_PER_QUERY = args.max_results
    SKIP_S2 = args.no_s2
    if args.download_dir:
        DOWNLOAD_DIR = Path(args.download_dir)
    if args.output_dir:
        LIT_REVIEW_DIR = Path(args.output_dir)
    LOG_FILE = DOWNLOAD_DIR / "download_log.json"
    DEDUP_FILE = DOWNLOAD_DIR / "known_papers.json"

    if not EMAIL:
        print("ERROR: Set POLITE_EMAIL environment variable first.", file=sys.stderr)
        sys.exit(1)

    # Set up logging (after paths are resolved)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(DOWNLOAD_DIR / "search_engine.log", encoding="utf-8"),
        ],
    )

    if args.query:
        # Custom single query
        topics = [{"name": "custom", "queries": [args.query]}]
        run_pipeline(topics=topics)
    elif args.topic:
        # Single topic
        matching = [t for t in RESEARCH_TOPICS if args.topic.lower() in t["name"].lower()]
        if matching:
            run_pipeline(topics=matching)
        else:
            log.error(f"Unknown topic: {args.topic}")
            log.info(f"Available: {[t['name'] for t in RESEARCH_TOPICS]}")
    else:
        run_pipeline(test_mode=args.test)
