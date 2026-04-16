import urllib.request
import xml.etree.ElementTree as ET
import time

ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def fetch_metadata(arxiv_id: str) -> dict:
    """Fetch title, authors, date, abstract from arXiv API."""
    # Strip version suffix (e.g. 2603.04162v2 -> 2603.04162)
    base_id = arxiv_id.split("v")[0]
    url = f"http://export.arxiv.org/api/query?id_list={base_id}"

    try:
        data = urllib.request.urlopen(url).read().decode()
        root = ET.fromstring(data)
        entry = root.find("atom:entry", ARXIV_NS)

        if entry is None:
            return {}

        title = entry.findtext("atom:title", "", ARXIV_NS).strip().replace("\n", " ")

        authors = [
            a.findtext("atom:name", "", ARXIV_NS)
            for a in entry.findall("atom:author", ARXIV_NS)
        ]

        published = entry.findtext("atom:published", "", ARXIV_NS)[:10]  # YYYY-MM-DD

        abstract = entry.findtext("atom:summary", "", ARXIV_NS).strip().replace("\n", " ")

        return {
            "title": title,
            "authors": authors,
            "published": published,
            "abstract": abstract,
        }
    except Exception as e:
        print(f"  Failed to fetch metadata for {arxiv_id}: {e}")
        return {}


def fetch_all_metadata(arxiv_ids: list[str]) -> dict:
    """Fetch metadata for multiple papers with rate limiting."""
    results = {}
    for i, arxiv_id in enumerate(arxiv_ids):
        print(f"  Fetching metadata {i+1}/{len(arxiv_ids)}: {arxiv_id}")
        results[arxiv_id] = fetch_metadata(arxiv_id)
        time.sleep(0.5)  # Be nice to arXiv API
    return results