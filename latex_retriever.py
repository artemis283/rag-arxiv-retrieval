import arxiv_to_prompt
from pathlib import Path

def download_arxiv_paper(arxiv_id):
    """
    Downloads the arXiv paper with the given ID and caches it in fixtures/latex/.
    Returns the paper content as a string.
    """
    cache_dir = Path("fixtures/latex")
    cache_dir.mkdir(parents=True, exist_ok=True)

    paper_content = arxiv_to_prompt.download_arxiv_source(arxiv_id, cache_dir=str(cache_dir))
    return paper_content