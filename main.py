import zipfile
from pathlib import Path
from latex_retriever import download_arxiv_paper


def extract_and_list_papers(zip_path: str = "rag_dataset.zip", output_dir: str = "fixtures") -> list[str]:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(f"Could not find {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
        print(f"Extracted {zip_path} to {output_dir}/")

    pdfs = sorted(p for p in output_dir.rglob("*.pdf") if not any(part.startswith("_") for part in p.parts))

    print(f"\nFound {len(pdfs)} papers:")
    for pdf in pdfs:
        print(f"  {pdf.stem}")

    return [pdf.stem for pdf in pdfs]


def main():
    paper_ids = extract_and_list_papers()

    print(f"\nDownloading latex files for {len(paper_ids)} papers...")
    for arxiv_id in paper_ids:
        try:
            download_arxiv_paper(arxiv_id)
            print(f"  ✓ Downloaded {arxiv_id}")
        except Exception as e:
            print(f"  ✗ Failed to download {arxiv_id}: {e}")


if __name__ == "__main__":
    main()

