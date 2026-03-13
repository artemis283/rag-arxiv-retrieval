import zipfile
from pathlib import Path


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
    extract_and_list_papers()


if __name__ == "__main__":
    main()
