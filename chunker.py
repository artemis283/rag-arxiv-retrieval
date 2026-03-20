from sentence_transformers import SentenceTransformer
from functools import lru_cache
from pathlib import Path
import re

EXAMPLE_LATEX_DIR = Path('fixtures/latex/2603.01399v1')
TRANSFORMER_MODEL = 'sentence-transformers/all-mpnet-base-v2'
MODEL_CACHE_DIR = Path('fixtures/models')


@lru_cache(maxsize=1)
def get_transformer_model():
    """
    Load sentence transformer model locally and cache it.
    Downloads the model to fixtures/models/ on first use.
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(TRANSFORMER_MODEL, cache_folder=str(MODEL_CACHE_DIR))


def find_main_tex_files(paper_dir: Path) -> list[Path]:
    """Find all .tex files in a paper directory, excluding style files."""
    return sorted(
        p for p in paper_dir.glob("*.tex")
        if p.suffix == ".tex"
    )


def strip_latex_commands(text: str) -> str:
    """Remove common LaTeX commands and environments, keeping readable text."""
    # Remove comments
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

    # Remove entire environments that are never readable (tables, figures, math, algorithms)
    for env in ['table', 'figure', 'algorithm', 'align', 'equation', 'tikzpicture', 'tabular']:
        text = re.sub(rf'\\begin\{{{env}\*?}}.*?\\end\{{{env}\*?}}', '', text, flags=re.DOTALL)

    # Remove \begin{...} and \end{...} for remaining environments
    text = re.sub(r'\\(begin|end)\{[^}]*\}', '', text)

    # Remove inline math $...$ and $$...$$
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]*\$', '', text)

    # Remove label/ref/cite noise like sec:intro, fig:overview
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\{[^}]*\}', '', text)
    text = re.sub(r'\\cite[tp]?\{[^}]*\}', '', text)

    # Remove figure/table positioning args like [htbp], [t], [!]
    text = re.sub(r'\[[htbp!]+\]', '', text)

    # Keep inner text for formatting commands
    text = re.sub(r'\\(?:textbf|textit|emph|underline)\{([^}]*)\}', r'\1', text)

    # Remove \newtheorem, \def, \newcommand and similar definitions
    text = re.sub(r'\\(newtheorem|newcommand|renewcommand|def)[^}]*\{[^}]*\}', '', text)

    # Remove remaining LaTeX commands with optional args
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])*(?:\{[^}]*\})*', '', text)

    # Clean up leftover braces, backslashes, and column format specs (e.g. cccccc, lccc)
    text = re.sub(r'[{}\\]', '', text)
    text = re.sub(r'\b[lcr|]+\b', '', text)  # table column specs

    # Collapse excess whitespace and blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def strip_latex_commands(text: str) -> str:
    """Remove common LaTeX commands and environments, keeping readable text."""
    # Remove comments
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

    # Remove entire environments that are never readable (tables, figures, math, algorithms)
    for env in ['table', 'figure', 'algorithm', 'align', 'equation', 'tikzpicture', 'tabular']:
        text = re.sub(rf'\\begin\{{{env}\*?}}.*?\\end\{{{env}\*?}}', '', text, flags=re.DOTALL)

    # Remove \begin{...} and \end{...} for remaining environments
    text = re.sub(r'\\(begin|end)\{[^}]*\}', '', text)

    # Remove inline math $...$ and $$...$$
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]*\$', '', text)

    # Remove label/ref/cite noise like sec:intro, fig:overview
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\{[^}]*\}', '', text)
    text = re.sub(r'~?\\cite[tp]?\{[^}]*\}', '', text)  # also remove leading ~

    # Remove figure/table positioning args like [htbp], [t], [!]
    text = re.sub(r'\[[htbp!]+\]', '', text)

    # Keep inner text for formatting commands
    text = re.sub(r'\\(?:textbf|textit|emph|underline)\{([^}]*)\}', r'\1', text)

    # Remove \newtheorem, \def, \newcommand and similar definitions
    text = re.sub(r'\\(newtheorem|newcommand|renewcommand|def)[^}]*\{[^}]*\}', '', text)

    # Remove remaining LaTeX commands with optional args
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])*(?:\{[^}]*\})*', '', text)

    # Clean up leftover braces, backslashes, and column format specs
    text = re.sub(r'[{}\\]', '', text)
    text = re.sub(r'\b[lcr|]+\b', '', text)

    # Remove orphaned tildes (from removed citations like ~\cite{...})
    text = re.sub(r'~', ' ', text)

    # Remove leftover square bracket content like [section], [theorem]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Collapse excess whitespace and blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def chunk_by_section(paper_dir: Path) -> list[dict]:
    """
    Parse .tex files from a paper directory and chunk by LaTeX sections/subsections.
    Returns a list of dicts with 'section', 'text', and 'source' keys.
    """
    tex_files = find_main_tex_files(paper_dir)
    chunks = []

    for tex_file in tex_files:
        content = tex_file.read_text(errors='ignore')
        # Split on section-level commands (section, subsection, subsubsection)
        parts = re.split(r'(\\(?:section|subsection|subsubsection)\*?\{[^}]*\})', content)

        current_section = "preamble"
        for part in parts:
            section_match = re.match(r'\\(?:section|subsection|subsubsection)\*?\{([^}]*)\}', part)
            if section_match:
                current_section = section_match.group(1)
            else:
                cleaned = strip_latex_commands(part)
                if cleaned and len(cleaned) > 100:  # raised threshold to skip junk
                    chunks.append({
                        "section": current_section,
                        "text": cleaned,
                        "source": f"{paper_dir.name}/{tex_file.name}",
                    })

    return chunks




if __name__ == "__main__":
    chunks = chunk_by_section(EXAMPLE_LATEX_DIR)
    for c in chunks:
        print(f"\n--- [{c['source']}] {c['section']} ({len(c['text'])} chars) ---")
        print(c['text'][:200] + "...")
