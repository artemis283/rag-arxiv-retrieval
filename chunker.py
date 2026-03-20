from sentence_transformers import SentenceTransformer
from functools import lru_cache
from pathlib import Path

EXAMPLE_LATEX_FILE = '/fixtures/latex/2603.07379v1'
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
