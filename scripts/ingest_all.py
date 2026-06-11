# scripts/ingest_all.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must set env vars BEFORE any module imports config.py
import os
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from loguru import logger
from ingestion.pipeline import IngestionPipeline
from vectorstore.store import VectorStore


def main():
    from config import get_settings
    settings = get_settings()
    print(f"Key loaded: {settings.openai_api_key[:20]}...")

    pipeline = IngestionPipeline()
    store    = VectorStore()
    store.load()

    raw_dir = ROOT / "data" / "raw"
    files   = [f for f in raw_dir.iterdir()
               if f.suffix.lower() in {".pdf", ".docx", ".csv", ".txt", ".xlsx"}]

    logger.info(f"Found {len(files)} files to ingest")

    for file in sorted(files):
        try:
            chunks = pipeline.run(file)
            store.add_chunks(chunks)
            logger.info(f"  ✅ {file.name} → {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"  ⚠️  {file.name} skipped: {e}")

    store.save()
    logger.info(f"Done. Total vectors in index: {store.total}")


if __name__ == "__main__":
    main()