# ingestion/pipeline.py
from pathlib import Path
from loguru import logger
from .parser import DocumentParser
from .chunker import TextChunker, Chunk


class IngestionPipeline:
    """End-to-end pipeline: file → ParsedDocument → Chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.parser  = DocumentParser()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def run(self, file_path: str | Path) -> list[Chunk]:
        """Parse and chunk a single file. Returns list of Chunk objects."""
        doc    = self.parser.parse(file_path)
        chunks = self.chunker.chunk(doc)
        logger.info(f"Ingested {Path(file_path).name}: {len(chunks)} chunks")
        return chunks

    def run_directory(self, dir_path: str | Path) -> list[Chunk]:
        """Process all supported files in a directory."""
        dir_path   = Path(dir_path)
        all_chunks = []
        for file in dir_path.iterdir():
            if file.suffix.lower() in DocumentParser.SUPPORTED:
                try:
                    all_chunks.extend(self.run(file))
                except Exception as e:
                    logger.warning(f"Skipping {file.name}: {e}")
        logger.info(f"Total chunks from directory: {len(all_chunks)}")
        return all_chunks