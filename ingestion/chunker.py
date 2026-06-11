# ingestion/chunker.py
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .parser import ParsedDocument

@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        raw = self.splitter.split_text(doc.text)
        return [
            Chunk(text=t, source=doc.source, chunk_index=i,
                  metadata={**doc.metadata, "file_type": doc.file_type})
            for i, t in enumerate(raw) if t.strip()
        ]