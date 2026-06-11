# tests/test_ingestion.py
import pytest
from ingestion.parser import DocumentParser
from ingestion.chunker import TextChunker
from ingestion.parser import ParsedDocument

def test_parse_txt(sample_txt_file):
    doc = DocumentParser().parse(sample_txt_file)
    assert doc.file_type == "txt"
    assert "Revenue" in doc.text
    assert doc.metadata["filename"] == "report.txt"

def test_parse_csv(sample_csv_file):
    doc = DocumentParser().parse(sample_csv_file)
    assert doc.file_type == "csv"
    assert "revenue" in doc.text
    assert len(doc.tables) == 1
    assert doc.tables[0].shape == (3, 3)

def test_unsupported_file(tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("data")
    with pytest.raises(ValueError, match="Unsupported"):
        DocumentParser().parse(f)

def test_chunker_splits_long_text():
    doc = ParsedDocument(source="x.txt", file_type="txt", text="word " * 600)
    chunks = TextChunker(chunk_size=200, chunk_overlap=20).chunk(doc)
    assert len(chunks) > 1
    assert all(c.chunk_index == i for i, c in enumerate(chunks))

def test_chunker_preserves_source(sample_txt_file):
    doc = DocumentParser().parse(sample_txt_file)
    chunks = TextChunker().chunk(doc)
    assert all(c.source == str(sample_txt_file) for c in chunks)

def test_chunker_empty_text():
    doc = ParsedDocument(source="empty.txt", file_type="txt", text="   ")
    chunks = TextChunker().chunk(doc)
    assert chunks == []