import pytest 
from docuchat.ingestion.loader import load_pdf

def test_load_pdf_returns_documents(tmp_pdf_path):
    # call load_pdf with a valid PDF
    # assert something about the result
    result = load_pdf(tmp_pdf_path)
    assert len(result) > 0


def test_load_pdf_raises_when_file_missing(tmp_path):
    # call load_pdf with a path that doesn't exist
    # assert FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path)
    