from pathlib import Path
from pypdf import PdfReader


def stream_pdf_chunks(pdf_path, chunk_size, overlap_size, tokenize):
    step_size = chunk_size - overlap_size

    if step_size <= 0:
        raise ValueError("Overlap size must be smaller than chunk size.")

    reader = PdfReader(pdf_path)
    token_buffer = []

    for page in reader.pages:
        text = page.extract_text() or ""

        page_tokens = tokenize(text)

        token_buffer.extend(page_tokens)

        while len(token_buffer) >= chunk_size:
            yield token_buffer[:chunk_size]
            token_buffer = token_buffer[step_size:]

    if token_buffer:
        yield token_buffer


def get_chunks(dir_path, chunk_size, overlap_size, tokenize):
    pdf_files = list(Path(dir_path).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {dir_path}")
        return

    for pdf_file in pdf_files:
        yield f"<s>{pdf_file}</s>"
        try:
            yield from stream_pdf_chunks(
                str(pdf_file),
                chunk_size,
                overlap_size,
                tokenize,
            )
        except Exception as e:
            print(f"Failed to read {pdf_file.name}: {e}")
            yield f"<e>{pdf_file}</e>"
            continue
        yield f"<e>{pdf_file}</e>"
