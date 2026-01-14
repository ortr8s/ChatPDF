from pathlib import Path
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def stream_pdf_chunks(pdf_path, chunk_size, overlap_size, tokenize):
    step_size = chunk_size - overlap_size

    if step_size <= 0:
        msg = "Overlap size must be smaller than chunk size."
        raise ValueError(msg)

    logger.debug(f"Processing PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        pages_processed = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += "\n" + text
            pages_processed += 1

        paragraphs = full_text.split("\n")
        text_buffer = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if text_buffer:
                text_buffer += " " + para
            else:
                text_buffer = para

            est_tokens = len(tokenize(text_buffer))

            if est_tokens >= chunk_size:
                yield text_buffer
                text_buffer = ""

        if text_buffer.strip():
            yield text_buffer

        logger.debug(
            f"Processed {pages_processed} pages from {pdf_path}"
        )
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        raise


def get_chunks(dir_path, chunk_size, overlap_size, tokenize):
    pdf_files = list(Path(dir_path).glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs found in {dir_path}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files in {dir_path}")

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
            logger.error(f"Failed to read {pdf_file.name}: {e}")
            yield f"<e>{pdf_file}</e>"
            continue
        yield f"<e>{pdf_file}</e>"
