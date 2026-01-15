from pathlib import Path
from pypdf import PdfReader
from src.utils.logger import Logger

logger = Logger(__name__)


def get_chunks(
        dir_path,
        chunk_size,
        overlap_size,
        tokenize,
        specific_files=None
        ):
    dir_p = Path(dir_path)
    if specific_files:
        pdf_files = []
        for f_name in specific_files:
            f_path = dir_p / Path(f_name).name
            if f_path.exists():
                pdf_files.append(f_path)
            else:
                logger.log(f"File requested but not found: {f_path}", "warning")
    else:
        pdf_files = list(dir_p.glob("*.pdf"))
    if not pdf_files:
        logger.log(f"No PDFs found in {dir_path}", "warning")
        return
    logger.log(f"Found {len(pdf_files)} PDF files in {dir_path}", "info")
    for pdf_file in pdf_files:
        yield f"<s>{pdf_file}</s>"
        try:
            yield from _stream_pdf_chunks(
                str(pdf_file),
                chunk_size,
                overlap_size,
                tokenize,
            )
        except Exception as e:
            logger.log(f"Failed to read {pdf_file.name}: {e}", "error")
            yield f"<e>{pdf_file}</e>"
            continue
        yield f"<e>{pdf_file}</e>"


def _stream_pdf_chunks(pdf_path, chunk_size, overlap_size, tokenize):
    step_size = chunk_size - overlap_size
    if step_size <= 0:
        msg = "Overlap size must be smaller than chunk size."
        raise ValueError(msg)
    logger.log(f"Processing PDF: {pdf_path}", "debug")
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
        logger.log(
            f"Processed {pages_processed} pages from {pdf_path}", "debug"
        )
    except Exception as e:
        logger.log(f"Error processing PDF {pdf_path}: {e}", "error")
        raise
