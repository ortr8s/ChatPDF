from typing import List, Tuple


def prepare_prompt(documents: List[Tuple[int, str, str]], query: str) -> str:
    prompt_parts = [
        "Numbered document list is defined below:",
        "----------------------------------------------"
    ]
    for doc_id, content, filename in documents:
        prompt_parts.append(format_document(doc_id, content, filename))
    prompt_parts.append("----------------------------------------------")
    prompt_parts.append(f"Query: {query}")
    return "\n".join(prompt_parts)


def format_document(doc_id: int, content: str, filename: str) -> str:
    return (
        f"Document ID: {doc_id}\n"
        f"Source: {filename}\n"
        f"Content: {content}\n"
    )
