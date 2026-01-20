from typing import List, Tuple
from src.utils.logger import Logger

logger = Logger(__name__)

# Maximum characters for summarization to avoid context overflow
# Approximately 4 chars per token, with buffer for prompt
MAX_SUMMARY_CHARS = 12000


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "What is the deductible?"
    },
    {
        "role": "assistant",
        "content": "The deductible is $500 per occurrence for collision coverage."
    },
    {
        "role": "user",
        "content": "List the safety requirements."
    },
    {
        "role": "assistant",
        "content": "The safety requirements are:\n* Safety glasses must be worn\n* Steel-toed boots are mandatory"
    },
    {
        "role": "user",
        "content": "How do I cook pasta?"
    },
    {
        "role": "assistant",
        "content": "I cannot answer this question based on the provided documents."
    }
]


def get_few_shot_messages() -> List[dict]:
    return FEW_SHOT_EXAMPLES.copy()


def prepare_prompt(documents: List[Tuple[int, str, str]], query: str) -> str:
    prompt_parts = [
        "Use the following documents to answer the question:",
        "----------------------------------------------"
    ]
    for doc_id, content, filename in documents:
        prompt_parts.append(format_document(doc_id, content, filename))
    prompt_parts.append("----------------------------------------------")

    prompt_parts.append("Instructions:")
    prompt_parts.append("1. Answer directly and professionally.")
    prompt_parts.append("2. Cite the source document ID using the format ``.")
    prompt_parts.append(
        "3. Do NOT mention 'ChunkID' or 'Chunk index' in the answer text.")

    prompt_parts.append(f"\nQuery: {query}")
    return "\n".join(prompt_parts)


def prepare_messages_with_few_shot(
    system_prompt: str,
    documents: List[Tuple[int, str, str]],
    query: str,
    conversation_history: List[dict] = None
) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]

    messages.extend(get_few_shot_messages())

    if conversation_history:
        messages.extend(conversation_history)

    user_prompt = prepare_prompt(documents, query)
    messages.append({"role": "user", "content": user_prompt})

    return messages


def format_document(doc_id: int, content: str, filename: str) -> str:
    return (
        f"ChunkID: {doc_id}\n"
        f"Source: {filename}\n"
        f"Content: {content}\n"
    )


def prepare_summary_prompt(
    chunks: List[str],
    filename: str,
    max_chars: int = MAX_SUMMARY_CHARS
) -> str:
    # Concatenate chunks with separators
    combined_text = ""
    chunks_used = 0
    truncated = False

    for chunk in chunks:
        # Check if adding this chunk would exceed the limit
        if len(combined_text) + len(chunk) + 2 > max_chars:
            truncated = True
            # Add as much of the chunk as we can fit
            remaining = max_chars - len(combined_text) - 50
            if remaining > 100:
                combined_text += chunk[:remaining] + "..."
            break

        if combined_text:
            combined_text += "\n\n"
        combined_text += chunk
        chunks_used += 1

    if truncated:
        logger.log(
            f"Document '{filename}' truncated: used {chunks_used}/"
            f"{len(chunks)} chunks ({len(combined_text)}/{max_chars} chars)",
            "WARNING"
        )

    prompt_parts = [
        f"Please provide a comprehensive summary of the document: "
        f"'{filename}'",
        "=" * 60,
        "DOCUMENT CONTENT:",
        "=" * 60,
        combined_text,
        "=" * 60,
        "\nProvide a well-structured summary covering the key points, "
        "findings, and conclusions."
    ]

    if truncated:
        prompt_parts.append(
            f"\n(Note: Partial document - showing {chunks_used} of "
            f"{len(chunks)} sections)"
        )

    return "\n".join(prompt_parts)


def prepare_summary_messages(
    system_prompt: str,
    chunks: List[str],
    filename: str,
    max_chars: int = MAX_SUMMARY_CHARS
) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]

    user_prompt = prepare_summary_prompt(chunks, filename, max_chars)
    messages.append({"role": "user", "content": user_prompt})

    return messages
