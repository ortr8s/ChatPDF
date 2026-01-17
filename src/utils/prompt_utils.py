from typing import List, Tuple


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
    """
    Return the few-shot example messages for consistent answer formatting.

    These examples are prepended to the conversation to demonstrate:
    - Professional tone
    - Concise answers with bullet points where appropriate
    - Proper citation format using document references

    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    return FEW_SHOT_EXAMPLES.copy()


def prepare_prompt(documents: List[Tuple[int, str, str]], query: str) -> str:
    """
    Prepare the user prompt with document context and query.

    This function formats the retrieved documents and user query into
    a structured prompt for the LLM.

    Args:
        documents: List of (doc_id, content, filename) tuples
        query: The user's question

    Returns:
        Formatted prompt string with documents and query
    """
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
    query: str
) -> List[dict]:
    """
    Prepare complete message list with few-shot examples for the LLM.

    This function constructs the full conversation including:
    1. System prompt
    2. Few-shot examples demonstrating desired format
    3. User's actual query with document context

    Args:
        system_prompt: The system instruction for the LLM
        documents: List of (doc_id, content, filename) tuples
        query: The user's question

    Returns:
        List of message dictionaries ready for the LLM
    """
    messages = [{"role": "system", "content": system_prompt}]

    messages.extend(get_few_shot_messages())

    user_prompt = prepare_prompt(documents, query)
    messages.append({"role": "user", "content": user_prompt})

    return messages


def format_document(doc_id: int, content: str, filename: str) -> str:
    """
    Format a single document chunk for inclusion in the prompt.

    Args:
        doc_id: The chunk identifier
        content: The text content of the chunk
        filename: The source file name

    Returns:
        Formatted document string with metadata
    """
    return (
        f"ChunkID: {doc_id}\n"
        f"Source: {filename}\n"
        f"Content: {content}\n"
    )
