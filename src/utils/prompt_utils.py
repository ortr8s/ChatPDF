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
