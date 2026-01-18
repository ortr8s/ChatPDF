from src.utils.logger import Logger
from src.utils.lexical_utils import Lemmatizer
from src.utils.config import get_config
from src.core.rag import RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import evaluate
from datasets import load_dataset, Dataset
import sys
import ast
from typing import List, Dict, Any, Tuple
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logger = Logger(__name__)


def extract_paper_text(paper: Dict[str, Any]) -> str:
    """
    Extract full text from a Qasper paper structure.

    Qasper papers contain:
    - title: Paper title
    - abstract: Paper abstract
    - full_text: Dict with 'section_name' and 'paragraphs' lists
    """
    text_parts = []

    if paper.get("title"):
        text_parts.append(f"Title: {paper['title']}\n")

    if paper.get("abstract"):
        text_parts.append(f"Abstract: {paper['abstract']}\n")

    full_text = paper.get("full_text", {})
    section_names = full_text.get("section_name", [])
    paragraphs = full_text.get("paragraphs", [])

    for section_name, section_paragraphs in zip(section_names, paragraphs):
        if section_name:
            text_parts.append(f"\n## {section_name}\n")
        if section_paragraphs:
            for para in section_paragraphs:
                if para and para.strip():
                    text_parts.append(para.strip() + "\n")

    return "\n".join(text_parts)


def extract_qas(paper: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Extract question-answer pairs from a Qasper paper.

    Returns list of (question, ground_truth_answer) tuples.
    Only includes questions with extractive or free-form answers.
    """
    qas = []
    qas_data = paper.get("qas", {})
    questions = qas_data.get("question", [])
    answers_list = qas_data.get("answers", [])

    for question, answers in zip(questions, answers_list):
        if not question or not answers:
            continue

        answer_entries = answers.get("answer", [])
        for answer_entry in answer_entries:

            free_form = answer_entry.get("free_form_answer", "")
            if free_form and free_form.strip():
                qas.append((question, free_form.strip()))
                break

            extractive = answer_entry.get("extractive_spans", [])
            if extractive:
                combined = " ".join([span for span in extractive if span])
                if combined.strip():
                    qas.append((question, combined.strip()))
                    break

    return qas


def chunk_text(
        text: str,
        chunk_size: int = 512,
        overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for ingestion.

    Args:
        text: Full document text
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:

            for sep in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < text_len else text_len

    return chunks


def initialize_rag() -> RAG:
    """Initialize RAG system with configuration."""
    config = get_config()

    reranker_model = config.get(
        "models.reranker",
        "cross-encoder/ms-marco-MiniLM-L-6-v2")
    semantic_model = config.get(
        "models.semantic_retriever",
        "all-MiniLM-L6-v2")
    generator_model = config.get("models.generator", "gpt-3.5-turbo")
    cache_dir = config.get("cache.dir", ".cache")

    lemmatizer = Lemmatizer()

    def tokenize_func(text: str) -> List[str]:
        return text.split()

    rag = RAG(
        reranker_model=reranker_model,
        semantic_retriever_model=semantic_model,
        generator_model=generator_model,
        tokenize_func=tokenize_func,
        lemmatizer=lemmatizer,
        cache_dir=cache_dir
    )

    return rag


def inject_paper_into_kb(
        rag: RAG,
        paper_text: str,
        paper_id: str) -> List[str]:
    """
    Inject paper text directly into knowledge base, bypassing PDF reader.

    Args:
        rag: RAG instance
        paper_text: Full text of the paper
        paper_id: Identifier for the paper

    Returns:
        List of text chunks created
    """

    rag.kb.corpus = []
    rag.kb.file_map = {}
    rag.kb.embeddings = None
    rag.kb.is_indexed = False

    chunks = chunk_text(paper_text)

    if not chunks:
        logger.log(f"No chunks generated for paper {paper_id}", "WARNING")
        return []

    file_map = {i: paper_id for i in range(len(chunks))}

    rag.semantic_retriever.update_corpus_embeddings(iter(chunks))
    embeddings = rag.semantic_retriever.corpus_embeddings

    rag.kb.update_data(chunks, file_map, embeddings)

    rag.lexical_retriever.build(rag.kb.corpus)

    logger.log(f"Injected {len(chunks)} chunks for paper {paper_id}", "INFO")

    return chunks


def retrieve_contexts(rag: RAG, query: str) -> Tuple[str, List[str]]:
    """
    Retrieve answer and contexts for a query.

    Directly accesses search_engine to capture retrieved contexts
    required by Ragas evaluation.

    Args:
        rag: RAG instance
        query: User question

    Returns:
        Tuple of (generated_answer, list_of_context_chunks)
    """

    try:
        top_docs = rag.search_engine.search(query, rag.kb)
    except ValueError as e:
        logger.log(f"Search error: {e}", "WARNING")
        return "", []

    if not top_docs:
        return "No relevant documents found.", []

    contexts = [rag.kb.corpus[idx] for idx, _, _ in top_docs]

    answer_parts = []
    for token in rag.stream_response(query):
        if isinstance(token, dict):

            continue
        answer_parts.append(token)

    answer = "".join(answer_parts)

    return answer, contexts


def run_evaluation(
    num_papers: int = 5,
    output_path: str = "evaluation_results.csv",
    input_valid_file: str = None
) -> Dataset:
    """
    Run full evaluation pipeline on Qasper dataset.

    Args:
        num_papers: Number of papers to evaluate
        output_path: Path to save evaluation results CSV
        input_valid_file: Path to existing evaluation results CSV (if provided, skips generation)

    Returns:
        Dataset with evaluation results
    """
    if input_valid_file:
        logger.log(
            f"Loading existing evaluation data from {input_valid_file}...",
            "INFO")
        eval_dataset = load_dataset(
            "csv", data_files=input_valid_file, split="train")

        def parse_lists(example):
            for col in ["contexts", "ground_truth", "ground_truths"]:
                if col in example and isinstance(example[col], str):
                    try:

                        val = example[col].strip()
                        if val.startswith("[") and val.endswith("]"):
                            example[col] = ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        pass
            return example

        eval_dataset = eval_dataset.map(parse_lists)
        logger.log(f"Loaded {len(eval_dataset)} samples", "INFO")

        logger.log("Initializing RAG system (for Judge)...", "INFO")
        rag = initialize_rag()

    else:
        logger.log(
            f"Loading Qasper validation set ({num_papers} papers)...",
            "INFO")

        qasper = load_dataset(
            "allenai/qasper",
            split=f"validation[:{num_papers}]",
            trust_remote_code=True
        )

        logger.log("Initializing RAG system...", "INFO")
        rag = initialize_rag()

        questions = []
        answers = []
        contexts_list = []
        ground_truths = []

        for paper_idx, paper in enumerate(qasper):
            paper_id = paper.get("id", f"paper_{paper_idx}")
            logger.log(
                f"Processing paper {
                    paper_idx + 1}/{num_papers}: {paper_id}",
                "INFO")

            paper_text = extract_paper_text(paper)
            if not paper_text.strip():
                logger.log(
                    f"Empty text for paper {paper_id}, skipping",
                    "WARNING")
                continue

            inject_paper_into_kb(rag, paper_text, paper_id)

            qas = extract_qas(paper)
            if not qas:
                logger.log(
                    f"No Q&A pairs found for paper {paper_id}",
                    "WARNING")
                continue

            logger.log(
                f"Found {
                    len(qas)} Q&A pairs for paper {paper_id}",
                "INFO")

            for question, ground_truth in qas:
                try:
                    answer, contexts = retrieve_contexts(rag, question)

                    if not contexts:
                        logger.log(
                            f"No contexts retrieved for: {question[:50]}...", "WARNING")
                        continue

                    questions.append(question)
                    answers.append(answer)
                    contexts_list.append(contexts)
                    ground_truths.append(ground_truth)

                    logger.log(
                        f"Processed Q: {question[:50]}... | "
                        f"Contexts: {len(contexts)} | Answer length: {len(answer)}",
                        "DEBUG"
                    )
                except Exception as e:
                    logger.log(f"Error processing question: {e}", "ERROR")
                    continue

        if not questions:
            logger.log("No evaluation samples collected!", "ERROR")
            return None

        logger.log(f"Collected {len(questions)} evaluation samples", "INFO")

        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        })

        eval_df = eval_dataset.to_pandas()
        eval_df.to_csv(output_path, index=False)
        logger.log(f"Saved evaluation data to {output_path}", "INFO")

    logger.log("Running Ragas evaluation...", "INFO")
    try:

        config = get_config()
        langchain_llm = HuggingFacePipeline(pipeline=rag.generator.pipeline)
        langchain_embeddings = HuggingFaceEmbeddings(
            model_name=config.get(
                "models.semantic_retriever",
                "all-MiniLM-L6-v2"),
            model_kwargs={
                'device': config.get(
                    "biencoder.device",
                    "cpu")})

        results = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=langchain_llm,
            embeddings=langchain_embeddings
        )

        logger.log("Evaluation Results:", "INFO")
        for metric in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
                "context_recall"]:
            try:
                val = results[metric]
                if isinstance(val, (int, float)):
                    logger.log(f"  {metric}: {val:.4f}", "INFO")
                else:
                    logger.log(f"  {metric}: {val} (raw)", "INFO")
            except Exception:
                logger.log(f"  {metric}: N/A", "WARNING")

        results_df = results.to_pandas()
        results_path = output_path.replace(".csv", "_ragas_scores.csv")
        results_df.to_csv(results_path, index=False)
        logger.log(f"Saved Ragas scores to {results_path}", "INFO")

    except Exception as e:
        logger.log(f"Ragas evaluation failed: {e}", "ERROR")
        logger.log(
            "Evaluation data saved. Run Ragas manually if needed.",
            "INFO")

    return eval_dataset


def main():
    """Main entry point for evaluation script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate ChatPDF RAG using Qasper dataset and Ragas"
    )
    parser.add_argument(
        "--num-papers",
        type=int,
        default=10,
        help="Number of papers to evaluate (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV path (default: evaluation_results.csv)"
    )
    parser.add_argument(
        "--grade-only",
        type=str,
        help="Path to existing evaluation results CSV. If provided, skips generation and only runs grading."
    )

    args = parser.parse_args()

    logger.log("=" * 60, "INFO")
    logger.log("ChatPDF RAG Evaluation with Qasper + Ragas", "INFO")
    logger.log("=" * 60, "INFO")

    run_evaluation(
        num_papers=args.num_papers,
        output_path=args.output,
        input_valid_file=args.grade_only
    )

    logger.log("Evaluation complete!", "INFO")


if __name__ == "__main__":
    main()
