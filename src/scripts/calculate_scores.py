import argparse
import os
import ast
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datasets import Dataset, load_dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Import local project utilities for RAG configuration
from src.core.rag import RAG
from src.utils.config import get_config
from src.utils.lexical_utils import Lemmatizer
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    ChatOpenAI = None
    OpenAIEmbeddings = None

def initialize_local_judge():
    """Initialize local LLM and Embeddings for the judge."""
    print("Initializing local RAG system for judging...")
    config = get_config()
    
    # Initialize basic RAG structure to get the generator
    # We don't need the full ingestion pipeline here, just the models
    reranker_model = config.get("models.reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    semantic_model = config.get("models.semantic_retriever", "all-MiniLM-L6-v2")
    generator_model = config.get("models.generator", "gpt-3.5-turbo")
    cache_dir = config.get("cache.dir", ".cache")
    
    rag = RAG(
        reranker_model=reranker_model,
        semantic_retriever_model=semantic_model,
        generator_model=generator_model,
        tokenize_func=lambda x: x.split(), # Dummy tokenizer
        lemmatizer=Lemmatizer(),
        cache_dir=cache_dir
    )
    
    # Wrap for Ragas
    langchain_llm = HuggingFacePipeline(pipeline=rag.generator.pipeline)
    langchain_embeddings = HuggingFaceEmbeddings(
        model_name=semantic_model,
        model_kwargs={'device': config.get("biencoder.device", "cpu")}
    )
    
    return langchain_llm, langchain_embeddings

def calculate_scores(input_path: str, output_path: str, force_local: bool = False):
    """
    Load data, run evaluation, and save results.
    """
    print(f"Loading evaluation data from {input_path}...")
    
    # Load dataset depending on file extension
    file_ext = os.path.splitext(input_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            dataset = load_dataset('csv', data_files=input_path, split='train')
            
            # Fix list parsing for CSV
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
            
            dataset = dataset.map(parse_lists)
            
        elif file_ext == '.json':
            dataset = load_dataset('json', data_files=input_path, split='train')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .json")
            
        print(f"Loaded {len(dataset)} samples.")
        
        # Validate dataset columns
        required_columns = {'question', 'answer', 'contexts', 'ground_truth'}
        if not required_columns.issubset(dataset.column_names):
            raise ValueError(
                f"Dataset missing required columns. "
                f"Found: {dataset.column_names}, Required: {required_columns}"
            )

        print("Starting Ragas evaluation...")
        
        # Setup Judge
        start_kwargs = {}
        if force_local:
            print("Force local flag set. Using local models as Judge.")
            llm, embeddings = initialize_local_judge()
            start_kwargs["llm"] = llm
            start_kwargs["embeddings"] = embeddings
        elif "OPENAI_API_KEY" not in os.environ:
            print("OPENAI_API_KEY not found. Using local models as Judge.")
            llm, embeddings = initialize_local_judge()
            start_kwargs["llm"] = llm
            start_kwargs["embeddings"] = embeddings
        else:
            print("Using OpenAI GPT-4 as Judge (OPENAI_API_KEY found).")
            if ChatOpenAI is None or OpenAIEmbeddings is None:
                raise ImportError("langchain-openai is not installed. Please install it to use OpenAI models.")
            
            # Explicitly instantiate to avoid Ragas internal default issues
            start_kwargs["llm"] = ChatOpenAI(model="gpt-4")
            start_kwargs["embeddings"] = OpenAIEmbeddings()
        
        # Run evaluation
        results = evaluate(
            dataset,
            metrics=[
                context_recall,
                context_precision,
                faithfulness,
                answer_relevancy,
            ],
            **start_kwargs
        )

        # Print Summary
        print("\n=== Evaluation Summary ===")
        print(results)

        # Save detailed results
        print(f"\nSaving detailed results to {output_path}...")
        results_df = results.to_pandas()
        results_df.to_csv(output_path, index=False)
        print("Done!")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Ragas scores for a RAG evaluation dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="evaluation_results.csv",
        help="Path to the input file (CSV or JSON) containing evaluation data."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_graded_results.csv",
        help="Path to save the graded results CSV."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force using local models for evaluation even if OPENAI_API_KEY is present."
    )

    args = parser.parse_args()

    # Check for API Key if using default Ragas configuration
    if not args.local and "OPENAI_API_KEY" not in os.environ:
        print("WARNING: 'OPENAI_API_KEY' not found. Will attempt to use local models configuration.")
    
    calculate_scores(args.input, args.output, force_local=args.local)
