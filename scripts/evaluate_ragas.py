import json
import pandas as pd
import sys
import types
from datasets import Dataset

# Monkey patch for ragas bug with langchain_community.chat_models.vertexai
import langchain_community.chat_models
dummy = types.ModuleType('langchain_community.chat_models.vertexai')
dummy.ChatVertexAI = type('ChatVertexAI', (object,), {})
sys.modules['langchain_community.chat_models.vertexai'] = dummy
langchain_community.chat_models.vertexai = dummy

from ragas import evaluate
from ragas.metrics import answer_correctness, context_precision, context_recall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

# Add src to python path so we can import rag
sys.path.append("src")
from rag.generate import answer as generate_answer, GroqLLM
from pathlib import Path
from rag.hybrid import HybridRetriever
from rag.rerank import CrossEncoderReranker, RerankedRetriever
from rag.route import RoutedRetriever 

def main():
    print("Loading testset...")
    with open("data/ragas_testset.json") as f:
        test_cases = json.load(f)
        
    print("Initializing RAG pipeline...")
    hybrid = HybridRetriever(Path("data/index-full"))
    reranked = RerankedRetriever(hybrid, CrossEncoderReranker())
    retriever = RoutedRetriever(reranked, hybrid.dense.payloads)
    llm = GroqLLM()
    
    results = []
    print("Generating answers for testset...")
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] Q: {tc['question']}")
        
        # Run our RAG
        ans = generate_answer(tc["question"], retriever, llm)
        
        # Ragas requires context as list of strings
        retrieved_contexts = [source.text for source in ans.sources]
        
        results.append({
            "question": tc["question"],
            "answer": ans.text,
            "contexts": retrieved_contexts,
            "ground_truth": tc["ground_truth"]
        })
        print(f" -> Generated answer length: {len(ans.text)}")
        
    dataset = Dataset.from_pandas(pd.DataFrame(results))
    
    print("Initializing evaluator models...")
    eval_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    eval_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    print("Running evaluation (this may take some time due to rate limits)...")
    run_config = RunConfig(max_workers=1, max_retries=10)
    
    try:
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=[answer_correctness, context_precision, context_recall],
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=run_config
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return
        
    print("Evaluation Results:")
    print(evaluation_result)
    
    # Save results
    out_path = "data/ragas_evaluation.csv"
    df = evaluation_result.to_pandas()
    df.to_csv(out_path, index=False)
    print(f"Saved evaluation results to {out_path}")

if __name__ == "__main__":
    main()
