import json
import os
import itertools
import sys
from pathlib import Path

# Add src to python path so we can import rag
sys.path.append("src")

from rag.ingest import load_documents
from rag.generate import GroqLLM

def main():
    print("Loading documents from corpus...")
    docs = list(itertools.islice(load_documents(), 5))
    
    llm = GroqLLM()
    
    test_cases = []
    
    system_prompt = """You are a legal expert. Given a legal document, generate exactly ONE conceptual legal question that can be answered using only this document, and the exact answer to that question based on the text.
Format your output EXACTLY as JSON:
{
    "question": "What is the legal test for X?",
    "ground_truth": "The legal test for X is..."
}
Do not include markdown blocks or any other text."""

    for i, doc in enumerate(docs):
        print(f"Generating test case for document {i+1}/5...")
        text = doc.text[:5000] # truncate to avoid huge contexts
        try:
            resp = llm.complete(system_prompt, text)
            # clean up markdown json blocks if present
            if resp.startswith("```json"):
                resp = resp[7:-3]
            elif resp.startswith("```"):
                resp = resp[3:-3]
                
            data = json.loads(resp.strip())
            test_cases.append({
                "question": data["question"],
                "ground_truth": data["ground_truth"],
                "contexts": [text]
            })
            print(f" -> Success: {data['question']}")
        except Exception as e:
            print(f" -> Failed to parse JSON from LLM: {e}")
            
    out_path = "data/ragas_testset.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(test_cases, f, indent=2)
        
    print(f"Saved {len(test_cases)} generated test cases to {out_path}")

if __name__ == "__main__":
    main()
