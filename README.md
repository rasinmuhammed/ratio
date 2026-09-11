# Ratio

> **Work in Progress.** The core retrieval and generation pipeline is functional, but active development is ongoing — the full index build, frontend polish, and evaluation benchmarks are still being completed.

Ratio is a Retrieval-Augmented Generation (RAG) engine built specifically for Indian legal jurisprudence. 

The name "Ratio" stems from ***Ratio Decidendi***—the rationale for the decision. In common law, the *ratio decidendi* is the binding legal principle upon which a court's decision is founded. The core philosophy of this engine is exactly that: to separate the binding legal holding from mere facts, procedural history, or the arguments submitted by counsel.

Built without black-box RAG frameworks, every layer of Ratio—from tokenization to chunking to retrieval and generation—is custom-engineered and mathematically measured against the complexities of real legal texts.

## The Dataset

Ratio is built over a corpus of **10,588 Indian court judgments** (sourced from Indian Kanoon), encompassing:
- The Supreme Court of India
- High Courts (Division and Single Benches)
- Specialized Tribunals

The raw corpus contains 11,970 rows that deduplicate to 10,588 distinct judgments, ultimately chunked into **414,122** chunks.

## Architecture Upgrades

Ratio implements several advanced techniques pioneered in recent legal NLP research (final benchmarking pending):

1. **Summary-Augmented Chunking (SAC):** Eliminates Document-Level Retrieval Mismatch. Every single chunk is prepended with an LLM-generated 3-sentence summary of the parent document (the core issue and outcome). This guarantees that the retriever never pulls a perfectly matching clause from a completely irrelevant case.
2. **Corrective RAG (CRAG):** Implements a self-reflection loop. If the initial retrieval sweep fails to find the answer, the system intercepts the refusal, uses the LLM to decompose the complex query into simpler sub-questions, runs independent retrieval for each, and regenerates the answer using the expanded context.
3. **Stance Detection:** A court judgment is not uniform text. It records what counsel argued ("submission") and what the court decided ("holding"). Ratio labels every retrieved chunk with its stance, ensuring the LLM never hallucinates a lawyer's argument as a Supreme Court ruling.
4. **Authority-Weighted Ranking:** Integrates the Indian Court hierarchy and citation network directly into the semantic scoring math. A Supreme Court precedent mathematically outranks a single-judge High Court order.
5. **Exact-Match Routing & Case-Sensitive BM25:** Legal identifiers (like `AIR 1974` or `No. 123`) destroy dense embeddings. Ratio uses a custom query router and a dual-pass case-sensitive BM25 tokenizer to perfectly isolate and retrieve exact legal citations before semantic backfill even begins.

### System Flow

```mermaid
graph TD
    %% Styling
    classDef user fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#039be5,stroke-width:2px;
    classDef data fill:#fff3e0,stroke:#ffb300,stroke-width:2px;
    classDef llm fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;

    Q([User Query]) --> Router{Query Router}
    
    %% Routing
    Router -->|Contains Legal Citation\n(e.g., 'AIR 1974')| Exact[Exact Match Index]
    Router -->|Conceptual Query\n(e.g., 'writ petition delay')| Semantic[Semantic Retriever]
    
    %% Retrieval
    Exact --> Fusion[Candidate Fusion]
    Semantic --> BGE[bge-small Embeddings]
    Semantic --> BM25[Case-Sensitive BM25]
    BGE --> Auth[Authority-Weighted Ranking]
    BM25 --> Auth
    Auth --> Cross[Cross-Encoder Reranker]
    Cross --> Fusion
    
    %% Generation
    Fusion --> Stance[Stance Detection Labeling]
    Stance --> Prompt[Prompt Assembly]
    Prompt --> IFM[K2-Horizon / Llama 3]
    IFM --> Check{Answer Refused?}
    
    %% CRAG
    Check -->|Yes| CRAG[CRAG Decomposition]
    CRAG --> Semantic
    
    %% Output
    Check -->|No| Answer([Cited & Stance-Aware Answer])
    
    %% Classes
    class Q,Answer user;
    class Router,Exact,Semantic,Fusion,Auth,Cross,Stance,Prompt,Check process;
    class BGE,BM25 data;
    class IFM,CRAG llm;
```

## Running Ratio

```bash
# 1. Install dependencies
uv sync

# 2. Download the corpus
uv run python scripts/download_corpus.py

# 3. Build the full Index (Chunking, Summarizing, and Embedding)
# This takes a few hours and produces roughly 1.2 GB
uv run python scripts/build_index.py --out data/index-full --summarise

# 4. Ask a question
export LLM_PROVIDER=ifm  # or GROQ_API_KEY=...
uv run python scripts/ask.py "grounds for granting a temporary injunction"
uv run python scripts/ask.py            # interactive mode
```

> **Note on Startup:** Loading the 414,122 chunks, embeddings, and BM25 postings into RAM takes about 85 seconds. Run `ask.py` without a query to pay that cost once and enter the interactive prompt.

## Reading the Source

In dependency order:

| | |
|---|---|
| [`chunk.py`](src/rag/chunk.py) | Boundary detection and Summary-Augmented Chunking (SAC) injection. |
| [`embed.py`](src/rag/embed.py) | Sharded index writes, vectors normalised at write time. |
| [`retrieve.py`](src/rag/retrieve.py) | Dense retrieval logic. |
| [`keyword.py`](src/rag/keyword.py) | Case-sensitive BM25 with `array.array` postings for memory efficiency. |
| [`hybrid.py`](src/rag/hybrid.py) | Reciprocal rank fusion for lexical + semantic search. |
| [`route.py`](src/rag/route.py) | Exact lookup routing and Authority-Weighted re-scoring. |
| [`corrective.py`](src/rag/corrective.py) | CRAG implementation for query decomposition and retry. |
| [`stance.py`](src/rag/stance.py) | Determines whose voice a passage is in (holding vs argument). |
| [`generate.py`](src/rag/generate.py) | Prompt assembly, citation parsing, refusal sentinel. |
| [`evaluate.py`](src/rag/evaluate.py) | Metric harnesses for recall@k, precision@k, and MRR. |

Corpus: [`opennyaiorg/InJudgements_dataset`](https://huggingface.co/datasets/opennyaiorg/InJudgements_dataset).
Embeddings: `BAAI/bge-small-en-v1.5`. Generation: `K2-Horizon` / `Llama 3.3`.
