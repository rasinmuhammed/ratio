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

## Measured, not claimed

Every number below comes from `scripts/evaluate.py` run against a 313-query
hand-verified label set (288 exact-citation, 25 conceptual), not from the
architecture alone. Full methodology, negative results included, in
[`retrieval_notes.md`](retrieval_notes.md).

| config | exact recall | exact precision | exact MRR |
|---|---|---|---|
| dense only | 0.040 | 0.033 | 0.099 |
| BM25 only | 0.586 | 0.440 | 0.760 |
| reranked (cross-encoder) | 0.730 | 0.565 | 0.896 |
| **routed (exact-index + semantic backfill)** | **0.892** | **0.710** | **1.000** |

Routing an identifier query to an exact-match index instead of any ranker
is the single largest win measured in this project, and it was found by
first proving hybrid fusion (BM25 + dense, reciprocal rank fusion)
underperforms BM25 alone on this exact query type — see
[`route.py`](src/rag/route.py)'s docstring for why averaging two systems
that are both wrong about identifiers is still wrong.

Structured JSON output (`answer_structured`) measured 91.2% citation
precision, 96.9% support, and 0.0% uncited claims on a 15-query
citation-compliance benchmark — the schema makes an unattributed claim
structurally impossible to emit, it does not by itself make the citation
correct, and roughly 40% of conceptual queries in that same benchmark
still refuse for lack of a clear holding.

## Architecture

1. **Summary-Augmented Chunking (SAC):** Every chunk is prepended with an LLM-generated 3-sentence summary of its parent document, so a chunk cannot look topically relevant while its parent case is not. Chunk size (450 tokens) was chosen for a stated reason, not itself swept against the label set — recorded as such, not implied as optimal.
2. **Corrective RAG (CRAG):** If the first retrieval pass produces a refusal, the query is decomposed into sub-questions, retrieved independently, and regenerated from the combined context. Not yet measured for how often it converts a refusal into a correct answer versus a confident wrong one.
3. **Stance Detection:** A court judgment records what counsel argued ("submission") alongside what the court decided ("holding"), often in the same paragraph. Every retrieved passage is labelled with its stance before reaching the prompt. This reduces the model treating an argument as settled law; it is a prompted mitigation, not a structural guarantee the way the citation schema is.
4. **Authority-Weighted Ranking:** The Indian court hierarchy (Supreme Court > High Court > Tribunal) and citation count are folded into re-ranking as a deterministic multiplier. Not yet isolated in the evaluation sweep — whether it helps or hurts conceptual-query precision specifically has not been measured on its own.
5. **Exact-Match Routing & Case-Sensitive BM25:** Legal identifiers (`AIR 1974`, `No. 123 of 2013`) are destroyed by both dense embeddings (which collapse `AIR` into `air`) and generic tokenizers. A deterministic router sends identifier queries to an exact-match index built at index time; everything else goes to the semantic path. Measured at 0.892 recall / 1.000 MRR on exact-citation queries (table above), not the same as zero loss.

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
    Prompt --> IFM[K2-Horizon-375B-A23B / GLM / gpt-oss]
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

> **Note on the current default index:** `scripts/ask.py` defaults to `data/index-full`, built above. The API server (`src/rag/api.py`) currently defaults to `data/index` (6,476 chunks) instead, a smaller but complete index built the same way, while `data/index-full`'s `payloads.jsonl` is being rebuilt. Override with `RATIO_INDEX_DIR`.

### Running the tests

```bash
uv run python -m pytest tests/
```

### Running the web UI

The `frontend/` directory is a Next.js app that talks to the FastAPI server in `src/rag/api.py`.

```bash
# Terminal 1: the API server
uv run uvicorn rag.api:app --reload

# Terminal 2: the frontend
cd frontend && npm install && npm run dev
```

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
Embeddings: `BAAI/bge-small-en-v1.5`, tested against `voyage-law-2`, which lost.
Reranking: `BAAI/bge-reranker-base`, tested against `bge-reranker-large`
(untested for a stated reason, see `retrieval_notes.md`) and
`ms-marco-MiniLM-L-12-v2` (measured, lost). Generation: `LLM_PROVIDER` switches
between Groq (`openai/gpt-oss-120b`, default), Mercury 2, TokenRouter (GLM) and
`IFM/K2-Horizon-375B-A23B`, chosen so a model swap is an env var, not a
temporary edit someone forgets to revert.

## License

MIT, see [`LICENSE`](LICENSE).
