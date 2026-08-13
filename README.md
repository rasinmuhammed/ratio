# rag-from-scratch

Retrieval-augmented generation over 10,588 Indian court judgments, built without
a RAG framework. Ingestion, chunking, embedding, dense retrieval, BM25, rank
fusion, query routing and generation are each written here and each measured.

The point was never the pipeline. Any of this can be assembled from a library in
an afternoon. The point was to find out what actually happens when you run it on
a real corpus, which turns out to be different from what the tutorials imply.

Everything measured is in [`retrieval_notes.md`](retrieval_notes.md), including
the results that went against me.

---

## What it looks like

```
$ uv run python scripts/ask.py "when can a writ petition be dismissed for delay"

routed as: conceptual  (semantic)

A writ petition can be dismissed for delay when there is inordinate and
unexplained delay [1] ... However, the petitioner's submission that a writ
petition ought to be dismissed on the ground of delay if filed after three
years, as referred to in Tilokchand Motichand, is not the court's finding but
rather a submission [2].

==============================================================================
#  stance      cited  court and judgment
------------------------------------------------------------------------------
*1 both            7  Madras High Court | G. Easwaran vs The Government Of
   https://indiankanoon.org/doc/95273398#67
*2 argument        8  Bombay High Court | Bombay Environmental Action ...
   https://indiankanoon.org/doc/1724514/#281
*3 holding         1  Karnataka High Court | M K Thyagaraja Gupta vs The S
   https://indiankanoon.org/doc/98490697#32

* cited by the model. 3 of 3 sources used.
```

That last paragraph of the answer is the interesting part, and it is worth
explaining why.

## A judgment is not uniform text

A judgment records what counsel argued as well as what the court decided, and
the two routinely disagree. "Learned counsel submitted that limitation does not
apply" is a proposition that may be rejected two paragraphs later.

Nothing in cosine similarity or BM25 can see that difference. Both score a
passage on whether it discusses limitation, not on whether the court agreed.

Here is the same query, the same retrieval and the same model, with and without
each source labelled by whose voice it is in.

**Without:**

> The Supreme Court has dismissed writ petitions filed after delays of 5 months
> [2], 8 months [2]...

Source [2] records that as counsel's argument. The answer has attributed it to
the Supreme Court, fluently, with a correct citation. Wrong and credible is the
worst failure available to a legal RAG system.

**With:**

> the petitioner's submission that a writ petition ought to be dismissed on the
> ground of delay if filed after three years ... is not the court's finding but
> rather a submission [2]

The control holds: on a query where no argument passage is retrieved, the two
answers say substantively the same thing.

This is an existence proof on one query, not a rate.

## Numbers

The corpus is 11,970 rows that deduplicate to **10,588 distinct judgments**,
chunked to **414,122 chunks**, none truncated at the embedding step.

**Exact-match retrieval**, 300 of 7,323 labelled queries, k=5:

```
config           recall  precision     MRR
dense             0.007      0.004   0.011
bm25              0.508      0.273   0.516
hybrid 1:1        0.321      0.163   0.243
hybrid 1:3        0.498      0.267   0.466
routed            0.978      0.559   1.000
```

Ceilings for this label set are 0.978 recall@5 and 0.559 precision@5, so BM25
retains 52% of what is achievable and dense retains 0.7%.

At k=20, which is closer to what a generation pipeline actually consumes, BM25
reaches 0.699 and hybrid 1:3 ties it exactly. **Nineteen percent of relevant
chunks sit between rank 6 and rank 20**, found and then never looked at. So
fusion costs precision at the top of the ranking and costs nothing in recall at
the depth that matters.

The `routed` row is close to tautological and should not be read as a retrieval
result: the labels define a relevant chunk as one containing the literal string,
and the router matches literal strings. What it establishes is architectural,
which is the next section.

## Three things this corpus taught me

**Fusion is the wrong operator when query types are heterogeneous.** For the
query `AIR 1006`, the token `air` appears in 41,382 chunks, `1006` in 151, and
the phrase in 2. The information is entirely in the adjacency, which BM25
discards at tokenisation and embeddings never had. Hybrid search landed between
dense and BM25 at every weighting tested, and no weighting of two wrong answers
produces a right one. Identifiers now go to an exact index and everything else
to the semantic path.

**Rank fusion is an accidental authority filter.** Retrieved judgments have a
median `cited_by` of 18 for dense and 16 for BM25 against a corpus median of 6,
but 39 for hybrid. RRF promotes what both systems found, and agreement between a
lexical and a semantic method is itself evidence that a passage is a canonical
statement rather than an incidental mention. Whether that helps relevance is
still unmeasured.

**Dense retrieval earns its place somewhere.** It returns 5.3% argument passages
against BM25's 10.7%, roughly half. Argument passages are formulaic, so BM25
matches the boilerplate while the embedding attends to substance. This is the
only measurement here where dense beats BM25 at anything, and it needed no
labels.

## What is not measured

Stated plainly because a benchmark that only reports its wins is not a
benchmark.

- **No conceptual relevance labels.** Every retrieval number above comes from
  exact-match queries, which is precisely the regime dense retrieval is expected
  to lose. Whether the semantic path is any good is unevidenced.
- **Generation is barely measured.** There is a refusal sentinel that works and
  a single stance comparison. No groundedness or faithfulness scoring.
- **Conceptual queries are unlabelled**, so the k=20 rerun below covers
  exact-match only and says nothing about the semantic path.
- **Chunk size, overlap and fusion depth are unmeasured guesses**, chosen for
  defensible reasons and never tested.
- The stance classifier is regex and agrees with a language model on 52% of a
  100-chunk sample. Much of that gap is definitional, but not all of it.

## Running it

```bash
uv sync
uv run python scripts/download_corpus.py
uv run python scripts/build_index.py --out data/index-full
```

The index build takes about three and a half hours and produces roughly 1.2 GB.

```bash
export GROQ_API_KEY=...
uv run python scripts/ask.py "grounds for granting a temporary injunction"
uv run python scripts/ask.py            # interactive
```

Startup is around 85 seconds: 414,122 chunks, 38,627 identifiers. Run without a
query to pay that once and ask repeatedly.

```bash
uv run pytest -q      # 86 tests
```

## Reading the source

In dependency order, which is also the order it was built:

| | |
|---|---|
| [`ingest.py`](src/rag/ingest.py) | Corpus boundary. Deduplicates the multi-label flattening, strips HTML that appears mid-sentence in 10% of documents. |
| [`chunk.py`](src/rag/chunk.py) | Numbered-paragraph boundaries with an ascending-order check, because the numbering is clean in only 47% of documents. Sizes measured with the model's own tokenizer. |
| [`embed.py`](src/rag/embed.py) | Sharded index writes, vectors normalised at write time so retrieval is one dot product. |
| [`retrieve.py`](src/rag/retrieve.py) | Dense retrieval. `rank()` is a pure function so it can be tested without an index. |
| [`keyword.py`](src/rag/keyword.py) | BM25 with the derivation in comments. Postings in `array.array`, which is the difference between 3.7 GB and 0.7 GB at corpus scale. |
| [`hybrid.py`](src/rag/hybrid.py) | Reciprocal rank fusion, fusing on rank because dense and BM25 scores are not comparable units. |
| [`route.py`](src/rag/route.py) | Identifier detection and exact lookup, with semantic backfill. |
| [`stance.py`](src/rag/stance.py) | Whose voice a passage is in. |
| [`generate.py`](src/rag/generate.py) | Prompt assembly, citation parsing, refusal sentinel. |
| [`evaluate.py`](src/rag/evaluate.py) | recall@k, precision@k, MRR, reported per query kind. |

Corpus: [`opennyaiorg/InJudgements_dataset`](https://huggingface.co/datasets/opennyaiorg/InJudgements_dataset).
Embeddings: `BAAI/bge-small-en-v1.5`. Generation: Groq, `llama-3.3-70b-versatile`.
