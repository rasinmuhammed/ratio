# Retrieval notes


## 1. What the corpus is

Indian court judgments from `opennyaiorg/InJudgements_dataset`.

**11,970 source rows are only 10,588 distinct judgments.** The dataset is multi-label flattened: a judgment tagged with two case types appears twice with byte-identical text and a different `Case_Type`. Ingesting naively would have embedded 13% of the corpus twice, paid double the compute, and returned the same judgment in multiple result slots without any error.

**The text is dense with identifiers.** Counted across 500 documents:

| Pattern | Occurrences |
|---|---:|
| Section references | 20,520 |
| Dates | 8,408 |
| Reported citations (AIR, SCC) | 1,685 |
| Case numbers | 1,442 |

Roughly one exact-match token every 570 characters. This single table predicts every retrieval failure below.

**Structure is inconsistent.** Structured headers (`PETITIONER:`, `BENCH:`) appear in fewer than 1% of documents, so they are unusable as boundaries. Numbered paragraphs appear in 94%, but the numbering is cleanly ascending in only 47%, so it needs a validity check before being trusted.

---

## 2. Where dense retrieval fails

Ten queries were run by hand, five conceptual and five exact-match.

### 2.1 `AIR 1974` is the clearest failure

**Six chunks contain the literal string. Dense retrieval returned none of them.** Scores collapsed to 0.52-0.53 the lowest of any query tested, against 0.70-0.80 for everything else.

Chunks containing the string:

```
https://indiankanoon.org/doc/154031036/#4
https://indiankanoon.org/doc/154031036/#5
https://indiankanoon.org/doc/137384840/#11
(+3 more)
```

What it returned instead:

```
1. 0.534  doc/865397/#13    "it would be useful to briefly trace the history of the model forms... Model forms applicable to all countries"
2. 0.526  doc/913850/#61    "There is no material before me to express any opinion on that aspect and the Election Officer ba..."
3. 0.524  doc/1457597/#37   "body but it was disregarded on the ground that it was not raised at appropriate time..."
4. 0.523  doc/980589/#17    "of the contract, and operates to defeat and annul it, upon the subsequent failure of either party..."
5. 0.521  doc/1707698/#11   "appeal before the Privy Council related to the amount of customs duty payable upon 256 Ford motor cars..."
```

None contain the string. The spread between rank 1 and rank 5 is 0.013, which is the signature of a query the model has no signal for: it is not choosing between candidates, it is returning near-arbitrary text at near-identical scores.

### 2.2 Same-document crowding

Query `adverse possession`. **Four of the top five results came from one
document**, leaving a single slot for the rest of the corpus.

```
1. 0.790  doc/601148/#74   "or such estate as lies in grant, under such circumstances as indicate that such enjoyment has been commenced..."
2. 0.746  doc/601148/#73   "be valid and binding on defendant No. 1 and the plaintiffs under the circumstances..."
3. 0.708  doc/1711664/#14  "The only defence which he is trying to put up is that it was only a nominal and sham document..."
4. 0.701  doc/601148/#76   "prescription, fact and right, possession and ownership, tend to coincidence. Ex facto oritur jus..."
5. 0.695  doc/601148/#6    "defendant No. 1 had perfected his title to them by adverse possession by the date of the suit..."
```

**Checked, and the crowding is mostly legitimate.** Document `601148` is *Yarlagadda Venkakka Choudary vs Daggubati Lakshminarayana* (Andhra High Court, 1995), 80 chunks long. Of the 13 chunks in the whole index that mention adverse possession, **9 are in that document, and only 4 documents mention the term at all.**

So the retriever is not malfunctioning, the subset genuinely contains almost all of its adverse possession material in one judgment. Two things follow. First, crowding of this kind should be judged against the corpus, not assumed to be a bug. Second, the finding is contaminated by working on 2% of the corpus: on the full 10,588 documents there would be far more competing material, and this specific case may disappear. It needs re-checking against the full index before being reported as a real problem.

> Re-checked against the full 414,122 chunk index only in aggregate, through the
> section 4 benchmark, not for this specific query. `adverse possession` is a
> conceptual query and there are still no conceptual labels, so whether the
> crowding survives at full scale remains open.

### 2.3 A plausible wrong answer

Rank 3 above, score 0.708, is the interesting failure:

> "This aspect has to be concentrated rather than drawing adverse inference."

**Adverse inference**, not adverse possession. A different legal concept entirely, matched on the shared word and general legal register. Worth recording separately from the obvious failures: a wrong result that looks right is more dangerous than one that looks wrong, because a reader skimming results has no reason to check it.

### 2.4 A case where it worked, and why

Query `Maheshwar Mandal`. All five results from the correct judgment.

This works because of the context prefix: every chunk carries `Patna High Court | Maheshwar Mandal & Anr vs ... | No.1091 of 2013`, so the party name appears in *every* chunk of that document rather than once. The same design that rescues party names does nothing for `AIR 1974`, which appears once, buried in a paragraph about something else and diluted by 400 surrounding tokens.

### 2.5 The whole class of exact-match queries fails the same way

`No.1091 of 2013`, `24.06.2014` and `Section 9` all behave like `AIR 1974`: none returned a chunk containing the string it searched for, and all scored in the same compressed 0.65-0.69 band that indicates no real signal.

The conceptual queries, `when can a writ petition be dismissed for delay`, `grounds for granting an injunction`, `principles for awarding compensation in land acquisition`, `when is a contract void for lack of consideration`, returned plausible and topically relevant text in every case. That contrast is the entire argument for hybrid retrieval: the two approaches fail in complementary places.

---

## 3. Why dense retrieval fails here

Embeddings encode meaning. `AIR 1974` and `AIR 1961` mean approximately the same thing, "a citation to the All India Reporter", so the model places them almost on top of each other. To a lawyer they are entirely different cases.
The model is not broken. It is being asked to do something it was not built for, and the corpus is unusually dense with exactly that thing.

---

## 4. Benchmark

Run twice: once on 200 documents, once on the full corpus after the index was
rebuilt to 414,122 chunks. Both k=5, exact-match labels.

```
                  200 docs, 288 q        full corpus, 300 q
config          recall  prec    MRR     recall  prec    MRR
dense            0.040  0.033  0.099     0.007  0.004  0.011
bm25             0.559  0.423  0.717     0.508  0.273  0.516
hybrid 1:1       0.334  0.242  0.425     0.321  0.163  0.243
hybrid 1:3       0.528  0.404  0.646     0.498  0.267  0.466
```

**These two columns are not directly comparable and it changes the reading.**
The label distribution moved when the corpus grew: mean relevant chunks per
query fell from 4.57 to 3.06 and the median from 4 to 2, because the 2-to-12
chunk window now selects far rarer phrases. That lifts the recall@5 ceiling
from 0.892 to 0.970 and drops the precision@5 ceiling from 0.710 to 0.561.
Precision looks like it collapsed partly because the ceiling did.

Normalised against what is actually achievable:

```
config          200 docs    full corpus
dense              4.5%           0.7%
bm25              62.7%          52.4%
hybrid 1:1        37.4%          33.1%
hybrid 1:3        59.2%          51.3%
```

**Dense retrieval was not being underrated by the small corpus. It got worse.**
0.7% of achievable recall is indistinguishable from returning random chunks.

**The small corpus was flattering BM25 by about a fifth**, 62.7% down to 52.4%.
This is exactly the re-run that section 6 said every result needed.

**Hybrid still never beats BM25**, now at 64x the competing material.

### Caveats

**Every query in this set is exact-match**, which is precisely the regime dense
retrieval is expected to lose. This benchmark cannot conclude that hybrid
search is bad. It concludes that *on exact-match queries, at the weightings
tested, adding dense retrieval reduced accuracy.*

**BM25 at 0.508 and hybrid 1:3 at 0.498 are inside the noise.** Standard error
on a proportion at 300 queries is about 0.029, so the 0.010 gap is not a
result. The full 7,323-query sweep would settle it and has not been run,
because section 4.1 makes the question much less interesting than it looked.

---

## 4.1 Why BM25 loses, and why the fix is not a better ranker

Measured on the full index for the query `AIR 1006`, which has 2 relevant
chunks:

| token | chunks containing it | share |
|---|---:|---:|
| `air` | 41,382 | 9.99% |
| `1006` | 151 | 0.04% |
| the phrase `AIR 1006` | 2 | 0.0005% |

Neither token identifies anything. **The information is entirely in their
adjacency, and adjacency is what BM25 discards at tokenisation.** Its real task
is: given 151 candidates containing `1006`, guess which 2 also have `air`
immediately before it, using only term frequency and length normalisation.
Retaining 52% of achievable recall on that is close to the ceiling for a bag of
words, not a failure.

Dense retrieval never had the adjacency either, and worse, it actively
discards the distinction: `AIR 1006` and `AIR 1974` both mean "a citation to
the All India Reporter", so the model places them almost on top of each other.

This also explains the hybrid result structurally rather than empirically.
**Reciprocal rank fusion averages two systems that are both wrong for this
query class**, which is why every weighting tested landed between dense and
BM25 and never above BM25. No weighting can fix that, because there is no
weighting of two wrong answers that produces a right one.

So the problem was never the ranker. `AIR 1006` is a lookup wearing a ranking
problem's clothes, and the fix is to stop sending it to rankers at all.

### Routing

Identifiers are detected with the same patterns `build_labels.py` greps for and
sent to an exact index; everything else goes to the semantic path. Where a
query carries both, exact hits are placed first and semantic results fill the
remaining slots, because an identifier is a hard constraint and topical
similarity is a soft one.

```
config           recall  precision     MRR
routed            0.978      0.559   1.000
```

**This number is close to tautological and should not be read as a retrieval
result.** The labels define a relevant chunk as one containing the literal
string, and the router does exact substring lookup using the same regexes that
built the labels, so it satisfies the metric by construction. Precision 0.559
against a ceiling of 0.561 is not evidence of a good retriever.

What it does establish is architectural: **an entire query class was being
handed to statistical rankers that structurally cannot answer it**, and moving
it costs one regex pass at index time. The right comparison is not
routed-versus-BM25, it is that the exact-match half of this workload should
never have been a ranking problem.

The corollary is that the exact-match benchmark is now finished as a
development target. Its optimum is grep, grep is available, and any further
tuning against it optimises toward reinventing grep.

---

## 5. How the labels were built

Every chunk containing a distinctive literal string is relevant to a query for that string. Terms appearing in 2 to 12 chunks were kept: fewer and the term may be a typo, more and "the correct answer" stops meaning anything. That yielded 288 labelled queries with no human annotation.

**Ground truth was deliberately not LLM-generated.** Asking a model to invent queries and mark relevance would measure whether the retriever agrees with a language model, not whether it retrieves well. The results would look fine and mean nothing.

The cost of this choice is that the label set only covers exact-match retrieval. That limitation is real and is the main gap below.

---

## 6. What is not yet known

- **No conceptual labels.** The entire case for keeping dense retrieval rests on conceptual queries, and there is currently zero evidence either way. Now the blocking item rather than one of several: with routing in place, every query the semantic path handles is a query this benchmark cannot see. Needs hand annotation, and pooling to keep it tractable, since judging 414,122 chunks per query is not possible.
- **The semantic branch of the router is an unevidenced choice.** It is hybrid because hybrid was the general-purpose default, not because anything showed it belongs there. That is the same question as the line above.
- **Generation is not measured at all.** `generate.py` has a refusal sentinel verified on one out-of-corpus query and no groundedness, faithfulness or answer-quality measurement. The R in RAG is instrumented and the AG is not.
- **Everything is measured at k=5**, which is a search-results-page metric. In RAG the model reads whatever it is given, so recall@20 is the number that decides whether an answer can be grounded, and it has never been run.
- **Weight sweep is incomplete.** Only 1:1 and 1:3 tested, and section 4.1 makes the question largely moot for identifier queries.
- **Chunk size (450 tokens), overlap (200 chars) and RRF depth (20) are still unmeasured guesses.** They were chosen for defensible reasons and never tested. They cannot usefully be tested against exact-match labels, where chunk size barely affects whether a literal string is present.
- **Case is discarded at tokenisation**, so `AIR` the reporter merges with `air` the noun. In a corpus this dense with identifiers, case is signal. Untested.

### Resolved

- ~~The index covers 200 of 10,588 documents.~~ Rebuilt to the full corpus: 414,122 chunks, 0 truncated. The corpus averages 39.1 chunks per judgment, not the 32.4 the 200-document sample implied, so the full index is 64x the old one rather than the 53x predicted. Section 4 reports both runs.

---

## Appendix: two measurement mistakes worth recording

**The model limit was not what the documentation implied.** `max_seq_length` on the loaded model read 256, not the 512 assumed when chunk size was chosen, and the corpus tokenises at 4.18 characters per token rather than the estimated 3.5. **34% of chunks were being silently truncated** before the model saw them, with no error and no warning. Fixed by measuring chunk size with the model's own tokenizer and switching to a model with a real 512 token context.

**Similarity scores cannot detect irrelevance.** Measured against the whole corpus:

| Query | top1 | median | gap |
|---|---:|---:|---:|
| adverse possession of land | 0.766 | 0.563 | 0.203 |
| what is the recipe for chocolate cake | 0.502 | 0.354 | 0.148 |
| zxqv nonsense tokens here | 0.603 | 0.483 | 0.120 |

**Nonsense outscored a coherent off-topic question.** No absolute threshold separates them, so refusal cannot be implemented on raw score and is delegated to the model with an explicit sentinel instead.
