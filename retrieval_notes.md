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

288 exact-match queries, k=5.

```
config           recall  precision     MRR
dense             0.040      0.033   0.099
bm25              0.559      0.423   0.717
hybrid 1:1        0.334      0.242   0.425
hybrid 1:3        0.528      0.404   0.646
```

**Dense retrieval is not weak here, it is absent.** Recall 0.040 across 288 queries is close to chance.

**Hybrid never beat BM25 alone**, at either weighting tested. Reciprocal rank fusion treats both rankings as equally trustworthy, and when one of them is near-random its confidently ranked wrong answers displace the correct ones. Increasing the keyword weight from 1:1 to 1:3 recovered most of the loss but still did not reach BM25 on its own.

### Two caveats

**Every query in this set is exact-match**, which is precisely the regime
dense retrieval is expected to lose. This benchmark cannot conclude that
hybrid search is bad. It concludes that *on exact-match queries, at the
weightings tested, adding dense retrieval reduced accuracy.*

**Recall@5 is capped by construction.** Queries have a mean of 4.57 relevant
chunks, so the mean achievable recall@5 is **0.892**, not 1.0. BM25's 0.559 is
therefore 63% of the ceiling rather than 56% of a perfect score.

---

## 5. How the labels were built

Every chunk containing a distinctive literal string is relevant to a query for that string. Terms appearing in 2 to 12 chunks were kept: fewer and the term may be a typo, more and "the correct answer" stops meaning anything. That yielded 288 labelled queries with no human annotation.

**Ground truth was deliberately not LLM-generated.** Asking a model to invent queries and mark relevance would measure whether the retriever agrees with a language model, not whether it retrieves well. The results would look fine and mean nothing.

The cost of this choice is that the label set only covers exact-match retrieval. That limitation is real and is the main gap below.

---

## 6. What is not yet known

- **No conceptual labels.** The entire case for keeping dense retrieval rests on conceptual queries, and there is currently zero evidence either way. This is the next piece of work and it needs hand annotation.
- **Weight sweep is incomplete.** Only 1:1 and 1:3 tested. As keyword weigh rises, hybrid must converge to BM25; whether it ever exceeds it is unknown.
- **The index covers 200 of 10,588 documents**, about 2%. This is not just an incompleteness, it actively distorts findings: the crowding case in 2.2 looked like a retrieval flaw and turned out to be a property of a th e subset. Every result here needs re-running against the full index before it is reported as final.
- **Chunk size (450 tokens), overlap (200 chars) and RRF depth (20) are still unmeasured guesses.** They were chosen for defensible reasons and never tested.

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
