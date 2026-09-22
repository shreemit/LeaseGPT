# LeaseGPT RAG Evaluation

Generated `2026-09-12T22:49:56.919077+00:00` against snapshot `2026-09-12` (`099d05411c0b`), using `BAAI/bge-small-en-v1.5`.

## Method

25 realistic queries were labeled with snapshot listing ids. Queries with no relevant ids are qualitative hard cases and are excluded from macro precision/recall.
The baseline is the app-equivalent dense FAISS ranking. The comparison retrieves 25 dense candidates and re-ranks explicit bedroom, price, neighborhood, and property-type constraints.
A chunk-size comparison was not used because every current listing is already one document; changing the 1,000-character split size would not alter this corpus.

## Retrieval summary

| Configuration | Recall@3 | Precision@3 | Recall@5 | Precision@5 | Recall@10 | Precision@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 7.2% | 9.1% | 8.7% | 6.4% | 19.3% | 6.8% |
| `constraint_rerank` | 26.1% | 31.8% | 39.8% | 30.0% | 42.0% | 15.9% |

## Findings

Constraint re-ranking changed mean recall@5 from 8.7% to 39.8% (+31.1%).
Across 25 judged queries, faithfulness averaged 4.72/5 and relevance averaged 4.76/5.

## Judge rubric

```text
Faithfulness (support from retrieved listings):
5 = Every factual claim is directly supported; no unsupported details.
4 = Grounded overall, with one minor unsupported or imprecise detail.
3 = Mostly grounded, but contains a meaningful unsupported claim or omission.
2 = Several claims are unsupported or conflict with the retrieved listings.
1 = Mostly fabricated, contradicted, or not based on the retrieved listings.

Relevance (answering the user's request):
5 = Directly answers the request and addresses all material constraints.
4 = Answers the request but misses one minor constraint or useful detail.
3 = Partially answers; one important constraint is ignored.
2 = Only loosely related to the request.
1 = Off-topic or fails to answer the request.
```

## Per-query retrieval

| ID | Query | Gold | Baseline hits@5 | Re-ranked hits@5 |
| --- | --- | ---: | ---: | ---: |
| q01 | Studio under $1,800 in Capitol Hill | 4 | 1 | 1 |
| q02 | 1 bedroom under $2,100 in Ballard | 4 | 1 | 2 |
| q03 | 2 bedroom under $2,800 in Fremont/Wallingford | 4 | 0 | 2 |
| q04 | 2 bedroom under $2,600 in West Seattle | 4 | 0 | 0 |
| q05 | 3 bedroom under $3,500 in Rainier Valley | 4 | 0 | 3 |
| q06 | Studio under $1,700 in the U-District | 4 | 1 | 1 |
| q07 | 1 bedroom under $2,000 in Northgate | 4 | 0 | 1 |
| q08 | 2 bedroom under $3,000 in Queen Anne | 4 | 0 | 1 |
| q09 | 1 bedroom under $2,300 in South Lake Union | 4 | 0 | 1 |
| q10 | 2 bedroom under $3,200 in Capitol Hill | 4 | 0 | 1 |
| q11 | 2 bedroom townhouse under $3,500 | 4 | 0 | 0 |
| q12 | 1 bedroom condo under $2,500 in Downtown | 2 | 0 | 0 |
| q13 | 3 bedroom single family home under $4,000 in West Seattle | 2 | 0 | 1 |
| q14 | 2 bedroom apartment under $2,600 in Bitter Lake | 4 | 0 | 3 |
| q15 | 4 bedroom under $4,500 in the U-District | 2 | 0 | 0 |
| q16 | 2 bedroom under $2,800 in Ballard | 4 | 0 | 3 |
| q17 | Studio under $1,900 in Central Seattle | 4 | 1 | 1 |
| q18 | 1 bedroom under $2,200 in Belltown | 4 | 0 | 4 |
| q19 | 3 bedroom under $4,000 in North Seattle | 4 | 0 | 2 |
| q20 | 1 bedroom apartment under $1,900 in Delridge | 3 | 2 | 3 |
| q21 | 2 bedroom under $2,700 in Mount Baker | 4 | 1 | 2 |
| q22 | 2 bedroom condo under $3,200 in Madison Park | 0 | — | — |
| q23 | 1 bedroom under $2,200 in Fremont/Wallingford | 4 | 0 | 1 |
| q24 | 2 bedroom under $2,000 near Capitol Hill with parking | 0 | — | — |
| q25 | Pet-friendly studio under $1,600 in Ballard | 0 | — | — |

## Per-query generation

| ID | Faithfulness | Relevance | Rationale |
| --- | ---: | ---: | --- |
| q01 | 5 | 5 | All listed facts are directly supported by the retrieved listings and the answer fully addresses the user’s request. |
| q02 | 5 | 5 | The answer accurately cites the single 1‑bedroom listing under $2,100 from the retrieved data and fully addresses the user’s request. |
| q03 | 5 | 5 | The answer correctly states that no 2‑bedroom listings under $2,800 exist, which is supported by the retrieved listings. |
| q04 | 5 | 5 | Answer is fully supported by retrieved listings and directly addresses the request. |
| q05 | 5 | 5 | Answer directly cites the retrieved 3‑bedroom listing under $3,500 and confirms no other such listings exist, fully matching the query. |
| q06 | 5 | 5 | All claims are directly supported by the retrieved listings and the answer fully addresses the user's request. |
| q07 | 3 | 3 | The answer correctly lists two 1‑bedroom units but mistakenly includes a 2‑bedroom listing as a 1‑bedroom option, violating the user’s bedroom requirement. |
| q08 | 5 | 5 | The answer correctly states that no 2‑bedroom listings under $3,000 exist, directly supported by the retrieved data. |
| q09 | 5 | 5 | All facts match the retrieved listing and the answer directly satisfies the user’s criteria. |
| q10 | 5 | 5 | Answer directly matches retrieved listing and satisfies constraints. |
| q11 | 5 | 5 | All listed properties match the user’s criteria and are directly supported by the retrieved data. |
| q12 | 5 | 5 | Answer correctly states no matching listings based on retrieved data. |
| q13 | 5 | 5 | Answer correctly states no listings meet all criteria based on retrieved data. |
| q14 | 5 | 5 | All facts match the retrieved listing and the answer directly addresses the user’s request. |
| q15 | 5 | 5 | Answer correctly states no 4‑bedroom listings under $4,500 in U‑District based on retrieved data. |
| q16 | 5 | 5 | Answer directly lists the only 2‑bedroom property in Ballard under $2,800, fully supported by retrieved listings. |
| q17 | 5 | 5 | Answer lists all Central Seattle studios under $1,900 from retrieved listings, directly addressing the request. |
| q18 | 5 | 5 | Answer correctly states no 1‑bedroom listings under $2,200 in Belltown based on the retrieved data. |
| q19 | 5 | 5 | Answer correctly states no matching listings based on retrieved data. |
| q20 | 5 | 5 | All listed apartments are directly supported by the retrieved data and fully answer the user’s request. |
| q21 | 5 | 5 | Answer correctly identifies the only 2‑bedroom listing under $2,700 in Mount Baker and provides accurate details from the retrieved data. |
| q22 | 5 | 5 | The answer correctly states that no 2‑bedroom condo under $3,200 is present in the retrieved listings, fully addressing the user’s request. |
| q23 | 5 | 5 | All listed units are 1‑bedroom, under $2,200, and match the retrieved data. |
| q24 | 4 | 5 | Answer correctly states no listings under $2000, but claims none include parking which is unsupported. |
| q25 | 1 | 1 | Answer does not provide requested info and lacks evidence. |
