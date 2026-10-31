# Parameter sweep experiments

## Baseline

### Baseline

zotero-rag eval --top-k 5
zotero-rag eval --top-k 7
zotero-rag eval --top-k 10
zotero-rag eval --top-k 20
zotero-rag eval --top-k 5  --reranker voyageai/rerank-3-lite
zotero-rag eval --top-k 7  --reranker voyageai/rerank-3-lite
zotero-rag eval --top-k 10 --reranker voyageai/rerank-3-lite

Defaults
zotero-rag eval
zotero-rag eval --reranker voyageai/rerank-3-lite


### Reconstruct best cases with different answer model

zotero-rag eval                                          # k=7, no reranker
zotero-rag eval
zotero-rag eval --reranker voyageai/rerank-3-lite        # k=7 + reranker
zotero-rag eval --reranker voyageai/rerank-3-lite


## Sweeping chunk-overlap ranges

# top_k=5, chunk-size=1000(baseline),500,1500
zotero-rag compare-embeddings --model openai/text-embedding-3-small \
  --chunk-size 500 --chunk-size 1500 --no-build --top-k 7 --yes
# top_k=7, chunk-size=1000(baseline),500,1500
zotero-rag compare-embeddings --model openai/text-embedding-3-small \
  --chunk-size 500 --chunk-size 1500 --no-build --top-k 7 --yes
# top_k=10, chunk-size=1000(baseline),500,1500
zotero-rag compare-embeddings --model openai/text-embedding-3-small --chunk-size 500 --chunk-size 1500 --yes
# top_k=20, chunk-size=1000(baseline),500,1500
zotero-rag compare-embeddings --model openai/text-embedding-3-small \
  --chunk-size 500 --chunk-size 1500 --no-build --top-k 20 --yes









Your baseline: 1000/200 → passage hit 88% (30 of 34 questions with a quote), mean score about 4.0.
