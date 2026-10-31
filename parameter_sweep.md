# Parameter sweep experiments

## Baseline
zotero-rag eval --top-k 5
zotero-rag eval --top-k 7
zotero-rag eval --top-k 10
zotero-rag eval --top-k 20
zotero-rag eval --top-k 5  --reranker voyageai/rerank-3-lite
zotero-rag eval --top-k 7  --reranker voyageai/rerank-3-lite
zotero-rag eval --top-k 10 --reranker voyageai/rerank-3-lite

## Sweeping chunk-overlap ranges

# top_k=5, chunk-size=1000(baseline),500,1500
zotero-rag compare-embeddings --model openai/text-embedding-3-small \
  --chunk-size 500 --chunk-size 1500 --no-build --top-k 7 --yes
# top_k=7, chunk-size=1000(baseline),500,1500
zotero-rag compare-embeddings --model openai/text-embedding-3-small \
  --chunk-size 500 --chunk-size 1500 --no-build --top-k 7 --yes
# top_k=10, chunk-size=1000(baseline),500,1500
zotero-rag compare-embeddings --model openai/text-embedding-3-small --chunk-size 500 --chunk-size 1500
# top_k=20, chunk-size=1000(baseline),500,1500
zotero-rag compare-embeddings --model openai/text-embedding-3-small \
  --chunk-size 500 --chunk-size 1500 --no-build --top-k 20 --yes









Your baseline: 1000/200 → passage hit 88% (30 of 34 questions with a quote), mean score about 4.0.
