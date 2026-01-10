import nmslib
import bm25s

def get_embeddings(texts, model, show_progress_bar=True) -> list:
    """Generate embeddings for a list of texts using the specified model."""
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True
    ).tolist()
    return embeddings

def create_nmslib_index(embeddings, space='l2', method='hnsw'):
    index = nmslib.init(space=space, method=method)
    index.addDataPointBatch(embeddings)
    index.createIndex({'post': 2}, print_progress=False)
    return index

def create_bm25_retriever(corpus_tokens:list[str]):
    corpus_tokens = bm25s.tokenize(corpus_tokens)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    return retriever

def ann_search(query, index, model, top_k=5) -> tuple[list[int], list[float]]:
    """Find top_k similar pages for a given query using the provided nmslib index and model."""
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    ids, distances = index.knnQuery(query_embedding[0], k=top_k)
    return ids, distances

def bm25_search(query, retriever, top_k=5) -> list[int]:
    query_tokens = bm25s.tokenize([query])
    results, scores = retriever.retrieve(query_tokens, k=top_k)
    # results is a 2D array, we need the first row (for our single query)
    indices = results[0].tolist()
    return indices