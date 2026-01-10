import numpy as np

def recall_at_k(results, ground_truth, k):
    """Compute Recall@K for a single query."""
    retrieved_ids = set(results[:k])
    return int(ground_truth in retrieved_ids)

def page_score_np(page_pred, page_true, n_pages, d):
    """Compute the page score using numpy."""
    score = (1 - np.abs(page_pred - page_true) / n_pages) * d
    return score


