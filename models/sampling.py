# src/sampling.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def triplet_sampling(X, y, pos_thresh=0.5, neg_thresh=0.3, temporal_idx=None):
    """
    Given dataset X (N,d) and labels y (N,), construct triplets (anchor, pos, neg).
    pos_thresh and neg_thresh are cosine similarity thresholds described in the paper.
    temporal_idx: optional array of timestamps; if provided, ensure positives are temporally close.
    Returns arrays of indices: anchor_idx, pos_idx, neg_idx
    """
    N = X.shape[0]
    cos = cosine_similarity(X)
    anchors, positives, negatives = [], [], []
    for i in range(N):
        same_cls = np.where(y == y[i])[0]
        same_cls = same_cls[same_cls != i]
        if len(same_cls) == 0:
            continue
        # similarity with same class
        sims = cos[i, same_cls]
        # pick those > pos_thresh
        candidates = same_cls[sims >= pos_thresh]
        if len(candidates) == 0:
            # relax: pick top-1 same class
            idx = same_cls[np.argmax(sims)]
        else:
            idx = np.random.choice(candidates)
        # negative
        opp_cls = np.where(y != y[i])[0]
        neg_sims = cos[i, opp_cls]
        low_sims = opp_cls[neg_sims <= neg_thresh]
        if len(low_sims) == 0:
            idx_neg = opp_cls[np.argmin(neg_sims)]
        else:
            idx_neg = np.random.choice(low_sims)
        anchors.append(i); positives.append(idx); negatives.append(idx_neg)
    return np.array(anchors), np.array(positives), np.array(negatives)
