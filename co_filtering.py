import torch
import pandas as pd






# From https://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
def censored_lstsq(A, B, M):
    """Solves least squares problem subject to missing data.

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Note: we should check A is full rank but we won't bother...

    # if B is a vector, simply drop out corresponding rows in A
    #if B.ndim == 1 or B.shape[1] == 1:
    #   return np.linalg.lstsq(A[M.to(torch.long)], B[M.to(torch.long)])[0]

    # else solve via tensor representation
    rhs = torch.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = torch.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    res = torch.linalg.lstsq(T.to('cuda'), torch.from_numpy(rhs).to('cuda'))
    return res.solution.to('cpu').squeeze().T # transpose to get r x n















def alternating_l_sq(interactions, clip_item_embs, epochs=8):
    user_embs = torch.randn(interactions.shape[0], 1280, dtype=torch.bfloat16, device='cuda')
    item_embs = torch.randn(interactions.shape[0], 1280, dtype=torch.bfloat16, device='cuda')

    interactions = torch.cat(interactions, clip_positives) 
    # see generativerecsys original repo for correct concat, zeroing, etc.
    # we can test by replacing SVC in the singular calc as well...
    # may want to add an l2 term, but then I think we're best off just using ridge?

    for ep in range(epochs):
        item_embs = torch.linalg.solve_ex(interactions, user_embs)
        user_embs = torch.linalg.solve_ex(interactions, item_embs)
    # we should obtain a training loss metric; could even holdout a val set instead of eyeballing
    return user_embs

def update_all_embeddings(df, all_user_embs=None):
    # starting with all but could shard for user's history & update
    # initialize all user & item embeddings with kept old for faster convergence?
    clip_item_embs = df['embeddings']
    user_ids = # collate all users from user:rating
               # create interactions matrix from there
               # probably should have a separate user emb matrix -- gotta be better ways lol
    for uid in 
    interactions = # do that thing with masking if you can;
                   #    otherwise you may want to calc by users so you can drop unseen
                   #    or use weighting, but that's implicit unnecessarily


    return user_embeddings, item_embeddings # easy to pluck


