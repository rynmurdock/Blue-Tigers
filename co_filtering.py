import torch
import pandas as pd


# TODO test; frequency? etc
# TODO compare user_emb and item_embs to original CLIP embedding


DEVICE = 'cpu'

# TODO can test with just the df, so import and run in python console

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
    rhs = torch.mm(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = torch.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    res = torch.linalg.lstsq(T.to(DEVICE), rhs.to(DEVICE))
    return res.solution.squeeze().T # transpose to get r x n




def alternating_l_sq(interactions, clip_item_embs, mask, match_images, epochs=4):
    user_embs = torch.randn(interactions.shape[0], 1280, dtype=torch.bfloat16, device=DEVICE)
    item_embs = clip_item_embs.clone()
    
    interactions = (interactions - interactions.mean())
    if interactions.abs().max() != 0:
        interactions = interactions / interactions.abs().max()
    
    print(f'''Shapes:
        Interactions: {interactions.shape}
        Image_features: {item_embs.shape}
        User_features: {user_embs.shape}
        Mask: {mask.shape}
        Clip_item_embs: {clip_item_embs.shape}
        ''')

    for ep in range(epochs):
        user_embs = censored_lstsq(item_embs, interactions.T, mask.T).T
        if ep == epochs - 1:
            break
        item_embs = censored_lstsq(torch.cat([user_embs, clip_item_embs]), 
             torch.cat([interactions, match_images]),
             torch.cat([mask, match_images])).T
        print(f'Epoch {ep}.')
        
    # we should obtain a training loss metric; could even holdout a val set instead of eyeballing
    return user_embs

@torch.no_grad()
def all_embeddings(df, uid):
    # calculate user embeddings; should return the whole thing & run every x ratings, but will do one user for now.
    
    # could shard for user's history & update?
    # initialize all user & item embeddings with kept old for faster convergence?
    clip_item_embs = torch.tensor(df['embeddings'].to_list()).to(DEVICE).to(torch.float32)
    
    rated_rows_all = df[[i[1]['user:rating'] != {0: 0} for i in df.iterrows()]]
    
    users = []
    for i in rated_rows_all.iterrows():
        row = i[1]
        users += row['user:rating'].keys()
    users = list(set(users))
    
    interactions_matrix = torch.zeros(len(users), clip_item_embs.shape[0], device=DEVICE)
    for n_u, u in enumerate(users):
        for n_emb, row in rated_rows_all.iterrows():
            # -1 indicates unseen, otherwise gets rating
            rating = row['user:rating'].get(u, -1)
            interactions_matrix[n_u, n_emb] = rating
    mask = torch.ones_like(interactions_matrix)
    # mask unseen; we can weight by frequency by dividing later
    mask[interactions_matrix == -1] = 0
    mask.to(DEVICE)
    
    match_images = torch.zeros(clip_item_embs.shape[0], clip_item_embs.shape[0])
    match_images = match_images.fill_diagonal_(1).to(DEVICE)
    
    # use weighting?
    user_embs = alternating_l_sq(interactions_matrix, clip_item_embs, mask, match_images)
    user_emb = user_embs[[uid == u for u in users]]    
    

    return user_emb.to('cuda', dtype=torch.bfloat16) # easy to pluck





def uid_embeddings(df, uid):
    # calculate user embeddings; should return the whole thing & run every x ratings, but will do one user for now.
    
    # could shard for user's history & update?
    # initialize all user & item embeddings with kept old for faster convergence?
    
    rated_rows_all = df[[i[1]['user:rating'] != {0: 0} for i in df.iterrows()]]
    rated_from_user = rated_rows_all[[i[1]['user:rating'].get(uid, 'gone') != 'gone' for i in rated_rows_all.iterrows()]].reset_index()
    clip_item_embs = torch.tensor(rated_from_user['embeddings'].to_list()).to(DEVICE).to(torch.float32)
    
    print(rated_from_user)
    
    users = []
    for i in rated_from_user.iterrows():
        row = i[1]
        users += row['user:rating'].keys()
    users = list(set(users))
    
    interactions_matrix = torch.zeros(len(users), clip_item_embs.shape[0], device=DEVICE)
    for n_u, u in enumerate(users):
        for n_emb, row in rated_from_user.iterrows():
            # -1 indicates unseen, otherwise gets rating
            rating = row['user:rating'].get(u, -1)
            interactions_matrix[n_u, n_emb] = rating
    mask = torch.ones_like(interactions_matrix)
    # mask unseen; we can weight by frequency by dividing later
    mask[interactions_matrix == -1] = 0
    mask.to(DEVICE)
    
    match_images = torch.zeros(clip_item_embs.shape[0], clip_item_embs.shape[0])
    match_images = match_images.fill_diagonal_(1).to(DEVICE)
    
    # use weighting?
    user_embs = alternating_l_sq(interactions_matrix, clip_item_embs, mask, match_images)
    user_emb = user_embs[[uid == u for u in users]]    
    

    return user_emb.to('cuda', dtype=torch.bfloat16) # easy to pluck



























