import torch
import pandas as pd


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


