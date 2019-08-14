import numpy as np
import torch
from tqdm import tqdm_notebook
# define a function that loads data from GloVe-like embedding files
# we will add tutorials for loading contextualized embeddings later
# 2196017 is the vocab size of GloVe here.

def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
        
    f = open(path_to_embedding, encoding="utf8")
    found = 0

    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()       
