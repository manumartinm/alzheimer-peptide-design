import numpy as np

from bbb_classifier.features import batch_esm_embeddings


def test_esm_embedding_shape():
    seqs = ["ACDE", "GGGG"]
    x = batch_esm_embeddings(seqs, dim=64)
    assert x.shape == (2, 64)
    assert np.isfinite(x).all()
