import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
from numpy.random import uniform
from numpy_tica import _g_sqrt
from tica import TopographicICA

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Save TICA embeddings.")

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--width", type=int, default=9)

    return parser.parse_args()


def generate_h(n_components, width, diag_value, off_diag_value, n_convolves):
    """Generate the neighborhood matrix h."""
    h_filter = np.zeros((n_components, n_components))
    for i in range(n_components):
        # To consider the left and right neighbors
        for j in range(-(width // 2), 1 + (width // 2)):
            h_filter[i, (i + j) % n_components] = off_diag_value
    h_filter += np.eye(n_components) * (diag_value - off_diag_value)
    h = h_filter
    for _ in range(n_convolves):
        h = h @ h_filter
    return h


def main():
    logger = get_logger()

    args = parse_args()
    logger.info(args)
    emb_type = args.emb_type
    assert emb_type in ("glove", "word2vec", "bert")
    seed = args.seed
    max_iter = args.max_iter
    width = args.width

    output_dir = Path("output/tica_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(f"output/pca_ica_embeddings/pca_ica_{emb_type}.pkl")
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")
    with open(input_path, "rb") as f:
        pca_embed, _, all_words = pkl.load(f)
    _, dim = pca_embed.shape

    n_components = dim
    n_dims = dim
    h = generate_h(
        n_components, width, diag_value=1.0, off_diag_value=1.0, n_convolves=2
    )

    np.random.seed(seed)
    w_init = uniform(-1.0, 1.0, (n_components, n_dims))

    tica = TopographicICA(
        n_components=n_components,
        max_iter=max_iter,
        fun=_g_sqrt,
        w_init=w_init,
        verbose=True,
    )
    tica_embed = tica.fit_transform(pca_embed, h)

    tica_words = (tica_embed, all_words)
    tica_words_path = output_dir / f"tica_width{width}_{emb_type}.pkl"
    logger.info(f"tica_words_path: {tica_words_path}")
    with open(tica_words_path, "wb") as f:
        pkl.dump(tica_words, f)


if __name__ == "__main__":
    main()
