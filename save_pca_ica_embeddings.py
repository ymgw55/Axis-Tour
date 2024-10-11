import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, FastICA

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Save PCA and ICA embeddings.")

    parser.add_argument("--emb_type", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-10)

    return parser.parse_args()


def main():
    logger = get_logger()

    args = parse_args()
    logger.info(args)
    emb_type = args.emb_type
    assert emb_type in ("glove", "word2vec", "bert")
    seed = args.seed
    max_iter = args.max_iter
    tol = args.tol

    output_dir = Path("output/pca_ica_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    if emb_type == "glove":
        input_path = Path("output/raw_embeddings/raw_glove.pkl")

        if input_path.exists():
            with open(input_path, "rb") as f:
                all_embeddings, all_words = pkl.load(f)
        else:
            txt_input_path = Path("data/embeddings/glove.6B/glove.6B.300d.txt")
            with open(txt_input_path, "r") as f:
                lines = f.readlines()
                all_words = []
                all_embeddings = []
                for line in lines:
                    word, *embedding = line.split()
                    all_words.append(word)
                    all_embeddings.append(embedding)
                all_words = np.array(all_words)
                all_embeddings = np.array(all_embeddings)

            with open(input_path, "wb") as f:
                pkl.dump((all_embeddings, all_words), f)

    elif emb_type == "word2vec":
        input_path = Path("output/raw_embeddings/raw_word2vec.pkl")
        with open(input_path, "rb") as f:
            all_embeddings, all_words = pkl.load(f)

    elif emb_type == "bert":
        input_path = Path("output/raw_embeddings/raw_bert.pkl")
        with open(input_path, "rb") as f:
            all_embeddings, all_words, _ = pkl.load(f)

    all_embeddings = all_embeddings.astype(np.float64)
    logger.info(f"all_embeddings.shape: {all_embeddings.shape}")
    # centering
    all_embeddings -= all_embeddings.mean(axis=0)

    # PCA
    rng = np.random.RandomState(seed)
    pca_params = {"random_state": rng}
    logger.info("pca_params: {}".format(pca_params))
    pca = PCA(**pca_params)
    pca_embed = pca.fit_transform(all_embeddings)
    pca_embed = pca_embed / pca_embed.std(axis=0)

    # ICA
    ica_params = {
        "n_components": None,
        "random_state": rng,
        "max_iter": max_iter,
        "tol": tol,
        "whiten": False,  # already whitened by PCA
    }
    logger.info("ica_params: {}".format(ica_params))
    ica = FastICA(**ica_params)
    ica.fit(pca_embed)
    R = ica.components_.T
    ica_embed = pca_embed @ R

    pca_ica_words = (pca_embed, ica_embed, all_words)
    pca_ica_words_path = output_dir / f"pca_ica_{emb_type}.pkl"
    logger.info(f"pca_ica_words_path: {pca_ica_words_path}")
    with open(pca_ica_words_path, "wb") as f:
        pkl.dump(pca_ica_words, f)


if __name__ == "__main__":
    main()
