import argparse
import pickle as pkl
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from web.evaluate import evaluate_analogy, evaluate_categorization, evaluate_similarity

from utils import MyEmbedding, get_logger, get_tasks, pos_direct, split_range

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate dimensionality reduction methods for TICA."
    )

    parser.add_argument("--emb_type", type=str, default="glove")

    return parser.parse_args()


def main():
    args = parse_args()
    emb_type = args.emb_type
    assert emb_type in ("glove", "word2vec")
    logger = get_logger()
    logger.info(args)

    # seed
    np.random.seed(0)

    # TICA9
    tica_embed_path = f"output/tica_embeddings/tica_width9_{emb_type}.pkl"
    if not Path(tica_embed_path).exists():
        raise FileNotFoundError(f"{tica_embed_path} does not exist")
    with open(tica_embed_path, "rb") as f:
        tica9_embed, words = pkl.load(f)
    tica9_embed = pos_direct(tica9_embed)
    _, dim = tica9_embed.shape

    # TICA75
    tica_embed_path = f"output/tica_embeddings/tica_width75_{emb_type}.pkl"
    if not Path(tica_embed_path).exists():
        raise FileNotFoundError(f"{tica_embed_path} does not exist")
    with open(tica_embed_path, "rb") as f:
        tica75_embed, _ = pkl.load(f)
    tica75_embed = pos_direct(tica75_embed)

    emb_names = ("TICA9", "TICA9_curt", "TICA75", "TICA75_curt")

    analogy_tasks, similarity_tasks, categorization_tasks = get_tasks()

    data = []
    ps = [1, 2, 5, 10, 20, 50, 100, 200, 300]
    alpha = 1 / 3
    for p in ps:
        for emb_name in emb_names:
            logger.info(f"p={p}, emb_name={emb_name}")
            # load embedding

            if "TICA9" in emb_name:
                tmp_emb = tica9_embed
            elif "TICA75" in emb_name:
                tmp_emb = tica75_embed
            else:
                raise ValueError(f"Unknown emb_name: {emb_name}")

            if "_curt" in emb_name:
                skews = np.mean(tmp_emb**3, axis=0)
                signs = np.sign(skews)

                # I_r
                bounds = split_range(p, dim)

                compressed = []
                for lb, ub in bounds:
                    assert lb < ub
                    sub_emb = tmp_emb[:, lb:ub]
                    sub_skews = skews[lb:ub]
                    sub_signs = signs[lb:ub]

                    # f_r
                    proj_direction = (sub_signs * (sub_skews**alpha)).reshape(-1, 1)
                    proj_direction = proj_direction / np.linalg.norm(proj_direction)

                    # Tf_r
                    proj_emb = np.dot(sub_emb, proj_direction).flatten()
                    compressed.append(proj_emb)

                # TF
                compressed = np.stack(compressed, axis=1)
                # shape check
                assert compressed.shape == (len(words), p)
                w = MyEmbedding.from_words_and_vectors(words, compressed)

            else:
                w = MyEmbedding.from_words_and_vectors(words, tmp_emb[:, :p])

            # analogy tasks
            for task_name, task in analogy_tasks.items():
                category_set = sorted(list(set(task.category)))
                for c in category_set:
                    ids = np.where(task.category == c)[0]
                    X, y = task.X[ids], task.y[ids]
                    category = task.category[ids]
                    res = evaluate_analogy(w=w, X=X, y=y, category=category)
                    acc = dict(res.loc[c])["accuracy"]

                    row = {
                        "emb_name": emb_name,
                        "p": p,
                        "task_type": "analogy",
                        "task": c,
                        "top1-acc": acc,
                    }
                    logger.info(row)
                    data.append(row)

            # sim tasks
            for task_name, task in similarity_tasks.items():
                spearman = evaluate_similarity(w, task.X, task.y)
                if np.isnan(spearman):
                    spearman = 0
                row = {
                    "emb_name": emb_name,
                    "p": p,
                    "task_type": "similarity",
                    "task": task_name,
                    "spearman": spearman,
                }
                logger.info(row)
                data.append(row)

            # categorization tasks
            for task_name, task in categorization_tasks.items():
                purity = evaluate_categorization(w=w, X=task.X, y=task.y, seed=0)
                row = {
                    "emb_name": emb_name,
                    "p": p,
                    "task_type": "categorization",
                    "task": task_name,
                    "purity": purity,
                }
                logger.info(row)
                data.append(row)

    # save
    df = pd.DataFrame(data)
    output_dir = Path("output/eval_dimred")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{emb_type}_tica.csv"
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
