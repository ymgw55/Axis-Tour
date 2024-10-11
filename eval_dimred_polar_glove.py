import warnings

import gensim
import numpy as np
import pandas as pd
from web.evaluate import evaluate_analogy, evaluate_categorization, evaluate_similarity

from utils import MyEmbedding, get_logger, get_tasks

warnings.filterwarnings("ignore")


def main():
    logger = get_logger()

    analogy_tasks, similarity_tasks, categorization_tasks = get_tasks()

    vecs_list = []
    words_list = []

    emb_names = (
        "rand_antonym",
        "variance_antonymy",
        "orthogonal_antonymy",
    )

    for emb_name in emb_names:
        emb_path = f"output/polar_glove_embeddings/{emb_name}_gl_500_StdNrml.bin"
        gensim_vecs = gensim.models.KeyedVectors.load_word2vec_format(
            emb_path, binary=True
        )
        # convert gensim vectors to numpy array
        words = np.array(gensim_vecs.wv.index2word)
        vectors = gensim_vecs.vectors
        vecs_list.append(vectors)
        words_list.append(words)

    data = []
    ps = [1, 2, 5, 10, 20, 50, 100, 200, 300]
    for p in ps:
        for emb_name, vectors, words in zip(emb_names, vecs_list, words_list):
            logger.info(f"Processing {emb_name} with p={p}")

            w = MyEmbedding.from_words_and_vectors(words, vectors[:, :p])

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
    save_path = "output/eval_dimred/polar_glove.csv"
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
