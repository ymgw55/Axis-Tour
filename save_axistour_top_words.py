import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save top words for Axis Tour embeddings."
    )

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--top_words", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    emb_type = args.emb_type
    topk = args.topk
    top_words = args.top_words
    logger = get_logger()
    logger.info(args)

    axistour_embed_path = (
        f"output/axistour_embeddings/axistour_top{topk}_{emb_type}.pkl"
    )
    if not Path(axistour_embed_path).exists():
        raise FileNotFoundError(f"{axistour_embed_path} does not exist")
    logger.info(f"Load {axistour_embed_path}")
    with open(axistour_embed_path, "rb") as f:
        axistour_embed, words = pkl.load(f)
    _, dim = axistour_embed.shape

    logger.info("show top5 words for each axis:")
    normed_axistour_embed = axistour_embed / np.linalg.norm(
        axistour_embed, axis=1, keepdims=True
    )
    data = []
    for idx in range(dim):
        topk_ids = np.argsort(-normed_axistour_embed[:, idx])[:top_words]
        row = dict()
        row["axis_idx"] = idx
        for idx, word in enumerate(words[topk_ids]):
            row[f"top{idx+1}_word"] = word
        logger.info(row)
        data.append(row)

    df = pd.DataFrame(data)
    output_dir = Path("output/axistour_top_words")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{emb_type}_top{topk}-top{top_words}_words.csv"
    logger.info(output_path)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
