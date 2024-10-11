import argparse
import pickle as pkl
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils import get_logger

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare k.")
    parser.add_argument("--emb_type", type=str, default="glove")

    return parser.parse_args()


def main():
    args = parse_args()
    emb_type = args.emb_type
    assert emb_type in ("glove", "word2vec")
    logger = get_logger()
    logger.info(args)

    topks = [1, 10, 100, 1000]
    k1_to_mean_cossims = {}
    for k1 in topks:
        # load embeddings
        axistour_embed_path = (
            f"output/axistour_embeddings/axistour_top{k1}_{emb_type}.pkl"
        )
        if not Path(axistour_embed_path).exists():
            raise FileNotFoundError(f"{axistour_embed_path} does not exist")
        logger.info(f"loading embeddings from {axistour_embed_path}")
        with open(axistour_embed_path, "rb") as f:
            axistour_embed, _ = pkl.load(f)
        _, dim = axistour_embed.shape

        normed_axistour_embed = axistour_embed / np.linalg.norm(
            axistour_embed, axis=1, keepdims=True
        )

        mean_cossims = []
        for k2 in topks:
            vecs = []
            for axis_idx in range(dim):
                indices = np.argsort(-normed_axistour_embed[:, axis_idx])[:k2]
                top_embeds = normed_axistour_embed[indices]
                axis_emb = top_embeds.mean(axis=0)
                vecs.append(axis_emb)
            fisrt_vec = vecs[0]
            vecs.append(fisrt_vec)

            cossims = []
            for i in range(len(vecs) - 1):
                cossim = (
                    np.dot(vecs[i], vecs[i + 1])
                    / np.linalg.norm(vecs[i])
                    / np.linalg.norm(vecs[i + 1])
                )
                cossims.append(cossim)
            mean_cossim = np.mean(cossims)
            mean_cossims.append(mean_cossim)
            logger.info(f"k1={k1}, k2={k2}, mean_cossim={mean_cossim:.3f}")
        k1_to_mean_cossims[k1] = mean_cossims

    # plot
    fig, ax = plt.subplots(figsize=(12, 12))
    fs = 35
    ls = 28
    lw = 3
    ms = 20

    top1_color = "limegreen"
    top10_color = "gray"
    top100_color = "red"
    top1000_color = "sandybrown"

    colors = {
        1: top1_color,
        10: top10_color,
        100: top100_color,
        1000: top1000_color,
    }
    means = []
    for k, mean_cossims in k1_to_mean_cossims.items():
        mean_cossims = np.array(mean_cossims)
        ax.plot(
            topks,
            mean_cossims,
            label=r"$C_{" + f"{k}" + r"}(k)$",
            linewidth=lw,
            marker="o",
            markersize=ms,
            color=colors[k],
        )
        means.append(np.mean(mean_cossims))

    logger.info(f"M_k={means}")
    means = np.array(means)
    ax.plot(
        topks,
        means,
        label=r"$M(k)$",
        linewidth=lw,
        marker="^",
        markersize=ms,
        linestyle="--",
        color="deepskyblue",
    )

    ax.set_xlabel(r"$k$", fontsize=int(1.5 * fs))
    ax.set_xscale("log")
    ax.tick_params(labelsize=int(1.5 * ls))

    ax.legend(fontsize=ls, loc="lower left")
    fig.tight_layout()

    output_dir = Path("output/images/comparing_k")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"comparing_k_{emb_type}.pdf"
    logger.info(f"Save figure to {output_path}")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
