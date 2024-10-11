import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.stats import spearmanr
from utils import get_logger, pos_direct


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show the relation between skewness and two cossims."
    )

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--topk", type=int, default=100)

    return parser.parse_args()


def main():
    logger = get_logger()

    args = parse_args()
    emb_type = args.emb_type
    topk = args.topk

    # axis tour
    axistour_embed_path = (
        f"output/axistour_embeddings/axistour_top{topk}_{emb_type}.pkl"
    )
    if not Path(axistour_embed_path).exists():
        raise FileNotFoundError(f"{axistour_embed_path} does not exist")
    logger.info(f"loading embeddings from {axistour_embed_path}")
    with open(axistour_embed_path, "rb") as f:
        axistour_embed, _ = pkl.load(f)

    # skew sort
    input_path = f"output/pca_ica_embeddings/pca_ica_{emb_type}.pkl"
    logger.info(f"loading embeddings from {input_path}")
    with open(input_path, "rb") as f:
        _, ica_embed, _ = pkl.load(f)
    ica_embed = pos_direct(ica_embed)
    n, dim = ica_embed.shape
    logger.info(f"ica_embed.shape: {ica_embed.shape}")
    skew_sort_idex = np.argsort(-np.mean(ica_embed**3, axis=0))
    skew_embed = ica_embed[:, skew_sort_idex]

    skew_color = "dodgerblue"
    cos_color = "deeppink"
    ls = 25
    fs = 20
    ds = 40
    lw = 2.5
    alpha = 0.85
    plt.rcParams["font.size"] = 18

    output_dir = Path("output/images/skew_two_cossims")
    output_dir.mkdir(exist_ok=True, parents=True)

    for embed, emb_name in [(axistour_embed, "axistour"), (skew_embed, "skewsort")]:
        skews = np.mean(embed**3, axis=0)
        normed_emb = embed / np.linalg.norm(embed, axis=1, keepdims=True)

        vecs = []
        for axis_idx in range(dim):
            indices = np.argsort(normed_emb[:, axis_idx])[-topk:]
            topk_embeds = normed_emb[indices]
            axis_embed = topk_embeds.mean(axis=0)
            vecs.append(axis_embed)

        cossims = []
        for i in range(len(vecs) - 1):
            cossim = (
                np.dot(vecs[i], vecs[i + 1])
                / np.linalg.norm(vecs[i])
                / np.linalg.norm(vecs[i + 1])
            )
            cossims.append(cossim)
        edge_cossim = (
            np.dot(vecs[0], vecs[-1])
            / np.linalg.norm(vecs[0])
            / np.linalg.norm(vecs[-1])
        )
        cossims.insert(0, edge_cossim)
        cossims.append(edge_cossim)

        # plot skew and cos in one figure
        fig, axes = plt.subplots(
            1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.5, 1]}
        )

        # adjust text
        fig.subplots_adjust(
            left=0.07, right=0.95, bottom=0.15, top=0.93, wspace=0.4, hspace=0.05
        )

        ax2 = axes[0]
        ax1 = ax2.twinx()
        skew_x = np.arange(dim)
        ax1.plot(skew_x, skews, color=skew_color, linewidth=lw, alpha=alpha)

        mean_two_cossims = []
        for i in range(len(cossims) - 1):
            mean_two_cossims.append((cossims[i] + cossims[i + 1]) / 2)
        ax2.plot(skew_x, mean_two_cossims, color=cos_color, linewidth=lw, alpha=alpha)
        ax2.set_xlabel("Axis", fontsize=ls)
        ax2.tick_params(axis="x", labelsize=fs)
        ax1.set_ylabel(
            "Skewness", color=skew_color, rotation=270, fontsize=ls, labelpad=22
        )
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax2.set_ylabel("Average of Two Cosines", color=cos_color, fontsize=ls)
        ax1.tick_params(axis="y", labelcolor=skew_color, labelsize=fs)
        ax2.tick_params(axis="y", labelcolor=cos_color, labelsize=fs)

        r = spearmanr(skews, mean_two_cossims)[0]
        logger.info(f"{emb_type}, {emb_name}, spearmanr: {r:.2f}")

        # plot skew and cos scatter
        cm = plt.cm.get_cmap("hsv")
        ax = axes[1]
        z = list(range(dim))
        ax.scatter(
            skews,
            mean_two_cossims,
            c=z,
            cmap=cm,
            vmin=0,
            vmax=dim,
            s=ds,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_xlabel("Skewness", color=skew_color, fontsize=ls)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.tick_params(axis="x", labelcolor=skew_color, labelsize=fs)
        ax.set_ylabel("Average of Two Cosines", color=cos_color, fontsize=ls)
        ax.tick_params(axis="y", labelcolor=cos_color, labelsize=fs)

        # add colorbar
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Axis", rotation=270, labelpad=28, fontsize=ls)
        cbar.ax.tick_params(labelsize=fs)

        output_path = output_dir / f"{emb_type}_top{topk}_{emb_name}.png"
        logger.info(f"Save {output_path}")
        plt.savefig(output_path, dpi=150)


if __name__ == "__main__":
    main()
