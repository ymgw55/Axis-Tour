import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import argparse
from utils import get_logger, pos_direct


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw histogram and scatter plot for cossims."
    )

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--topk", type=int, default=100)

    return parser.parse_args()


def main():
    logger = get_logger()

    args = parse_args()
    emb_type = args.emb_type
    topk = args.topk

    # seed
    np.random.seed(0)

    input_path = f"output/pca_ica_embeddings/pca_ica_{emb_type}.pkl"
    logger.info(f"loading embeddings from {input_path}")
    with open(input_path, "rb") as f:
        _, ica_embed, _ = pkl.load(f)
    ica_embed = pos_direct(ica_embed)
    n, dim = ica_embed.shape
    logger.info(f"ica_embed.shape: {ica_embed.shape}")

    normed_ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)

    # axis tour
    axis_tour_path = f"LKH-3.0.6-{emb_type}/axistour.top{topk}.txt"
    logger.info(f"loading axis tour from {axis_tour_path}")
    axistour = []
    with open(axis_tour_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            idx = int(line[len("axis") :])
            axistour.append(idx)
    normed_axis_tour_embed = normed_ica_embed[:, axistour]

    # random
    random_idx = np.random.permutation(dim)
    normed_random_embed = normed_ica_embed[:, random_idx]
    random_sign = np.random.choice([-1, 1], size=dim)
    normed_random_embed = normed_random_embed * random_sign.reshape(1, -1)

    # TICA9
    tica_embed_path = f"output/tica_embeddings/tica_width9_{emb_type}.pkl"
    if not Path(tica_embed_path).exists():
        raise FileNotFoundError(f"{tica_embed_path} does not exist")
    with open(tica_embed_path, "rb") as f:
        tica9_embed, _ = pkl.load(f)
    tica9_embed = pos_direct(tica9_embed)
    normed_tica9_embed = tica9_embed / np.linalg.norm(
        tica9_embed, axis=1, keepdims=True
    )

    # TICA75
    tica_embed_path = f"output/tica_embeddings/tica_width75_{emb_type}.pkl"
    if not Path(tica_embed_path).exists():
        raise FileNotFoundError(f"{tica_embed_path} does not exist")
    with open(tica_embed_path, "rb") as f:
        tica75_embed, _ = pkl.load(f)
    tica75_embed = pos_direct(tica75_embed)
    normed_tica75_embed = tica75_embed / np.linalg.norm(
        tica75_embed, axis=1, keepdims=True
    )

    logger.info("computing cosine similarity")
    cossims_list = []
    for emb_name, normed_embed in zip(
        ["Random Order", "Axis Tour", "TICA9", "TICA75"],
        [
            normed_random_embed,
            normed_axis_tour_embed,
            normed_tica9_embed,
            normed_tica75_embed,
        ],
    ):
        vecs = []
        for axis_idx in range(dim):
            indices = np.argsort(normed_embed[:, axis_idx])[-topk:]
            topk_embeds = normed_embed[indices]
            axis_embed = topk_embeds.mean(axis=0)
            vecs.append(axis_embed)
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

        cossims_list.append((cossims, emb_name))

    argsort = np.argsort([np.mean(cossims) for cossims, _ in cossims_list])
    ranks = np.argsort(argsort)

    # draw cossims histgram
    logger.info("drawing cossims histgram")
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-0.2, 1.0, 50)

    emb_name2color = {
        "Random Order": "orange",
        "Axis Tour": "red",
        "TICA9": "lime",
        "TICA75": "gray",
    }

    for idx, (cossims, emb_name) in enumerate(cossims_list):
        r = ranks[idx]
        ax.hist(
            cossims,
            bins=bins,
            label=emb_name,
            alpha=0.5,
            color=emb_name2color[emb_name],
            density=True,
        )
        mean = np.mean(cossims)
        ax.axvline(
            mean, color=emb_name2color[emb_name], linestyle="dashed", linewidth=2
        )
        dx = (
            0.1
            * (-1) ** (r < 2)
            * (abs(r - 1.5) + 0.5 * int(0 < r < 3) - 0.25 * int(r == 1))
        )
        y = 4 + 4 * int(0 < r < 3)
        ax.text(
            mean + dx,
            y,
            f"${mean:.3f}$",
            color=emb_name2color[emb_name],
            fontsize=25,
            ha="center",
        )

    # label fontsize
    fs = 25
    ax.set_xlabel("Cosine Similarity", fontsize=fs)
    ax.set_ylabel("Density", fontsize=fs)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=fs)

    # tick fontsize
    ts = 25
    ax.tick_params(labelsize=ts)

    # adjust margin
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)

    # save fig
    output_dir = Path("output/images/tica")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tica_histogram_{emb_type}_top{topk}.pdf"
    fig.savefig(output_path)
    plt.close()

    # scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(dim)
    ls = 13

    for idx, (cossims, emb_name) in enumerate(cossims_list):
        r = ranks[idx]
        color = emb_name2color[emb_name]
        ax.scatter(xs, cossims, label=emb_name, alpha=0.5, color=color)
        mean = np.mean(cossims)
        ax.axhline(mean, linestyle="dashed", color=color, linewidth=2)
        if r == 3:
            dx = 0.04
        else:
            dx = -0.08
        ax.text(-23, mean + dx, f"${mean:.3f}$", color=color, fontsize=ls)

    # label fontsize
    fs = 20
    ax.set_ylabel("Cosine Similarity", fontsize=fs)
    ax.set_xlabel("Axis", fontsize=fs)

    # tick fontsize
    ts = 20
    ax.tick_params(labelsize=ts)

    # limit
    ax.set_xlim(-25, dim + 5)
    ax.set_ylim(-0.19, 1.19)

    ax.legend(loc="upper right", fontsize=17, ncol=4)

    # adjust margin
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.98)

    # save fig
    output_path = output_dir / f"tica_scatter_{emb_type}_top{topk}.pdf"
    fig.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
