import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import argparse
from utils import get_logger, pos_direct


def parse_args():
    parser = argparse.ArgumentParser(description="Draw cossim histogram.")

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

    # calculate skewness before normalizing ICA-transformed embeddings
    skew_sort_idex = np.argsort(-np.mean(ica_embed**3, axis=0))

    normed_ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)

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

    # skew sort
    normed_skew_embed = normed_ica_embed[:, skew_sort_idex]

    # random
    random_idx = np.random.permutation(dim)
    normed_random_embed = normed_ica_embed[:, random_idx]
    random_sign = np.random.choice([-1, 1], size=dim)
    normed_random_embed = normed_random_embed * random_sign.reshape(1, -1)

    logger.info("computing cosine similarity")
    cossims_list = []
    # random sample
    rand_sample_idx = np.random.choice(n, dim, replace=False)
    rand_sample_embed = []
    for idx in rand_sample_idx:
        rand_sample_embed.append(normed_random_embed[idx])
    rand_sample_embed.append(rand_sample_embed[0])
    cossims = []
    for i in range(len(rand_sample_embed) - 1):
        cossim = (
            np.dot(rand_sample_embed[i], rand_sample_embed[i + 1])
            / np.linalg.norm(rand_sample_embed[i])
            / np.linalg.norm(rand_sample_embed[i + 1])
        )
        cossims.append(cossim)
    cossims_list.append((cossims, f"{dim} samples"))

    for emb_name, normed_embed in zip(
        ["Random Order", "Skewness Sort", "Axis Tour"],
        [normed_random_embed, normed_skew_embed, normed_axis_tour_embed],
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
    bins = np.linspace(-0.2, 0.8, 50)

    emb_name2color = {
        f"{dim} samples": "blue",
        "Random Order": "orange",
        "Skewness Sort": "green",
        "Axis Tour": "red",
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
            * (abs(r - 1.5) + 0.5 * int(0 < r < 3) + 0.25 * int(r < 3))
        )
        y = 4 + 2 * int(0 < r < 3)
        ax.text(
            mean + dx,
            y,
            f"${mean:.3f}$",
            color=emb_name2color[emb_name],
            fontsize=25,
            ha="center",
        )

    # plot normal distribution
    mu = 0
    sigma2 = 1 / dim
    x = np.linspace(-0.25, 0.8, 100)
    y = np.exp(-((x - mu) ** 2) / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
    ax.plot(x, y, label=r"$\mathcal{N}(0, 1/300)$", color="black", linewidth=5)

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
    output_dir = Path("output/images/cossim_histogram")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ccossim_histogram_{emb_type}_top{topk}.pdf"
    fig.savefig(output_path)


if __name__ == "__main__":
    main()
