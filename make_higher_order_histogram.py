import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import argparse
from utils import get_logger, pos_direct


def parse_args():
    parser = argparse.ArgumentParser(description="Draw higher order histogram.")

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
    axistour_embed = ica_embed[:, axistour]

    # random
    random_idx = np.random.permutation(dim)
    random_embed = ica_embed[:, random_idx]
    random_sign = np.random.choice([-1, 1], size=dim)
    random_embed = random_embed * random_sign.reshape(1, -1)

    # TICA9
    tica_embed_path = f"output/tica_embeddings/tica_width9_{emb_type}.pkl"
    if not Path(tica_embed_path).exists():
        raise FileNotFoundError(f"{tica_embed_path} does not exist")
    with open(tica_embed_path, "rb") as f:
        tica9_embed, _ = pkl.load(f)
    tica9_embed = pos_direct(tica9_embed)

    # TICA75
    tica_embed_path = f"output/tica_embeddings/tica_width75_{emb_type}.pkl"
    if not Path(tica_embed_path).exists():
        raise FileNotFoundError(f"{tica_embed_path} does not exist")
    with open(tica_embed_path, "rb") as f:
        tica75_embed, _ = pkl.load(f)
    tica75_embed = pos_direct(tica75_embed)

    logger.info("computing higher order correlation")
    ho_corrs_list = []
    for emb_name, embed in zip(
        ["Random Order", "Axis Tour", "TICA9", "TICA75"],
        [random_embed, axistour_embed, tica9_embed, tica75_embed],
    ):
        ho_corrs = []
        for i in range(dim):
            energy_corr = np.mean(embed[:, i] ** 2 * embed[:, (i + 1) % dim] ** 2)
            ho_corrs.append(energy_corr)
        ho_corrs_list.append((ho_corrs, emb_name))

    argsort = np.argsort([np.mean(ho_corrs) for ho_corrs, _ in ho_corrs_list])
    ranks = np.argsort(argsort)

    # draw higher order histogram
    logger.info("drawing higher order correlation histogram")
    fig, ax = plt.subplots(figsize=(10, 6))
    xmin = min([np.min(ho_corrs) for ho_corrs, _ in ho_corrs_list])
    xmax = max([np.max(ho_corrs) for ho_corrs, _ in ho_corrs_list])
    bins = np.linspace(0.99 * xmin, 1.01 * xmax, 50)

    emb_name2color = {
        "Random Order": "orange",
        "Axis Tour": "red",
        "TICA9": "lime",
        "TICA75": "gray",
    }

    for idx, (ho_corrs, emb_name) in enumerate(ho_corrs_list):
        r = ranks[idx]
        ax.hist(
            ho_corrs,
            bins=bins,
            label=emb_name,
            alpha=0.5,
            color=emb_name2color[emb_name],
            density=True,
        )
        mean = np.mean(ho_corrs)
        ax.axvline(
            mean, color=emb_name2color[emb_name], linestyle="dashed", linewidth=2
        )
        dx = 0.2 * (-1) ** (r < 2) * (abs(r - 1.5) + 1.0 + 1.0 * int(0 < r < 3))
        y = 2 + 2 * int(0 < r < 3)
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
    ax.set_xlabel("Higher-Order Correlation", fontsize=fs)
    ax.set_ylabel("Density", fontsize=fs)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=fs)

    # tick fontsize
    ts = 25
    ax.tick_params(labelsize=ts)

    # adjust margin
    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.98)

    # save fig
    output_dir = Path("output/images/tica")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tica_ho_histogram_{emb_type}_top{topk}.pdf"
    fig.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
