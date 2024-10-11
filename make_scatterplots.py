import argparse
import pickle as pkl
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

from utils import calc_c_I

warnings.filterwarnings("ignore")


def plot_scatterplot(axistour_embed, topk, words, left_axis_index, length, output_path):
    n, dim = axistour_embed.shape
    normed_axistour_embed = axistour_embed / np.linalg.norm(
        axistour_embed, axis=1, keepdims=True
    )

    axis_idxs = np.array([left_axis_index + i for i in range(length)])
    axis_idx2axis_idx_idx = {axis_idx: idx for idx, axis_idx in enumerate(axis_idxs)}

    k = 5
    axis2top_word_ids = {}
    for axis_idx in axis_idxs:
        top_word_ids = np.argsort(normed_axistour_embed[:, axis_idx])[-k:]
        axis2top_word_ids[axis_idx] = top_word_ids

    print(f"axis indexes: {axis_idxs}")
    proj_matrix = []
    for idx in range(length - 1):
        theta = np.pi * idx / (length - 1)
        proj_matrix.append((np.cos(theta), np.sin(theta)))
    proj_matrix.append((-1, 0))
    proj_matrix = np.array(proj_matrix)

    picked_emb = normed_axistour_embed[:, axis_idxs]  # (n, length)

    word_idx2axis_idx_idx = {}
    axis_idx2word_idx_list = defaultdict(list)
    for i in range(n):
        max_idx = np.argmax(picked_emb[i])
        word_idx2axis_idx_idx[i] = max_idx
        axis_idx2word_idx_list[left_axis_index + max_idx].append(i)
    for axis_idx in axis_idxs:
        assert len(axis_idx2word_idx_list[axis_idx]) > 0
        axis_idx2word_idx_list[axis_idx] = np.array(axis_idx2word_idx_list[axis_idx])

    # make color map for each axis, total length colors
    color_map = {}
    for i in range(length):
        color_map[i] = plt.cm.get_cmap("rainbow")(i / length)

    proj_emb = np.dot(picked_emb, proj_matrix)  # (n, 2)

    max_x = np.max(np.abs(proj_emb[:, 0]))
    max_y = np.max(np.abs(proj_emb[:, 1]))
    max_x = max(max_x, max_y)
    max_y = max_x

    fig, axes = plt.subplots(2, figsize=(11, 10))

    for ax_idx in range(2):
        if ax_idx == 0:
            print("\nAxis Tour")
        else:
            print("\nSkew Sort")

        ax = axes[ax_idx]

        if ax_idx == 1:
            picked_emb = normed_axistour_embed[:, axis_idxs]
            picked_skews = np.sum(axistour_embed[:, axis_idxs] ** 3, axis=0)
            picked_skew_sort_idex = np.argsort(-picked_skews)
            picked_emb = picked_emb[:, picked_skew_sort_idex]
            proj_matrix = []
            for idx in range(length - 1):
                theta = 2 * np.pi - np.pi * idx / (length - 1)
                proj_matrix.append((np.cos(theta), np.sin(theta)))
            proj_matrix.append((-1, 0))
            proj_matrix = np.array(proj_matrix)
            proj_emb = np.dot(picked_emb, proj_matrix)  # (n, 2)
            axis_idxs = axis_idxs[picked_skew_sort_idex]

        for idx, axis_idx in enumerate(axis_idxs):
            if ax_idx == 0:
                if idx < len(axis_idxs) - 1:
                    theta = np.pi * idx / (length - 1)
                else:
                    theta = np.pi
            else:
                if idx < len(axis_idxs) - 1:
                    theta = 2 * np.pi - np.pi * idx / (length - 1)
                else:
                    theta = np.pi

            x, y = np.cos(theta), np.sin(theta)
            x *= max_x
            y *= max_y
            point = {"start": (0, 0), "end": (x, y)}
            ax.annotate(
                "",
                xytext=point["start"],
                xy=point["end"],
                arrowprops=dict(
                    shrink=0,
                    width=2,
                    headwidth=7,
                    headlength=7,
                    connectionstyle="arc3",
                    facecolor=color_map[axis_idx2axis_idx_idx[axis_idx]],
                    edgecolor="black",
                    linewidth=0.5,
                ),
            )
            ax.text(
                1.05 * x,
                1.05 * y,
                f"{axis_idx}",
                fontsize=18,
                ha="center",
                va="center",
                color=color_map[axis_idx2axis_idx_idx[axis_idx]],
            )

        word_idxs = np.argsort(np.linalg.norm(picked_emb, axis=1))
        xs = proj_emb[word_idxs, 0]
        ys = proj_emb[word_idxs, 1]
        colors = [color_map[word_idx2axis_idx_idx[word_idx]] for word_idx in word_idxs]
        ax.scatter(xs, ys, s=10, marker="o", color=colors, alpha=0.5)

        texts = []
        lens = []
        for axis_idx in axis_idxs:
            idx = axis_idx2axis_idx_idx[axis_idx]
            top_word_ids = axis2top_word_ids[axis_idx]
            for id_ in top_word_ids:
                x, y = proj_emb[id_]
                if ax_idx == 0:
                    if y < 0:
                        continue
                else:
                    if y > 0:
                        continue
                lens.append(np.linalg.norm((x, y)))
                ax.scatter(
                    x,
                    y,
                    s=30,
                    marker="o",
                    color=color_map[idx],
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=11,
                )

                texts.append(
                    ax.text(
                        x,
                        y,
                        words[id_],
                        fontsize=13,
                        color="black",
                        bbox=dict(
                            facecolor=color_map[idx],
                            boxstyle="round,pad=0.1",
                            edgecolor="gray",
                            linewidth=1,
                        ),
                    )
                )

        print(f"d_I: {np.mean(lens):.2f}")
        print(f"c_I: {calc_c_I(picked_emb, normed_axistour_embed, topk): .2f}")
        ax.set_xlim(-max_x * 1.1, max_x * 1.1)

        if ax_idx == 0:
            ax.set_ylim(0, max_y * 1.1)
        else:
            ax.set_ylim(-max_y * 1.1, 0.0)
        ax.axis("off")
        ax.set_aspect("equal")

        # adjust text
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="k", lw=0.5, zorder=10),
            force_pull=(0.05, 0.05),
            expand=(1.1, 1.1),
        )

        for text in texts:
            text.set_zorder(10)

        if length == 9:
            scale_y = 0.9
        elif length == 10:
            scale_y = 0.98
        else:
            raise NotImplementedError

        if ax_idx == 0:
            ax.text(
                -1.15 * max_x,
                scale_y * max_y,
                "Axis Tour",
                fontsize=25,
                ha="left",
                va="center",
                fontweight="bold",
            )
        else:
            ax.text(
                -1.15 * max_x,
                -scale_y * max_y,
                "Skewness Sort",
                fontsize=25,
                ha="left",
                va="center",
                fontweight="bold",
            )

    fig.tight_layout()
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.05
    )
    fig.savefig(output_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Scatterplot for Axis Tour.")

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--left_axis_index", type=int, default=86)
    parser.add_argument("--length", type=int, default=9)
    return parser.parse_args()


def main():
    args = parse_args()
    emb_type = args.emb_type
    topk = args.topk
    left_axis_index = args.left_axis_index
    length = args.length

    axistour_embed_path = (
        f"output/axistour_embeddings/axistour_top{topk}_{emb_type}.pkl"
    )
    if not Path(axistour_embed_path).exists():
        raise FileNotFoundError(f"{axistour_embed_path} does not exist")
    with open(axistour_embed_path, "rb") as f:
        axistour_embed, words = pkl.load(f)

    output_dir = Path("output/images/scatterplots")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = (
        output_dir / f"scatterplot_{emb_type}_top{topk}_"
        f"left{left_axis_index}_length{length}.png"
    )

    plot_scatterplot(axistour_embed, topk, words, left_axis_index, length, output_path)


if __name__ == "__main__":
    main()
