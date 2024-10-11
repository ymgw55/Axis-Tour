import argparse
import pickle as pkl
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from utils import get_logger

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Draw 3D figure.")

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--start_axis_index", type=int, default=89)

    return parser.parse_args()


class Annotation3D(Annotation):
    """Annotate the point xyz with text s"""

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, s, xy=(0, 0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy = (xs, ys)
        Annotation.draw(self, renderer)


def annotate3D(ax, s, *args, **kwargs):
    """add anotation text s to to Axes3d ax"""

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)


def main():
    # seed
    np.random.seed(0)

    args = parse_args()
    emb_type = args.emb_type
    topk = args.topk
    start_axis_index = args.start_axis_index
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
    assert 0 <= start_axis_index < dim - 2
    skews = np.mean(axistour_embed**3, axis=0)
    normed_axistour_embed = axistour_embed / np.linalg.norm(
        axistour_embed, axis=1, keepdims=True
    )

    colormap = {
        0: "magenta",
        1: "lime",
        2: "cyan",
    }

    alpha = 1 / 3

    lb = start_axis_index
    ub = start_axis_index + 3
    sub_emb = axistour_embed[:, lb:ub]
    sub_skews = skews[lb:ub]

    # normed projection
    proj_direction = (sub_skews**alpha).reshape(-1, 1)
    proj_direction = proj_direction / np.linalg.norm(proj_direction)
    proj_emb = np.dot(sub_emb, proj_direction).flatten()
    proj_direction = proj_direction.flatten()

    idx2top5_ids = {}

    # calculate top-5
    all_top5_ids = set()
    for idx in range(lb, ub):
        xs = normed_axistour_embed[:, idx]
        top5_ids = np.argsort(-xs)[:5]
        idx2top5_ids[idx] = top5_ids
        all_top5_ids |= set(top5_ids)

    # select random words from all words, excluding the top 5 words.
    cands = [id_ for id_ in range(len(words)) if id_ not in all_top5_ids]
    random_ids = np.random.choice(cands, 10000, replace=False)

    logger.info("Draw 3D figure")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(5, 15)

    fs = 15
    ls = 25
    ds = 25

    # draw random words
    xs = []
    ys = []
    zs = []
    colors = []
    for random_id in random_ids:
        x, y, z = sub_emb[random_id]
        arg_max = np.argmax([x, y, z])
        xs.append(x)
        ys.append(y)
        zs.append(z)
        colors.append(colormap[arg_max])

    ax.scatter(xs, ys, zs, c=colors, s=ds, alpha=0.1, zorder=0)

    # draw top-5 words
    L = 0
    for idx in range(lb, ub):
        for jdx, id_ in enumerate(idx2top5_ids[idx]):
            x, y, z = sub_emb[id_]
            ax.scatter(
                x, y, z, c=colormap[idx - lb], s=2 * ds, edgecolors="k", linewidths=0.5
            )

            L = max(L, x, y, z)
            if jdx != 0 and jdx != 1:
                continue

            if jdx == 0:
                annotate3D(
                    ax,
                    s=words[id_],
                    xyz=(x, y, z + 0.25),
                    fontsize=fs,
                    xytext=(-3, 3),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    color="k",
                    zorder=100,
                )
            else:
                annotate3D(
                    ax,
                    s=words[id_],
                    xyz=(x, y, z + 0.25),
                    fontsize=fs,
                    xytext=(3, 3),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    color="k",
                    zorder=100,
                )

    # draw lines from origin to projected points direction
    ax.quiver(
        0,
        0,
        0,
        np.sqrt(3) * L * proj_direction[0],
        np.sqrt(3) * L * proj_direction[1],
        np.sqrt(3) * L * proj_direction[2],
        color="orange",
        arrow_length_ratio=0.15,
        linewidths=3,
        linestyles="-",
        alpha=0.9,
    )
    ax.text(
        1.1 * np.sqrt(3) * L * proj_direction[0],
        1.1 * np.sqrt(3) * L * proj_direction[1],
        1.1 * np.sqrt(3) * L * proj_direction[2],
        "Projection\ndirection",
        size=ls,
        zorder=1,
        color="orange",
        ha="left",
        weight="bold",
    )

    # draw projected points for top-5 words
    for idx in range(lb, ub):
        for jdx, id_ in enumerate(idx2top5_ids[idx]):
            if jdx != 0:
                continue
            l_ = proj_emb[id_]

            x1, y1, z1 = sub_emb[id_]
            x2, y2, z2 = l_ * proj_direction

            # draw line
            ax.quiver(
                x1,
                y1,
                z1,
                x2 - x1,
                y2 - y1,
                z2 - z1,
                color="orange",
                arrow_length_ratio=0,
                linewidths=2,
                linestyles="--",
                alpha=0.9,
            )

            ax.scatter(
                l_ * proj_direction[0],
                l_ * proj_direction[1],
                l_ * proj_direction[2],
                c=colormap[idx - lb],
                s=5 * ds,
                edgecolors="black",
                linewidths=0.5,
                marker="^",
            )

    # draw axis
    ax.quiver(
        0,
        0,
        0,
        0,
        0,
        1.1 * L,
        color="black",
        arrow_length_ratio=0.1,
        linewidths=2,
        alpha=0.75,
    )
    ax.quiver(
        0,
        0,
        0,
        0,
        1.1 * L,
        0,
        color="black",
        arrow_length_ratio=0.1,
        linewidths=2,
        alpha=0.75,
    )
    ax.quiver(
        0,
        0,
        0,
        1.1 * L,
        0,
        0,
        color="black",
        arrow_length_ratio=0.1,
        linewidths=2,
        alpha=0.75,
    )

    # draw axis name
    ax.text(0, 0, 1.2 * L, f"{lb+2}", size=fs, zorder=100, color="k", ha="center")
    ax.text(0, 1.2 * L, -0.5, f"{lb+1}", size=fs, zorder=100, color="k", ha="center")
    ax.text(1.4 * L, 0, -0.5, f"{lb}", size=fs, zorder=100, color="k", ha="center")

    # limit
    ax.set_xlim(-1, 1.3 * L)
    ax.set_ylim(-1, 1.3 * L)
    ax.set_zlim(-1, 1.3 * L)

    # tick size
    ax.tick_params(labelsize=fs)
    # label [0, 5, 10, 15]
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_zticks([0, 5, 10, 15])

    # adjust margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # save
    output_dir = Path("output/images/3d_figures")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = (
        output_dir / f"3d_figure_{emb_type}_top{topk}_axis{start_axis_index}.png"
    )
    logger.info(f"Save {output_path}")
    # tight_layout
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(output_path, dpi=150)


if __name__ == "__main__":
    main()
