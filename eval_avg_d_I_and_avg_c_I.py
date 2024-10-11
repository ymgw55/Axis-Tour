import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import calc_c_I


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate average d_I and average c_I."
    )

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--length", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    emb_type = args.emb_type
    topk = args.topk
    length = args.length

    axistour_embed_path = (
        f"output/axistour_embeddings/axistour_top{topk}_{emb_type}.pkl"
    )
    if not Path(axistour_embed_path).exists():
        raise FileNotFoundError(f"{axistour_embed_path} does not exist")
    with open(axistour_embed_path, "rb") as f:
        axistour_embed, _ = pkl.load(f)
    _, dim = axistour_embed.shape
    normed_axistour_embed = axistour_embed / np.linalg.norm(
        axistour_embed, axis=1, keepdims=True
    )

    # seed
    np.random.seed(0)

    k = 5
    axis2id = {}
    selected_ids = set()
    for i in range(dim):
        top_ids = np.argsort(normed_axistour_embed[:, i])[-k:]
        axis2id[i] = top_ids
        for id_ in top_ids:
            selected_ids.add(id_)

    d_I_list = [[] for _ in range(2)]
    c_I_list = [[] for _ in range(2)]

    # plot
    for left_axis_idx in tqdm(list(range(dim))):
        axis_idxs = np.array([(left_axis_idx + i) % dim for i in range(length)])

        proj_matrix = []
        for idx in range(length - 1):
            theta = np.pi * idx / (length - 1)
            proj_matrix.append((np.cos(theta), np.sin(theta)))
        proj_matrix.append((-1, 0))
        proj_matrix = np.array(proj_matrix)
        picked_emb = normed_axistour_embed[:, axis_idxs]
        proj_emb = np.dot(picked_emb, proj_matrix)

        for ax_idx in range(2):
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
                proj_emb = np.dot(picked_emb, proj_matrix)
                axis_idxs = axis_idxs[picked_skew_sort_idex]

            ds = []
            for axis_idx in axis_idxs:
                top_ids = axis2id[axis_idx]
                for id_ in top_ids:
                    x, y = proj_emb[id_]
                    if ax_idx == 0:
                        if y < 0:
                            continue
                    else:
                        if y > 0:
                            continue
                    ds.append(np.linalg.norm((x, y)))
            d_I_list[ax_idx].append(np.mean(ds))
            c_I_list[ax_idx].append(calc_c_I(picked_emb, normed_axistour_embed, topk))

    # Axis Tour
    print("Axis Tour")
    print(f"Avg. d_I: {np.mean(d_I_list[0]):.2f}")
    print(f"Avg. c_I: {np.mean(c_I_list[0]):.2f}")

    # Skew Sort
    print("Skew Sort")
    print(f"Avg. d_I: {np.mean(d_I_list[1]):.2f}")
    print(f"Avg. c_I: {np.mean(c_I_list[1]):.2f}")


if __name__ == "__main__":
    main()
