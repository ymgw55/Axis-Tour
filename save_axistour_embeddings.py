import argparse
import os
import pickle as pkl
import shutil
import subprocess
from pathlib import Path

import numpy as np

from utils import get_logger, pos_direct


def parse_args():
    parser = argparse.ArgumentParser(description="Save Axis Tour embeddings.")

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--topk", type=int, default=100)
    return parser.parse_args()


def main():
    logger = get_logger()
    args = parse_args()
    logger.info(args)
    emb_type = args.emb_type
    topk = args.topk

    assert emb_type in ("glove", "word2vec", "bert")

    # copy LKH-3.0.6 to LKH-3.0.6-{emb_type}
    logger.info(f"copy LKH-3.0.6 to LKH-3.0.6-{emb_type}")
    LKH_dir = Path("LKH-3.0.6")
    LKH_emb_dir = Path(f"LKH-3.0.6-{emb_type}")
    if not LKH_emb_dir.exists():
        shutil.copytree(LKH_dir, LKH_emb_dir)

    # calculate axis embeddings
    logger.info("loading embeddings...")
    input_path = f"output/pca_ica_embeddings/pca_ica_{emb_type}.pkl"
    with open(input_path, "rb") as f:
        _, ica_embed, words = pkl.load(f)
    logger.info(f"ica_embed.shape: {ica_embed.shape}")
    _, dim = ica_embed.shape

    ica_embed = pos_direct(ica_embed)
    norm_ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)

    logger.info("computing axis embeddings...")
    topk_indices = [[] for _ in range(dim)]
    for axis_idx in range(dim):
        axis = norm_ica_embed[:, axis_idx]
        topk_indices[axis_idx] = np.argsort(axis)[-topk:]
    topk_indices = np.array(topk_indices)

    axis_embeds_path = LKH_emb_dir / f"axis_embeddings_top{topk}.txt"
    logger.info(f"saving axis embeddings to {axis_embeds_path}")
    with open(axis_embeds_path, "w") as f:
        for axis_idx in range(dim):
            idices = topk_indices[axis_idx]
            embeds = norm_ica_embed[idices]
            mean_embed = embeds.mean(axis=0)
            print(f"axis{axis_idx} ", file=f, end="")
            print(*mean_embed.tolist(), file=f)

    # compile makefile
    logger.info("compiling LKH...")
    subprocess.run(["make"])

    # create config file
    logger.info("creating config file...")
    config_path = LKH_emb_dir / f"wordtour.tsp.top{topk}"
    with open(config_path, "w") as f:
        subprocess.run(["./make_LKH_file", axis_embeds_path, str(dim)], stdout=f)

    # create parameter file
    logger.info("creating parameter file...")
    param_path = LKH_emb_dir / f"wordtour.par.top{topk}"
    with open(param_path, "w") as f:
        print(f"PROBLEM_FILE = wordtour.tsp.top{topk}", file=f)
        print("PATCHING_C = 3", file=f)
        print("PATCHING_A = 2", file=f)
        print("RUNS = 1", file=f)
        print(f"OUTPUT_TOUR_FILE = wordtour.out.top{topk}", file=f)

    # change directory to LKH_emb_dir
    logger.info(f"changing directory to {LKH_emb_dir}")
    os.chdir(LKH_emb_dir)

    # make clean
    logger.info("making clean...")
    subprocess.run(["make", "clean"])

    # make
    logger.info("making...")
    subprocess.run(["make"])

    # run LKH
    logger.info("running LKH...")
    subprocess.run(["./LKH", f"wordtour.par.top{topk}"])

    # change LKH_emb_dir to current directory
    logger.info(f"changing {LKH_emb_dir} to current directory...")
    os.chdir("..")

    # save axis tour results
    logger.info("saving axis tour results...")
    output_path = Path(f"LKH-3.0.6-{emb_type}/wordtour.out.top{topk}")
    axis_tour_path = Path(f"LKH-3.0.6-{emb_type}/axistour.top{topk}.txt")
    with open(output_path, "r") as f:
        lkh = f.readlines()
    axis_idxs = [f"axis{i}" for i in range(dim)]
    axis_idxs = [axis_idxs[i - 1] for i in map(int, lkh[6:-2])]
    with open(axis_tour_path, "w") as f:
        for axis_idx in axis_idxs:
            print(axis_idx, file=f)

    # save axis tour embeddings
    logger.info("saving axis tour embeddings...")
    axistour = []
    with open(axis_tour_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            idx = int(line[len("axis") :])
            axistour.append(idx)
    axistour_embed = ica_embed[:, axistour]

    # choose the best shift
    normed_axistour_embed = axistour_embed / np.linalg.norm(
        axistour_embed, axis=1, keepdims=True
    )
    vecs = []
    for axis_idx in range(dim):
        indices = np.argsort(normed_axistour_embed[:, axis_idx])[-topk:]
        top_emb = normed_axistour_embed[indices]
        mean_emb = top_emb.mean(axis=0)
        vecs.append(mean_emb)
    fisrt_vec = vecs[0]
    vecs.append(fisrt_vec)
    cos_sims = []
    for i in range(len(vecs) - 1):
        cos_sim = (
            np.dot(vecs[i], vecs[i + 1])
            / np.linalg.norm(vecs[i])
            / np.linalg.norm(vecs[i + 1])
        )
        cos_sims.append((cos_sim, i, i + 1))
    cos_sims.sort()
    _, _, shift = cos_sims[0]
    tmp = []
    for i in range(shift, shift + dim):
        tmp.append(axistour_embed[:, i % dim])
    axistour_embed = np.stack(tmp, axis=1)

    output_dir = Path("output/axistour_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    axistour_embed_path = output_dir / f"axistour_top{topk}_{emb_type}.pkl"
    logger.info(f"saving axis tour embeddings to {axistour_embed_path}")
    with open(axistour_embed_path, "wb") as f:
        pkl.dump((axistour_embed, words), f)

    logger.info("show top5 words for each axis:")
    normed_axistour_embed = axistour_embed / np.linalg.norm(
        axistour_embed, axis=1, keepdims=True
    )
    for idx in range(dim):
        top5_ids = np.argsort(-normed_axistour_embed[:, idx])[:5]
        print(f"axis{idx} {words[top5_ids]}")


if __name__ == "__main__":
    main()
