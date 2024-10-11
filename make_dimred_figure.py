import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Make dimension reduction figure.")

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--fig_type", type=str, default="main")

    return parser.parse_args()


def main():
    args = parse_args()
    emb_type = args.emb_type
    assert emb_type in ("glove", "word2vec")
    fig_type = args.fig_type

    logger = get_logger()
    logger.info(f"emb_type: {emb_type}, fig_type: {fig_type}")

    input_dir = Path("output/eval_dimred")

    if fig_type == "main":
        main_csv_path = input_dir / f"{emb_type}_top100_main.csv"

        if not main_csv_path.exists():
            logger.info(f"{main_csv_path} does not exist.")
            run = [
                "python",
                "eval_dimred_main.py",
                "--emb_type",
                emb_type,
                "--topk",
                "100",
            ]
            logger.info(f'Run: {" ".join(run)}')
            subprocess.run(run)

        assert main_csv_path.exists()
        df = pd.read_csv(main_csv_path)
        emb_names = ("Original", "PCA", "randICA", "skewICA", "axisICA_curt")

        if emb_type == "glove":
            # polar
            polar_csv_path = input_dir / f"polar_{emb_type}.csv"
            if not polar_csv_path.exists():
                alreay_run = True
                for method_name in [
                    "rand_antonym_",
                    "orthogonal_antonymy_",
                    "variance_antonymy_",
                ]:
                    polar_embed_path = (
                        Path("output/polar_glove_embeddings")
                        / f"{method_name}gl_500_StdNrml.bin"
                    )
                    if not polar_embed_path.exists():
                        alreay_run = False
                        break
                if alreay_run:
                    logger.info(f"{polar_csv_path} does not exist.")
                    run = ["python", "eval_dimred_polar_glove.py"]
                    logger.info(f'Run: {" ".join(run)}')
                    subprocess.run(run)

            if polar_csv_path.exists():
                polar_df = pd.read_csv(polar_csv_path)
                df = pd.concat([df, polar_df], ignore_index=True)
                emb_names = (
                    "Original",
                    "PCA",
                    "randICA",
                    "skewICA",
                    "rand_antonym",
                    "variance_antonymy",
                    "orthogonal_antonymy",
                    "axisICA_curt",
                )

    elif fig_type == "alpha":
        main_csv_path = input_dir / f"{emb_type}_top100_main.csv"

        if not main_csv_path.exists():
            logger.info(f"{main_csv_path} does not exist.")
            run = [
                "python",
                "eval_dimred_main.py",
                "--emb_type",
                emb_type,
                "--topk",
                "100",
            ]
            logger.info(f'Run: {" ".join(run)}')
            subprocess.run(run)

        assert main_csv_path.exists()
        df = pd.read_csv(main_csv_path)
        emb_names = (
            "PCA",
            "axisICA_zero",
            "axisICA_curt",
            "axisICA_sqrt",
            "axisICA_one",
        )

    elif fig_type == "topk":
        main_csv_path = input_dir / f"{emb_type}_top100_main.csv"

        if not main_csv_path.exists():
            logger.info(f"{main_csv_path} does not exist.")
            run = [
                "python",
                "eval_dimred_main.py",
                "--emb_type",
                emb_type,
                "--topk",
                "100",
            ]
            logger.info(f'Run: {" ".join(run)}')
            subprocess.run(run)
        assert main_csv_path.exists()
        df = pd.read_csv(main_csv_path)

        topk_csv_path = input_dir / f"{emb_type}_topk.csv"
        if not topk_csv_path.exists():
            logger.info(f"{topk_csv_path} does not exist.")
            run = ["python", "eval_dimred_topk.py", "--emb_type", emb_type]
            logger.info(f'Run: {" ".join(run)}')
            subprocess.run(run)
        assert topk_csv_path.exists()
        topk_df = pd.read_csv(topk_csv_path)

        df = pd.concat([df, topk_df], ignore_index=True)
        emb_names = (
            "PCA",
            "axisICA_curt_top1",
            "axisICA_curt_top10",
            "axisICA_curt",
            "axisICA_curt_top1000",
        )

    elif fig_type == "projection":
        main_csv_path = input_dir / f"{emb_type}_top100_main.csv"

        if not main_csv_path.exists():
            logger.info(f"{main_csv_path} does not exist.")
            run = [
                "python",
                "eval_dimred_main.py",
                "--emb_type",
                emb_type,
                "--topk",
                "100",
            ]
            logger.info(f'Run: {" ".join(run)}')
            subprocess.run(run)
        assert main_csv_path.exists()
        df = pd.read_csv(main_csv_path)

        projection_csv_path = input_dir / f"{emb_type}_projection.csv"
        if not projection_csv_path.exists():
            logger.info(f"{projection_csv_path} does not exist.")
            run = ["python", "eval_dimred_projection.py", "--emb_type", emb_type]
            logger.info(f'Run: {" ".join(run)}')
            subprocess.run(run)
        assert projection_csv_path.exists()
        projection_df = pd.read_csv(projection_csv_path)

        df = pd.concat([df, projection_df], ignore_index=True)
        emb_names = (
            "PCA",
            "randICA",
            "randICA_curt",
            "skewICA",
            "skewICA_curt",
            "axisICA_curt",
        )

    elif fig_type == "tica":
        main_csv_path = input_dir / f"{emb_type}_top100_main.csv"

        if not main_csv_path.exists():
            logger.info(f"{main_csv_path} does not exist.")
            run = [
                "python",
                "eval_dimred_main.py",
                "--emb_type",
                emb_type,
                "--topk",
                "100",
            ]
            logger.info(f'Run: {" ".join(run)}')
            subprocess.run(run)
        assert main_csv_path.exists()
        df = pd.read_csv(main_csv_path)

        tica_csv_path = input_dir / f"{emb_type}_tica.csv"
        if not tica_csv_path.exists():
            logger.info(f"{tica_csv_path} does not exist.")
            run = ["python", "eval_dimred_tica.py", "--emb_type", emb_type]
            logger.info(f'Run: {" ".join(run)}')
            subprocess.run(run)
        assert tica_csv_path.exists()
        tica_df = pd.read_csv(tica_csv_path)

        df = pd.concat([df, tica_df], ignore_index=True)
        emb_names = (
            "PCA",
            "axisICA_curt",
            "TICA9",
            "TICA9_curt",
            "TICA75",
            "TICA75_curt",
        )

    ps = [1, 2, 5, 10, 20, 50, 100, 200, 300]
    results = {}
    task_types = ["analogy", "similarity", "categorization"]
    for task_type in task_types:
        task_df = df[df["task_type"] == task_type]
        emb2scores = {}
        for emb_name in emb_names:
            emb_df = task_df[task_df["emb_name"] == emb_name]
            scores = []
            for p in ps:
                p_df = emb_df[emb_df["p"] == p]
                if task_type == "analogy":
                    score = np.mean(p_df["top1-acc"].values)
                elif task_type == "similarity":
                    score = np.mean(p_df["spearman"].values)
                elif task_type == "categorization":
                    score = np.mean(p_df["purity"].values)
                scores.append(score)
            emb2scores[emb_name] = scores

        results[task_type] = emb2scores

    # plot
    fig, axes = plt.subplots(1, len(task_types), figsize=(27, 8))
    fig.subplots_adjust(
        left=0.1, right=0.95, bottom=0.05, top=0.88, wspace=0.1, hspace=0.1
    )
    ts = 40
    fs = 35
    ls = 28
    les = 21
    polar_color = "gray"

    for i, (task_type, emb2scores) in enumerate(results.items()):
        ax = axes[i]
        for emb_name, scores in emb2scores.items():
            linestyle = "-"
            marker = "o"

            if emb_name == "Original":
                color = "black"
                label = "Original"

            elif emb_name == "PCA":
                color = "blue"
                label = "PCA"

            elif emb_name == "randICA":
                color = "orange"
                label = "Random Order"
                if fig_type == "projection":
                    linestyle = "--"

            elif emb_name == "randICA_curt":
                color = "orange"
                label = "Random Order Projection"

            elif emb_name == "skewICA":
                color = "green"
                label = "Skewnes Sort"
                if fig_type == "projection":
                    linestyle = "--"

            elif emb_name == "skewICA_curt":
                color = "green"
                label = "Skewnes Sort Projection"

            elif emb_name == "axisICA_zero":
                color = "purple"
                label = r"Axis Tour ($\alpha=0$)"

            elif emb_name == "axisICA_curt":
                color = "red"
                if fig_type == "topk":
                    label = r"Axis Tour ($k=100$)"
                else:
                    label = r"Axis Tour ($\alpha=1/3$)"

            elif emb_name == "axisICA_sqrt":
                color = "cyan"
                label = r"Axis Tour ($\alpha=1/2$)"

            elif emb_name == "axisICA_one":
                color = "limegreen"
                label = r"Axis Tour ($\alpha=1$)"

            elif emb_name == "rand_antonym":
                color = polar_color
                label = "POLAR (Random)"
                linestyle = "--"
                marker = "s"

            elif emb_name == "variance_antonymy":
                color = polar_color
                label = "POLAR (Var. Max.)"
                linestyle = "-."
                marker = "^"

            elif emb_name == "orthogonal_antonymy":
                color = polar_color
                label = "POLAR (Ortho. Max.)"
                linestyle = ":"
                marker = "x"

            elif emb_name == "axisICA_curt_top1":
                color = "limegreen"
                label = r"Axis Tour ($k=1$)"

            elif emb_name == "axisICA_curt_top10":
                color = "gray"
                label = r"Axis Tour ($k=10$)"

            elif emb_name == "axisICA_curt_top1000":
                color = "sandybrown"
                label = r"Axis Tour ($k=1000$)"

            elif emb_name == "TICA9":
                color = "lime"
                label = "TICA9"
                linestyle = "--"

            elif emb_name == "TICA9_curt":
                color = "lime"
                label = "TICA9 Projection"

            elif emb_name == "TICA75":
                color = "gray"
                label = "TICA75"
                linestyle = "--"

            elif emb_name == "TICA75_curt":
                color = "gray"
                label = "TICA75 Projection"

            else:
                raise NotImplementedError

            if color == "red":
                ax.plot(
                    ps,
                    scores,
                    label=label,
                    marker="o",
                    linewidth=3,
                    markersize=10,
                    linestyle=linestyle,
                    color=color,
                    zorder=10,
                )
            else:
                ax.plot(
                    ps,
                    scores,
                    label=label,
                    marker=marker,
                    linewidth=3,
                    markersize=10,
                    linestyle=linestyle,
                    color=color,
                )

        ax.set_ylim(-0.05, 0.7)
        ax.set_yticks([0, 0.2, 0.4, 0.6])

        if task_type == "similarity":
            ax.set_title("Word Similarity", fontsize=ts, pad=15)
            ax.set_xlabel("Dimension", fontsize=fs, labelpad=10)
            ax.set_ylabel(r"Spearman's $\rho$", fontsize=fs)

        elif task_type == "analogy":
            ax.set_title("Analogy", fontsize=ts, pad=15)
            ax.set_xlabel("Dimension", fontsize=fs, labelpad=10)
            ax.set_ylabel("Top1 acc.", fontsize=fs)

        elif task_type == "categorization":
            ax.set_title("Categorization", fontsize=ts, pad=15)
            ax.set_xlabel("Dimension", fontsize=fs, labelpad=10)
            ax.set_ylabel("Purity", fontsize=fs)

        # x is log scale
        ax.set_xscale("log")
        if i == 0:
            ax.legend(loc="upper left", fontsize=les)

        # tick
        ax.tick_params(labelsize=ls, which="major", length=10)
        ax.tick_params(axis="x", which="minor", length=5)

    plt.tight_layout()
    output_dir = Path("output/images/dimred")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dimred_{emb_type}_{fig_type}.pdf"
    logger.info(f"Save figure to {output_path}")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
