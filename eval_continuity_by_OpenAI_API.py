import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import pos_direct


def get_api_results(client, model, wordset, wordset_A, wordset_B):

    system_prompt = "You are an excellent NLP annotator. "\
        "Your response should be in JSON format with the key 'choice'."

    user_prompt = f"Which of the following words are related to "\
        f"the words [{', '.join(wordset)}]? Answer A or B.\n"\
        f"A. [{', '.join(wordset_A)}]\nB. [{', '.join(wordset_B)}]"

    # API call
    api_result = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
    )
    return api_result


def main():

    pca_ica_embed_path = 'output/pca_ica_embeddings/pca_ica_glove.pkl'
    with open(pca_ica_embed_path, 'rb') as f:
        _, ica_embed, words = pkl.load(f)
    ica_embed = pos_direct(ica_embed)

    axis_tour_path = 'LKH-3.0.6-glove/axistour.top100.txt'
    axistour_idx2ica_idx = []
    with open(axis_tour_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            idx = int(line[len('axis'):])
            axistour_idx2ica_idx.append(idx)
    axistour_embed = ica_embed[:, axistour_idx2ica_idx]
    normed_axistour_embed = axistour_embed / \
        np.linalg.norm(axistour_embed, axis=1, keepdims=True)

    skew = np.mean(ica_embed, axis=0)
    skew_idx2ica_idx = np.argsort(-skew)
    ica_idx2skew_idx = {ica_idx: idx for idx,
                        ica_idx in enumerate(skew_idx2ica_idx)}

    skew_embed = ica_embed[:, skew_idx2ica_idx]
    normed_skew_embed = skew_embed / \
        np.linalg.norm(skew_embed, axis=1, keepdims=True)

    _, dim = axistour_embed.shape

    client = OpenAI()
    output_dir = Path('output/continuity_by_OpenAI_API')
    models = ("gpt-3.5-turbo-0125",
              "gpt-4-turbo-2024-04-09",
              "gpt-4o-2024-05-13",
              "gpt-4o-mini-2024-07-18")

    for model in models:
        model_output_dir = output_dir / model
        model_output_dir.mkdir(parents=True, exist_ok=True)

    topk = 10
    data = []
    for axistour_idx in tqdm(list(range(dim))):
        ica_idx = axistour_idx2ica_idx[axistour_idx]
        next_axistour_idx = (axistour_idx + 1) % dim
        skew_idx = ica_idx2skew_idx[ica_idx]
        next_skew_idx = (skew_idx + 1) % dim

        axistour_topk_word_ids = np.argsort(
            -normed_axistour_embed[:, axistour_idx])[:topk]
        skew_topk_word_ids = np.argsort(
            -normed_skew_embed[:, skew_idx])[:topk]

        assert (axistour_topk_word_ids == skew_topk_word_ids).all()

        wordset = words[axistour_topk_word_ids]

        next_axistour_topk_word_ids = np.argsort(
            -normed_axistour_embed[:, next_axistour_idx])[:topk]
        next_skew_topk_word_ids = np.argsort(
            -normed_skew_embed[:, next_skew_idx])[:topk]

        wordset_A = words[next_axistour_topk_word_ids]
        wordset_B = words[next_skew_topk_word_ids]

        wordset = wordset.tolist()
        wordset_A = wordset_A.tolist()
        wordset_B = wordset_B.tolist()

        row = dict()
        row['wordset'] = wordset
        row['wordset_A'] = wordset_A
        row['wordset_B'] = wordset_B

        for model in models:
            output_path = output_dir / model /\
                f'axis{axistour_idx}_top{topk}words.pkl'
            if output_path.exists():
                with open(output_path, 'rb') as f:
                    api_result = pkl.load(f)
            else:
                api_result = get_api_results(client, model, wordset,
                                             wordset_A, wordset_B)
                with open(output_path, 'wb') as f:
                    pkl.dump(api_result, f)

            choice = eval(api_result.choices[0].message.content)['choice']
            row[model] = choice

        data.append(row)

    output_path = output_dir / f'top{topk}words.csv'
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    # plot bar chart
    As = []
    Bs = []
    for model in models:
        results = df[model]
        answer_types = results.unique()

        print(f'model: {model} - answer types: {answer_types}')

        # exact match, care A and B not to be mixed
        As.append(sum(df[model] == 'A'))
        Bs.append(sum(df[model] == 'B'))

    _, ax = plt.subplots(figsize=(13, 7))

    x = np.arange(len(models))
    fs = 27
    ls = 30
    legend_ls = 21
    width = 0.35

    rec1 = ax.bar(x - width/2, As, width, label='Axis Tour',
                  color='red', alpha=0.5)
    # different texture for gray image
    rec2 = ax.bar(x + width/2, Bs, width, label='Skewness Sort',
                  color='green', alpha=0.5, hatch='//')

    recs = [rec1, rec2]
    for rec in recs:
        for rect in rec:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=fs)

    ax.set_xticks(x)
    ax.set_xticklabels(['GPT3.5 Turbo', 'GPT4 Turbo', 'GPT4o', 'GPT4o-mini'])
    ax.set_ylim(0, 299)
    ax.tick_params(labelsize=ls)
    ax.legend(loc='upper left', fontsize=legend_ls)

    # adjust margin
    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.99)

    save_path = output_dir / f'GPTs_answer_top{topk}words.pdf'
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    main()
