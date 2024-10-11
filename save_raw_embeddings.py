import argparse
import logging
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from gensim.models import KeyedVectors
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from wordfreq import word_frequency

from utils import get_logger

logger = get_logger(log_level=logging.INFO, stream=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"device: {device}")
logger.info("tokenizer loading...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
logger.info("model loading...")
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)
model.eval()


def bert_encode(model, x, attention_mask):
    with torch.no_grad():
        result = model(x.to(device), attention_mask=attention_mask.to(device))
    embeddings = result.last_hidden_state
    return embeddings


def truncate(tokens):
    if len(tokens) > tokenizer.model_max_length - 2:
        tokens = tokens[0 : (tokenizer.model_max_length - 2)]
    return tokens


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask


def collate_idf(arr, tokenize, numericalize, pad="[PAD]"):
    tokens = [["[CLS]"] + truncate(tokenize(a)) + ["[SEP]"] for a in arr]
    arr = [numericalize(a) for a in tokens]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)

    return padded, lens, mask, tokens


def get_bert_embedding(all_sens, model, tokenizer):
    padded_sens, lens, mask, token_lists = collate_idf(
        all_sens, tokenizer.tokenize, tokenizer.convert_tokens_to_ids
    )

    with torch.no_grad():
        torch_embeddings = bert_encode(model, padded_sens, attention_mask=mask)

    numpy_embeddings = torch_embeddings.cpu().numpy()
    numpy_lens = lens.cpu().numpy()

    embeddings = []
    for embedding, len_, token_list in zip(numpy_embeddings, numpy_lens, token_lists):
        embedding = embedding[:len_]
        assert len(embedding) == len(token_list)
        embeddings.append(embedding)

    return embeddings, token_lists


def save_embed_token(sents, output_path, word_num, batch_size=512):
    logger.info(f"batch_size: {batch_size}")
    all_embeddings = []
    all_tokens = []
    all_sents = []
    n = 0
    for batch_start in tqdm(range(0, len(sents), batch_size)):
        batch_sents = sents[batch_start : batch_start + batch_size]
        try:
            batch_embeddings, batch_tokens = get_bert_embedding(
                batch_sents, model, tokenizer
            )

            embeddings = []
            tokens = []
            sentences = []
            for e, t, s in zip(batch_embeddings, batch_tokens, batch_sents):
                embeddings.extend(e)
                tokens.extend(t)
                sentences.extend([s] * len(t))

            if n + len(embeddings) >= word_num:
                embeddings = embeddings[: word_num - n]
                tokens = tokens[: word_num - n]
                sentences = sentences[: word_num - n]
            n += len(embeddings)

            all_embeddings += embeddings
            all_tokens += tokens
            all_sents += sentences

        except RuntimeError as e:
            logger.error(f"{e}, skip this batch")
            continue

        if n >= word_num:
            break

    all_embeddings = np.array(all_embeddings)
    all_tokens = np.array(all_tokens)
    all_sents = np.array(all_sents)
    logger.info(f"all_embeddings.shape: {all_embeddings.shape}")
    logger.info(f"all_tokens.shape: {all_tokens.shape}")
    logger.info(f"all_sents.shape: {all_sents.shape}")
    assert (
        all_embeddings.shape[0] == all_tokens.shape[0] == all_sents.shape[0] == word_num
    )

    token_count = defaultdict(int)
    token_cs = []
    for token in all_tokens:
        c = token_count[token]
        token_c = f"{token}_{c}"
        token_cs.append(token_c)
        token_count[token] += 1
    all_tokens = np.array(token_cs)

    embed_token = (all_embeddings, all_tokens, all_sents)

    with open(output_path, "wb") as f:
        pkl.dump(embed_token, f)


def loadFile(data_path):
    with open(data_path) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def bert(word_num):
    data_path = Path(
        "data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"
    )  # noqa
    logger.info("loading data...")
    sents = loadFile(data_path)
    logger.info(f"number of sentences: {len(sents)}")
    output_path = "output/raw_embeddings/raw_bert.pkl"
    save_embed_token(sents, output_path, word_num)


def word2vec(word_num):
    model_file = "data/embeddings/word2vec/GoogleNews-vectors-negative300.bin"

    model = KeyedVectors.load_word2vec_format(model_file, binary=True)
    words = list(model.vocab.keys())
    fws = []
    checked = set()
    for word in tqdm(words):
        if word.lower() in checked:
            continue
        checked.add(word.lower())
        freq = word_frequency(word, "en")
        fws.append((freq, word))
    fws.sort(reverse=True)
    words = [fw[1] for fw in fws[:word_num]]
    embeddings = []
    for word in tqdm(words):
        embeddings.append(model[word])
    words = [word.lower() for word in words]
    assert len(words) == len(set(words))
    words = np.array(words)
    embeddings = np.array(embeddings)
    logger.info(f"embeddings.shape: {embeddings.shape}")

    output_path = "output/raw_embeddings/raw_word2vec.pkl"
    with open(output_path, "wb") as f:
        pkl.dump((embeddings, words), f)


def parse_args():
    parser = argparse.ArgumentParser(description="Save raw embeddings.")

    parser.add_argument("--emb_type", type=str, default="glove")
    parser.add_argument("--word_num", type=int, default=40000)

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)
    emb_type = args.emb_type
    word_num = args.word_num

    output_dir = Path("output/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    if emb_type == "bert":
        bert(word_num)
    elif emb_type == "word2vec":
        word2vec(word_num)
    else:
        raise ValueError(f"emb_type: {emb_type}")


if __name__ == "__main__":
    main()
