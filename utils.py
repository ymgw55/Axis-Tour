import logging
import numpy as np
import scipy.stats
from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy
from web.datasets.categorization import (
    fetch_AP,
    fetch_battig,
    fetch_BLESS,
    fetch_ESSLI_1a,
    fetch_ESSLI_2b,
    fetch_ESSLI_2c,
)
from web.datasets.similarity import (
    fetch_MEN,
    fetch_MTurk,
    fetch_RG65,
    fetch_RW,
    fetch_SimLex999,
    fetch_WS353,
)
from web.embedding import Embedding
from web.vocabulary import OrderedVocabulary


def pos_direct(vecs):
    vecs = vecs * np.sign(scipy.stats.skew(vecs, axis=0))
    return vecs


def get_logger(log_file=None, log_level=logging.INFO, stream=True):
    logger = logging.getLogger(__name__)
    handlers = []
    if stream:
        stream_handler = logging.StreamHandler()
        handlers.append(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(str(log_file), "w")
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    return logger


class MyEmbedding(Embedding):
    # override
    def __init__(self, vocab, vectors):
        super().__init__(vocab, vectors)

    @staticmethod
    def from_words_and_vectors(words, vectors):
        vocab = OrderedVocabulary(words)
        return MyEmbedding(vocab, vectors)


def split_range(p, dim):
    # Splits the range 0 to dim-1 into 'p' roughly equal parts
    # and returns the endpoints of each part.
    avg = dim // p
    remainder = dim % p

    ranges = []
    start = 0

    for i in range(p):
        # Calculate the end of the current segment
        end = start + avg + (i < remainder) - 1

        # Append the range endpoints
        ranges.append((start, end + 1))

        # Update the start for the next segment
        start = end + 1

    return ranges


def get_tasks():
    analogy_tasks = {"Google": fetch_google_analogy(), "MSR": fetch_msr_analogy()}

    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
    }

    categorization_tasks = {
        "AP": fetch_AP(),
        "BLESS": fetch_BLESS(),
        "Battig": fetch_battig(),
        "ESSLI_2c": fetch_ESSLI_2c(),
        "ESSLI_2b": fetch_ESSLI_2b(),
        "ESSLI_1a": fetch_ESSLI_1a(),
    }

    return analogy_tasks, similarity_tasks, categorization_tasks


def calc_c_I(picked_emb, normed_embed, topk):
    vecs = []
    _, length = picked_emb.shape
    for idx in range(length):
        topk_ids = np.argsort(picked_emb[:, idx])[-topk:]
        mean_emb = np.mean(normed_embed[topk_ids], axis=0)
        vecs.append(mean_emb)
    cossims = []
    for i in range(len(vecs) - 1):
        cossim = (
            np.dot(vecs[i], vecs[i + 1])
            / np.linalg.norm(vecs[i])
            / np.linalg.norm(vecs[i + 1])
        )
        cossims.append(cossim)
    return np.mean(cossims)


def test():
    print(split_range(3, 10))

    print(split_range(4, 10))
    print(split_range(5, 10))

    print(split_range(3, 11))
    print(split_range(3, 12))


if __name__ == "__main__":
    test()
