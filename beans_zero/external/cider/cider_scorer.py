"""Cider Scorer implementation based on
https://github.com/salaniz/pycocoevalcap
"""

from typing import Union
import copy
from collections import defaultdict
import numpy as np
import math


def precook(s: str, n: int = 4) -> dict:
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.

    Arguments
    ---------
        s : str
            The string to be processed
        n : int
            The number of ngrams for which (ngram) representation is calculated

    Returns
    -------
        counts : dict
            A dictionary containing ngram counts
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] += 1
    return counts


# lhuang: oracle will call with "average"
def cook_refs(refs: list[str], n: int = 4) -> list[dict]:
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.

    Arguments
    ---------
        refs : list[str]
            List of reference sentences
        n : int
            The number of ngrams for which (ngram) representation is calculated
    Returns
    -------
        list[dict]
            A list of dictionaries containing ngram counts for each reference
    """
    return [precook(ref, n) for ref in refs]


def cook_test(test: str, n: int = 4) -> dict:
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.

    Arguments
    ---------
        test : str
            The test sentence to be processed
        n : int
            The number of ngrams for which (ngram) representation is calculated

    Returns
    -------
        dict
            A dictionary containing ngram counts for the test sentence
    """
    return precook(test, n)


class CiderScorer(object):
    """CIDEr scorer."""

    def copy(self) -> "CiderScorer":
        """Copy the refs.

        Returns
        -------
            CiderScorer
                A new CiderScorer instance with copied references
        """
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(
        self, test: str = None, refs: list[str] = None, n: int = 4, sigma: float = 6.0
    ) -> None:
        """singular instance"""
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test: list[str], refs: list[str]) -> None:
        """called by constructor and __iadd__ to avoid creating new instances."""

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))  # N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

    def size(self) -> int:
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (
            len(self.crefs),
            len(self.ctest),
        )
        return len(self.crefs)

    def __iadd__(self, other: Union[tuple, "CiderScorer"]) -> "CiderScorer":
        """Add an instance (e.g., from another sentence)
        or a tuple (test, refs) to this instance.
        This is used to accumulate scores.

        Arguments
        ---------
            other : CiderScorer or tuple
                An instance of CiderScorer or a tuple containing test and refs

        Returns
        -------
            CiderScorer
                The updated CiderScorer instance
        """

        if isinstance(other, tuple):
            # avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    def compute_doc_freq(self) -> None:
        """
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        """
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self) -> list[float]:
        def counts2vec(cnts: dict[str, int]) -> tuple[list[dict], np.ndarray, int]:
            """Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store
            mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.

            Arguments
            ---------
                cnts : dict
                    The ngram counts

            Returns
            -------
                vec : list[dict]
                    The vector representation of the ngram counts
                norm : np.ndarray
                    The norm of the vector
                length : int
                    The length of the ngram counts
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for ngram, term_freq in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram) - 1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                # compute norm for the vector.
                # the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(
            vec_hyp: list[dict],
            vec_ref: list[dict],
            norm_hyp: np.ndarray,
            norm_ref: np.ndarray,
            length_hyp: int,
            length_ref: int,
        ) -> np.ndarray:
            """Compute the cosine similarity of two vectors.

            Arguments
            ---------
                vec_hyp : list[dict]
                    The vector representation of the hypothesis
                vec_ref : list[dict]
                    The vector representation of the reference
                norm_hyp : np.ndarray
                    The norm of the hypothesis vector
                norm_ref : np.ndarray
                    The norm of the reference vector
                length_hyp : int
                    The length of the hypothesis
                length_ref : int
                    The length of the reference

            Returns
            -------
                list[float]
                    The similarity score for each ngram
            """
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for ngram, _ in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += (
                        min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                    )

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= norm_hyp[n] * norm_ref[n]

                assert not math.isnan(val[n])
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e ** (-(delta**2) / (2 * self.sigma**2))
            return val

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs, strict=False):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self) -> tuple[float, np.ndarray]:
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert len(self.ctest) >= max(self.document_frequency.values())
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)
