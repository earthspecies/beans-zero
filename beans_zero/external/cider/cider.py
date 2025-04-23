"""CIDEr module entrypoint"""

# Description: Describes the class to compute the CIDEr
# (Consensus-Based Image Description Evaluation) Metric
# by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
import numpy as np
from .cider_scorer import CiderScorer


class Cider:
    """Main Class to compute the CIDEr metric

    CIDEr is a metric for evaluating image captioning systems.

    Arguments
    ---------
    n : int
        The maximum n-gram order to consider. Default is 4.
    sigma : float
        The standard deviation parameter for the Gaussian penalty. Default is 6.0.
    """

    def __init__(self, n: int = 4, sigma: float = 6.0) -> None:
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(
        self, gts: dict[str, list[str]], res: dict[str, list[str]]
    ) -> tuple[float, np.ndarray]:
        """Main function to compute CIDEr score

        Arguments
        ----------
        gts: dict[str, list[str]]
            Ground truth dictionary with key <image> and value
            <reference sentence>
        res: dict[str, list[str]]
            Predictions dictionary with key <image> and value <tokenized hypothesis
            / candidate sentence>

        Returns
        -------
        tuple[float, np.ndarray]
            CIDEr score and the per-sample scores
        """

        assert gts.keys() == res.keys()
        imgIds = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0

            cider_scorer += (hypo[0], ref)

        return cider_scorer.compute_score()

    def method(self) -> str:
        return "CIDEr"
