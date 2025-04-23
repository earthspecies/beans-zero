"""Custom pytorch tensor based metrics.

Examples
--------
>>> ap = AveragePrecision()
>>> output = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
>>> target = torch.tensor([[0, 1], [1, 0]])
>>> ap.update(output, target)
>>> ap.get_metric()
tensor([1., 1.])
"""

import math
import numpy as np
import torch

# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice
from beans_zero.external.spice.spice import Spice
from beans_zero.external.cider.cider import Cider


class AveragePrecision:
    """Computes the average precision for multilabel classification

    Taken from https://github.com/amdegroot/tnt

    Examples
    --------
    >>> ap = AveragePrecision()
    >>> output = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    >>> target = torch.tensor([[0, 1], [1, 0]])
    >>> ap.update(output, target)
    >>> ap.get_metric()
    tensor([1., 1.])
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Resets the meter with empty member variables"""
        self.scores = torch.tensor(
            torch.UntypedStorage(), dtype=torch.float32, requires_grad=False
        )
        self.targets = torch.tensor(
            torch.UntypedStorage(), dtype=torch.int64, requires_grad=False
        )
        self.weights = torch.tensor(
            torch.UntypedStorage(), dtype=torch.float32, requires_grad=False
        )

    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        """
        Updates the meter with new data

        Arguments
        ---------
        output: tensor
            NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model. The probabilities should
            sum to one over all classes
        target: tensor
            binary NxK tensor that encodes which of the K classes are associated
            with the N-th input (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)
        weight: tensor (optional)
            Nx1 tensor representing the weight for each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if isinstance(weight, np.ndarray):
                weight = torch.from_numpy(weight)
            elif isinstance(weight, list):
                weight = torch.Tensor(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert (
                output.dim() == 2
            ), "wrong output size (should be 1D or 2D with one column per class)"
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert (
                target.dim() == 2
            ), "wrong target size (should be 1D or 2D with one column per class)"

        if weight is not None:
            assert weight.dim() == 1, "Weight dimension should be 1"
            assert weight.numel() == target.size(
                0
            ), "Weight dimension 1 should be the same as that of target"
            assert torch.min(weight) >= 0, "Weight should be non-negative only"

        assert torch.equal(target**2, target), "targets should be binary (0 or 1)"
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(
                1
            ), "dimensions for output should match previously added examples."

        # make sure storage is of sufficient size
        if self.scores.untyped_storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.untyped_storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.untyped_storage().size() * 1.5)
            self.scores.untyped_storage().resize_(int(new_size + output.numel()))
            self.targets.untyped_storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.untyped_storage().resize_(
                    int(new_weight_size + output.size(0))
                )

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output.detach())
        self.targets.narrow(0, offset, target.size(0)).copy_(target.detach())

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

    def get_metric(self) -> torch.Tensor:
        """Returns the model's average precision for each class
        Returns
        -------
        ap: tensor
            1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0  # not a tensor ?

        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0) + 1).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(truth.sum(), 1)

        return ap


class BinaryF1Score:
    """Binary F1 score.

    This class is used to compute the F1 score for binary classification tasks.

    Examples
    --------
    >>> f1 = BinaryF1Score()
    >>> logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    >>> y = torch.tensor([1, 0])
    >>> f1.update(logits, y)
    >>> f1.get_metric()
    {'prec': 1.0, 'rec': 1.0, 'f1': 1.0}
    >>> f1 = BinaryF1Score(); logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    >>> y = torch.tensor([0, 1])
    >>> f1.update(logits, y)
    >>> f1.get_metric()
    {'prec': 0.0, 'rec': 0.0, 'f1': 0.0}
    """

    def __init__(self) -> None:
        self.num_positives = 0
        self.num_trues = 0
        self.num_tps = 0

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """Updates the metric with new data

        Arguments
        ---------
        logits : tensor
            Nx2 tensor that for each of the N examples
            indicates the logits of the example belonging to each of
            the 2 classes, according to the model.
            The positive class is the second column.

        y : tensor
            binary Nx1 tensor that encodes which of the 2 classes are associated
            with the N-th input (eg: a row [0, 1] indicates that the example is
            associated with the positive class)
        """
        assert logits.ndim == 2 and logits.size(1) == 2
        positives = logits.argmax(axis=1) == 1
        trues = y == 1
        tps = trues & positives
        self.num_positives += torch.sum(positives).cpu().item()
        self.num_trues += torch.sum(trues).cpu().item()
        self.num_tps += torch.sum(tps).cpu().item()

    def get_metric(self) -> dict[str, float]:
        """Returns the model's precision, recall, and F1 score based on
        the stored statistics

        Returns
        -------
        prec: float
            Precision of the model, on the current data
        rec: float
            Recall of the model
        f1: float
            F1 score of the model
        """
        prec = 0.0 if self.num_positives == 0 else self.num_tps / self.num_positives
        rec = 0.0 if self.num_trues == 0 else self.num_tps / self.num_trues
        if prec + rec > 0.0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        return {"prec": prec, "rec": rec, "f1": f1}

    def get_primary_metric(self) -> float:
        return self.get_metric()["f1"]


class MulticlassBinaryF1Score:
    """Multiclass binary F1 score for multi-label classification tasks.

    This class is used to compute the F1 score for multi-label classification tasks,
    where each example can belong to multiple classes.

    Examples
    --------
    >>> f1 = MulticlassBinaryF1Score(3)
    >>> logits = torch.tensor([[0., 9, 0], [9, 0., 7], [0., 0., 7]])
    >>> y = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 1]])
    >>> f1.update(logits, y)
    >>> f1.get_metric()
    {'macro_prec': 1.0, 'macro_rec': 1.0, 'macro_f1': 1.0}
    >>> logits = torch.tensor([[0.01, 9, 0.01], [9, 0.01, 7], [0.01, 0.01, 7]])
    >>> y = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 1]])
    >>> f1.update(logits, y)
    >>> metrics = f1.get_metric()
    >>> metrics["macro_prec"] < 1.0
    True
    >>> metrics["macro_f1"] < 1.0
    True
    """

    def __init__(self, num_classes: int) -> None:
        self.metrics = [BinaryF1Score() for _ in range(num_classes)]
        self.num_classes = num_classes

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """Updates the metric with new data

        Arguments
        ---------
        logits : tensor
            NxK tensor that for each of the N examples
            contains the unnormalized logits of the example belonging to each of
            the K classes. A torch.sigmoid is applied to the logits to get the
            probability that the example belongs to each class.

        y : tensor
            binary NxK tensor that encodes which of the K classes are associated.
            Multiple classes can be associated with each example.
            Eg: a row [1, 1, 0] indicates that the example is associated with
            classes 1 and 2.

        """
        probs = torch.sigmoid(logits)  # probability of a positive label
        for i in range(self.num_classes):
            # TODO: this step is unreliable. If the logit value for the
            # negative case (not detected)
            # is even a little higher than 0.0, it will be considered as
            # a positive case,
            binary_logits = torch.stack((1 - probs[:, i], probs[:, i]), dim=1)
            self.metrics[i].update(binary_logits, y[:, i])

    def get_metric(self) -> dict[str, float]:
        """Computes the macro-averaged precision, recall, and F1 score for all classes

        Returns
        -------
        macro_prec: float
            Macro-averaged precision of the model, on the current data
        macro_rec: float
            Macro-averaged recall of the model
        macro_f1: float
            Macro-averaged F1 score of the model
        """
        macro_prec = 0.0
        macro_rec = 0.0
        macro_f1 = 0.0
        for i in range(self.num_classes):
            metrics = self.metrics[i].get_metric()
            macro_prec += metrics["prec"]
            macro_rec += metrics["rec"]
            macro_f1 += metrics["f1"]
        return {
            "macro_prec": macro_prec / self.num_classes,
            "macro_rec": macro_rec / self.num_classes,
            "macro_f1": macro_f1 / self.num_classes,
        }

    def get_primary_metric(self) -> float:
        return self.get_metric()["macro_f1"]


class MeanAveragePrecision:
    """Average precision over classes for multilabel classification

    Examples
    --------
    >>> map = MeanAveragePrecision()
    >>> output = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    >>> target = torch.tensor([[0, 1], [1, 0]])
    >>> map.update(output, target)
    >>> map.get_metric()
    {'map': 1.0}
    """

    def __init__(self) -> None:
        self.ap = AveragePrecision()

    def reset(self) -> None:
        self.ap.reset()

    def update(
        self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None
    ) -> None:
        self.ap.update(output, target, weight)

    def get_metric(self) -> dict[str, float]:
        return {"map": self.ap.get_metric().mean().item()}

    def get_primary_metric(self) -> float:
        return self.get_metric()["map"]


def compute_spider(references: list[str], hypotheses: list[str]) -> float:
    """Compute the SPIDEr metric (SPICE + CIDEr)

    Arguments
    ---------
    references: list[str]
        List of reference captions ('true labels')
    hypotheses: list[str]
        List of hypothesis (predicted) captions

    Reference:
        # add ref

    Returns
    -------
    spider_score: float
        SPIDEr score for the captioning task

    Example
    -------
    >>> references = ["a cat is sitting on a mat", "a dog is running in the park"]
    >>> hypotheses = ["a cat is sitting on a mat", "a dog is running in the park"]
    >>> compute_spider(references, hypotheses)
    1.0
    """
    # Convert lists to dictionaries with IDs as keys
    refs = {f"image{i:08d}": [ref] for i, ref in enumerate(references)}
    hyps = {f"image{i:08d}": [hyp] for i, hyp in enumerate(hypotheses)}

    # Compute CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(refs, hyps)
    # NOTE: CIDEr score is multiplied by x10
    cider_score /= 10

    # Compute SPICE
    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(refs, hyps)

    # SPIDEr = (CIDEr + SPICE) / 2
    spider_score = (cider_score + spice_score) / 2

    return float(spider_score)
